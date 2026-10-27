"""Blind review page: 2018 / 2022 aerial + LiDAR height-change chip per case, radio buttons, CSV export.

The page links each case to Atlas by street address, so it is for local use only (gitignored).
"""

import base64
import io
from urllib.parse import quote

import pandas as pd
from PIL import Image

import ortho_chips as O
from dh_chip import dh_chip


def b64(img, fmt="JPEG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def types(r, rear_col="chg_rear"):
    return ", ".join(t for t, c in [("new", r.chg_new), ("pop-up", r.chg_popup), ("rear addition", r[rear_col]),
                                    ("demolition", r.chg_demo)] if c) or "none"


def build(items, title, key, intro="", rear_col="chg_rear", export_name="review_labels.csv", years=(2018, 2022),
          dh=None, role_names=None) -> str:
    """items: rows with item, role, unit geometry, address and the unit features / change flags.
    years: the two aerial years; a row may carry its own (y0, y1) columns. dh(row) -> height-change chip (default:
    the per-ZIP 2018 -> 2022 rasters). role_names: label shown per role."""
    role_names = role_names or {"precision": "precision sample", "unpermitted": "unpermitted candidate"}
    cards = []
    for _, r in items.iterrows():
        ys = (int(r.y0), int(r.y1)) if "y0" in r.index else years
        aerial = O.strip(r.geometry, years=ys, size=300)
        d = dh(r) if dh else dh_chip(r.geometry, zip_code=str(r.zip) if "zip" in r.index else "19146")
        row = Image.new("RGB", (aerial.width + 6 + d.width, 300), (255, 255, 255))
        row.paste(aerial, (0, 0))
        row.paste(d, (aerial.width + 6, 0))
        q = (f"<fieldset><legend>Change?</legend>"
             f"<label><input type=radio name='{r['item']}_chg' value='real'> Real change</label>"
             f"<label><input type=radio name='{r['item']}_chg' value='none'> No change</label>"
             f"<label><input type=radio name='{r['item']}_chg' value='unsure'> Unsure</label>"
             f"<input type=text name='{r['item']}_chg_note' placeholder='if no change: why (shadow, tree, edge, misregistration…)'></fieldset>")
        if r.role == "unpermitted":
            q += (f"<fieldset><legend>Permit history in Atlas</legend>"
                  f"<label><input type=radio name='{r['item']}_permit' value='no_permit'> Confirmed no permit</label>"
                  f"<label><input type=radio name='{r['item']}_permit' value='found'> Found a permit (match failure)</label>"
                  f"<label><input type=radio name='{r['item']}_permit' value='unsure'> Unsure</label>"
                  f"<input type=text name='{r['item']}_permit_note' placeholder='if found: why it was missed'></fieldset>")
        cards.append(f"""<section class=card id="{r['item']}"><h2>{r['item']} · {role_names.get(r.role, r.role)}
    · detected: <b>{types(r, rear_col)}</b></h2>
    <img src="data:image/jpeg;base64,{b64(row)}" alt="{r['item']} aerial {ys[0]}, {ys[1]} and height change">
    <div class=cap><span>aerial {ys[0]}</span><span>aerial {ys[1]}</span><span>ΔnDSM {ys[1]}−{ys[0]} (red = higher, ±6 m)</span></div>
    <p class=meta>cover {r.cover18:.0%} → {r.cover22:.0%} · rising cells {0 if pd.isna(r.frac_up) else r.frac_up:.0%} ·
    new area outside 2017 footprint {r.new_outside_fp_m2:.0f} m² ·
    <a href="https://atlas.phila.gov/{quote(str(r.address))}" target="_blank" rel="noopener">Atlas: {r.address}</a></p>{q}</section>""")

    html = f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
    <title>{title}</title><style>
    :root{{--bg:#fafafa;--fg:#1b1f23;--muted:#5b6770;--card:#fff;--line:#dde3e8;--accent:#1f5f8b}}
    @media (prefers-color-scheme:dark){{:root{{--bg:#15181b;--fg:#e6e9ec;--muted:#9aa5ae;--card:#1e2226;--line:#2e353b;--accent:#6aa9d8}}}}
    body{{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;margin:0 auto;max-width:1000px;padding:16px}}
    .card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin:14px 0}}
    .card h2{{margin:0 0 8px;font-size:15px}} .card img{{width:100%;height:auto;display:block;border-radius:4px}}
    .cap{{display:grid;grid-template-columns:1fr 1fr 1fr;color:var(--muted);font-size:12px;text-align:center}}
    .meta{{color:var(--muted);font-size:13px}} a{{color:var(--accent)}}
    fieldset{{border:1px solid var(--line);border-radius:6px;margin:6px 0;display:flex;gap:14px;flex-wrap:wrap;align-items:center}}
    input[type=text]{{flex:1;min-width:220px;padding:4px 6px;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:4px}}
    #bar{{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center;z-index:2}}
    button{{background:var(--accent);color:#fff;border:0;border-radius:6px;padding:8px 14px;cursor:pointer}}
    </style></head><body>
    <h1>{title} ({len(items)} cases)</h1>
    <p>{intro} For every case mark whether the outlined parcel really changed between the two images shown.
    For <b>unpermitted candidates</b>, open Atlas, check the permit history for 2015–2022, and mark whether a permit exists.
    Press <b>Export</b> and save <code>{export_name}</code> next to this file. Local use only: contains addresses.</p>
    <div id=bar><button id=exp>Export {export_name}</button><span id=count></span></div>
    {''.join(cards)}
    <script>
    const KEY="{key}";let saved={{}};try{{saved=JSON.parse(localStorage.getItem(KEY)||"{{}}")}}catch(e){{}}
    const inputs=[...document.querySelectorAll("input")];
    function collect(){{const a={{}};inputs.forEach(i=>{{if(i.type==="radio"){{if(i.checked)a[i.name]=i.value}}else if(i.value)a[i.name]=i.value}});return a}}
    function update(){{const a=collect();try{{localStorage.setItem(KEY,JSON.stringify(a))}}catch(e){{}}
    const done=[...document.querySelectorAll(".card")].filter(c=>a[c.id+"_chg"]).length;document.getElementById("count").textContent=done+" / {len(items)} labelled";return a}}
    inputs.forEach(i=>{{const v=saved[i.name];if(v!==undefined){{if(i.type==="radio")i.checked=(v===i.value);else i.value=v}}i.addEventListener("change",update);i.addEventListener("input",update)}});update();
    const esc=s=>'"'+String(s||"").replace(/"/g,'""')+'"';
    document.getElementById("exp").onclick=()=>{{const a=update();const ids=[...document.querySelectorAll(".card")].map(c=>c.id);
    const csv="item,change,change_note,permit,permit_note\\n"+ids.map(i=>[i,a[i+"_chg"],a[i+"_chg_note"],a[i+"_permit"],a[i+"_permit_note"]].map(esc).join(",")).join("\\n")+"\\n";
    const el=document.createElement("a");el.href=URL.createObjectURL(new Blob([csv],{{type:"text/csv"}}));el.download="{export_name}";el.click()}};
    </script></body></html>"""
    return html
