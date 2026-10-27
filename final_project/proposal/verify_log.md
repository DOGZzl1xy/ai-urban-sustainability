# Fact-check log for the proposal

Checked 2026-10-22. Each item gives the claim, what was found, the source, and what changes in the proposal.

## 1. ⚠️ Philadelphia flew LiDAR again in 2025

- **Claim in draft:** five LiDAR collections (2008, 2010, 2015, 2018, 2022), "the latest is from March 2022".
- **Found:** PASDA has `phillyLiDAR/2025/LAS/`, posted 2026-01-12 and 2026-01-15.
  - 963 tiles, 389.6 GB citywide. Tiles are 2,640 ft squares named by their lower-left corner, the same scheme as 2018.
- **Flight date, from the files themselves.** PASDA's summary page (dataset 7158) and the BTAA record both say "April 2022" and repeat the 2022 description. The file contents show that this metadata was copied over:
  - `26479E196152N.las` (the smallest tile, 169 KB, downloaded in full and read with laspy):
    - Header creation date 2025-07-15; written by LAStools las2las.
    - `gps_time_type` = 1 (adjusted standard GPS time).
    - All 5,628 points fall between **2025-04-18 16:05 and 16:34 UTC**.
  - `26479E204072N.las` (first 4 MB): header day 196 of 2025; median GPS time **2025-04-18**.
  - Check on the method: the same conversion on the 2022 tiles gives 2022-03-29/30 (`qc.md`), which matches that flight.
  - The tile naming (lower-left corner in ft, like 2018) and the upload dates (2026-01-12/15) also differ from 2022 (`Philadelphia_NNN.las`).
  - The `2025/` folder has only `LAS/` and `Contours/`. There is **no `Metadata_and_Reports/` acquisition report** (2022 has one).
- **Checked on the 17 tiles covering 19146 (2026-10-22):**
  - Every tile was flown on 2025-04-18 (GPS time).
  - **Units:** the header has no CRS. x/y are PA South US ft (same tile corners as 2018). z is US ft too: 2025 ground points have a median of 31.43 against 31.19 in the 2018 terrain model at the same cells (ratio 1.008).
  - **Density** (noise and overlap classes dropped): median **29.6 pts/m²** (range 14.9–44.0), against 43.1 in 2018 and 38.6 in 2022. That is about 70% of 2018, not "half". The first sample tile (≈17) was a sparse edge tile. Proposal corrected.
  - **Classes:** fully classified, unlike 2022. One urban tile: 1 unclassified 22%, 2 ground 35%, 3–5 vegetation 16%, **6 building 22%**, 7 noise 1%, 9 water 2%, 17 bridge 1%.
  - ⚠️ **Truncated files on PASDA:** `26822E227832N.las` (19146) and `26954E227832N.las` (19147) are shorter than their headers say. The server's Content-Length equals the truncated size (modified 2026-01-12), so the files on the server are incomplete; this is not a download error. Only 7.09 M of 11.99 M points are readable in the first. `rasters.read_points` keeps every complete point, and the coverage maps show the gap. Worth reporting to PASDA.
- **Volume per ZIP (2025):** 19146: 17 tiles, 10.0 GB · 19147: 9.6 GB · 19125: 6.8 GB · 19132: 12.5 GB · 19121: 12.8 GB · 19143: 20.3 GB.
- **Sources:** https://www.pasda.psu.edu/download/phillyLiDAR/2025/LAS/ ; https://www.pasda.psu.edu/uci/DataSummary.aspx?dataset=7158 ; https://geo.btaa.org/catalog/pasda-7158 (listing saved to `data/raw/lidar/pasda_LAS2025_listing.html`)
- **Consequence / decision (2026-10-22):** the draft's motivation ("LiDAR stops in 2022") no longer holds. Decision: keep the model and reframe it.
  - LiDAR 2018 → 2022 → 2025 is the main measurement.
  - The imagery model fills the years in between (2019, 2020, 2023, 2024) and later years.
  - The model trains on 2018 only and is tested on both 2022 and 2025.
  - Only study-area tiles are downloaded (19146 + 19147: 36 tiles, about 20 GB).

## 2. City Council restrictions on pop-ups (`[VERIFY]` in §1)

- **Found:** Bill No. 230574 (Council President Darrell Clarke).
  - Creates the Brewerytown–Sharswood overlay in Lower North Philadelphia.
  - Caps residential building height at the tallest building on an abutting lot. Where there is none, the cap is 35 ft or the block majority height.
  - Bans roof decks.
  - Passed Council 2023-11-16. Mayor Kenney returned it unsigned on 2023-11-28.
- **Sources:**
  - Inquirer, 2023-11-16: https://www.inquirer.com/real-estate/housing/darrell-clarke-zoning-overlay-brewerytown-sharswood-height-limit-20231116.html
  - Mayor's return letter on phlcouncil.com: https://phlcouncil.com/document/november-28-2023-to-the-president-and-members-of-the-council-of-the-city-of-philadelphia-i-am-hereby-returning-unsigned-bill-number-230574-passed-by-city-council-on-november-16-2023-by-councilmem/
- **Related but not used:**
  - Councilmember Kenyatta Johnson's 2018 overlay bill for Point Breeze / Grays Ferry, which would bar new three-story homes next to shorter neighbours (WHYY, 2018-11-02). I could not confirm the bill number or whether it passed in that form, so it is not cited.
  - A 2015 rowhouse height bill was dropped (Inquirer).
- **Change:** replace `[VERIFY]` with the Bill 230574 citation.

## 3. Aerial imagery 2023–2025 (and the series)

- **City tile services** (ArcGIS Online, Web Mercator, tiles to level 22) exist for:
  2008 (3in), 2009 (6in), 2010 (3in), 2012 (3in), 2014 (6in), 2015–2020 (3in), 2022 (2in), 2023 (name has no resolution suffix), 2024 (3in), 2025 (3in).
- **Missing under this naming:** 2011, 2013, 2021.
- **PASDA:** the 2018, 2023 and 2024 summaries say the imagery was captured in **April–May**, from nadir images, "at various resolutions". All are downloadable as TIFF tiles (PASDA datasets 2019 / 7000 / 7001, 2025 tile index pasda-7157), and the tile service can also be cut into tiles directly.
- **Resolution check:** fine-detail energy (mean |Laplacian| on the same 9 tiles in 19146 at level 21) is 2008: 6.1, 2010: 6.9, 2015: 10.5, 2018: 10.4, 2022: 8.4, 2023: 11.4, 2024: 7.1, 2025: 7.2. No year is clearly coarser than 3 in, except that 2008/2010 look softer. All years are far finer than the 0.3–0.5 m model input.
- **Change:** the draft says "about 15 cm" and "gaps: 2013, 2021". Correct this to "3-inch (≈ 7.5 cm; 2022 is 2-inch), April–May, gaps 2011, 2013, 2021 (2009 and 2014 are 6-inch)".

## 4. 2008 LiDAR point density

- **Found:** PASDA metadata `LAS2008/PAPhiladelphiaLiDAR2008.xml`: Optech ALTM Gemini 3100 (Pictometry), "nominal post spacing of 1 meter", about 1 pt/m², flown April 2008. That is about 40× sparser than 2018.
- **Decision:** leave 2008 out of training. At 1 pt/m², a 0.5 m nDSM is mostly interpolation, and roof edges would be blurred labels.

## 5. DINOv3 satellite weights

- **Models:** two SAT-493M models: ViT-7B/16 (trained from scratch) and **ViT-L/16, 300M parameters, distilled from the 7B**. SAT-493M is 493 M 512×512 Maxar RGB chips at 0.6 m.
  - https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m
  - https://huggingface.co/facebook/dinov3-vit7b16-pretrain-sat493m
- **Access:** gated on Hugging Face. The account must accept Meta's DINOv3 license; weights also come by request from Meta's download form.
- **License:** the DINOv3 License grants a non-exclusive, royalty-free license to use, modify and create derivative works, which covers fine-tuning. Research use is allowed. Publications must acknowledge DINOv3, and there are acceptable-use restrictions (military etc.). https://ai.meta.com/resources/models-and-libraries/dinov3-license/
- **GPU memory estimate for ViT-L/16** (local GPU: RTX 5060 Ti, 16 GB):
  - A 512×512 input gives 1,024 patch tokens.
  - Frozen backbone + head, bf16: the weights take 0.6 GB, so batch 16 fits easily.
  - Fine-tuning the last 4 of 24 blocks with AdamW (fp32 master weights for those blocks ≈ 50M params × 16 B ≈ 0.8 GB), plus activations for those blocks (with gradient checkpointing): batch 8 at 512² should fit in < 12 GB.
  - Full fine-tuning (300M × 16 B ≈ 4.8 GB of optimizer state + activations) is possible at batch 2–4 with checkpointing, but is not planned.
  - The 7B model does not fit locally and is not needed.
- **Conclusion:** the pilot runs locally.

## 6. CHMv2 R² 0.86

- **Found:** Meta AI blog, 2026-03-10, "Canopy Height Maps v2" with WRI. R² rose "from 0.53 to 0.86" after switching the DINOv2 backbone to DINOv3 pretrained on SAT-493M, together with more and better airborne-LiDAR training data.
  - https://ai.meta.com/blog/world-resources-institute-dino-canopy-height-maps-v2/
- **Caveats for the citation:**
  - The 0.86 measures tree canopy height, not buildings.
  - The gain comes from both the backbone and the data, not the backbone alone.
  - A technical report is on arXiv (2603.06382), not yet read in full.

## 7. Numbers in §4 vs result files

| Proposal | Result file | Status |
|---|---|---|
| 18,182 parcel units | summary.md | ✓ |
| Recall new 96.6% [94.7, 98.3] | summary.md (n = 417) | ✓ (add n) |
| Recall upward 87.1% [80.6, 93.5] | summary.md (n = 93) | ✓ (add n) |
| Precision 85.1% [74.5, 95.7], "50 random detections" | review_scores.json: 47 judged, 3 unsure | change to "40 of 47 judged real (3 unsure)" |
| 456 new, 398 pop-ups | summary.md | ✓ |
| Every pop-up had a permit on own / adjacent parcel | summary.md (0 of 398) | ✓ |
| 4 changes without any permit | review_scores.json | ✓ |
| Errors only from the rear rule (cars and trees) | review labels: also 1 footprint misalignment, 1 unexplained | soften to "almost all from cars and trees" |
| Permits 93–98% geocoded to parcels per year, 2014–2025 | qc.md checks 2014–2022 only (92.9–98.5%) | change years to 2014–2022 or check 2023–2025 |

## 8. What changed from the received draft (`proposal_v0_received.md` → `proposal.md` / `proposal.pdf`)

1. **2025 LiDAR**
   - Six collections, latest April 2025.
   - The motivation changed from "extend beyond 2022" to "LiDAR gives three snapshots; imagery dates the change and fills the years in between".
   - RQ1 now tests on 2022 **and** 2025.
   - RQ2 adds *when*; RQ3 adds time to completion.
2. **`[VERIFY]` pop-up restriction** replaced with Bill No. 230574 (2023, Brewerytown–Sharswood), reference [3].
3. **Data table**
   - LiDAR years now list 2025; 2008 is marked as not used (about 1 pt/m²).
   - Imagery is 3-inch, not "about 15 cm"; 2022 is 2-inch; captured April–May; gaps are 2011, 2013 and 2021, not 2013 and 2021.
   - Permit geocoding is now stated as checked for 2014–2022.
4. **Stage 1**: 2022→2025 added; notes that the 2025 density is about half and the rules will be re-checked.
5. **Stage 2**: the backbone is named (DINOv3 ViT-L/16, SAT-493M, 300 M params); training schedule added (frozen first, then the last blocks).
6. **Data splits**
   - Training is 2018 (2015/2010 added later only if their density allows); the draft had 2010/2015/2018 (+2008).
   - Test years are 2022 and 2025.
   - The prediction set is 2019, 2020, 2023, 2024 and later.
7. **Stage 3** now covers timing (the first year the predicted height steps up). Evaluation (3) is now timing against permit dates.
8. **§4 numbers**
   - Precision now reads "40 of 47 judged real (3 unsure)" instead of implying all 50.
   - Recall now gives n (417 / 93).
   - "Errors came only from the rear rule" softened to "almost all".
   - §4 was shortened to fit 2 pages: the change and permit bullets were merged.
9. **Expected outcomes**: the interactive map was folded into "Maps"; a transferability point was added.
10. **Risks** shortened. **References** added: CHMv2 blog, DINOv3 arXiv, Bill 230574, L&I permits, PASDA.
11. **Figure 1**
    - (a) ZIP 19146 on a 100 m grid: pop-ups and new buildings in two colours, sized by count.
    - (b) One permitted pop-up, aerial 2018/2022 plus the LiDAR profile. It was chosen by fixed criteria: full lot built in 2018, footprint unchanged, a two-story start and a one-story rise.
    - No address appears. Script: `scripts/make_fig1.py`.

PDF: `xelatex proposal.tex` (Times New Roman 10.5 pt everywhere, including caption and references); 2 pages.

## 9. Second revision (2026-10-22, after the height-model pilot)

1. **§4 precision:** now the fresh sample, 95.5% [88.6, 100] (42 of 44 judged real). Counting all 6 unsure cases as errors gives a lower bound of 84%. The first rule set's 85.1% is kept as the v1 result.
2. **§4 Stage 2 pilot added:**
   - DINOv3 + DPT building-level MAE 1.17–1.60 m, best in all four test sets.
   - From 2022 to 2025 in 19146, its error goes 1.17 → 1.54 m; the U-Net's goes 1.26 → 2.10 m.
   - New construction 92% / 91%; pop-ups 44% / 44%; the model sees a rise of 1.9 m where LiDAR measures 3.8 m.
3. **§3 restructured:**
   - Stage 2 is now the single-image model (pilot done).
   - Stage 3 is the bi-temporal model in two variants, A (two images) and B (LiDAR-anchored), chosen because single images miss pop-ups.
   - Training labels are 2018→2022 changes in the training ZIPs.
   - Tests are 2018→2022 (space) and 2022→2025 (time and space, primary).
   - Thresholds are now set on the validation blocks.
4. **RQ1** now asks whether LiDAR anchoring makes vertical additions detectable.
5. **Data table:**
   - LiDAR row lists the three anchor flights and the 2025 flight date.
   - Earlier flights are marked sparse. The 2015 figure (≈3 pts/m²) was measured on one 19132 tile including the overlap class.
   - The last two rows were merged to save space.
6. **Risks** rewritten for the new design. Text tightened throughout to stay at 2 pages.

## 10. Third revision (2026-10-23, citywide phase)

Changes:
- **Data:** LiDAR is framed as three anchors (2018, 2022, 2025), with 2010 and 2015 for extra training pairs. The 2025 flight date (18 April 2025, from point GPS time) and the wrong PASDA metadata are stated in the text.
- **Method:** the LiDAR-anchored bi-temporal DINOv3 unit classifier. Vertical additions in anchor years are measured by LiDAR; the model dates grew / new construction in 2019, 2020, 2023 and 2024. The dropped variants are named in one sentence.
- **RQ1** now asks how far the model goes and at which spatial scale it is reliable.
- **Scope:** residential and low/mid mixed use, 152.5 km² (`results/citywide/study_area.md`).
- **Pilot results** are now a table: LiDAR recall 96.6% / 87.1%, fresh precision 95.5%, single-image MAE 1.17–1.60 m, round-1 final test grid r 0.83–0.88 with totals within 10%, pop-up AP 0.16 as a limitation.
- **Next step:** citywide training and a round-2 test on 8 sealed ZIPs.
- **Remaining work** is now RQ2, RQ3 and the interactive map.

Numbers checked against:
- `results/popup_feasibility/summary.md`: recall n = 417 / 93; precision 42 / 44.
- `results/height_model_pilot/summary.md`: MAE 1.17–1.60 m; rise 1.86 vs 3.82 m, written 1.9 vs 3.8 m.
- `results/bitemporal/summary.md`:
  - final test grid r for grew: 0.83 at 250 m, 0.88 at block group;
  - for new construction: 0.87 and 0.88;
  - totals 345 vs 369 (−6.5%) and 175 vs 163 (+7.5%);
  - pop-up AP 0.159.
- `results/citywide/study_area.md`: 152.5 km².

PDF: 2 pages, every text span 10.5 pt or larger.

## 11. Fourth revision (2026-10-26, after citywide processing)

- **LiDAR density** in the data table: "30–43 pts/m²" → **"15–50 pts/m²"**. The old range was measured on 19146 tiles.
  Citywide, mean returns per m² inside the study area (2018 / 2022): 19146 46 / 42, 19104 40 / 53, 19143 16 / 47,
  19154 15 / 50, 19115 15 / 24, 19111 17 / 33, 19124 17 / 24, 19128 21 / 34. The 2018 flight is about 15–20 pts/m²
  outside the centre, which is why 2018 periods cover fewer units (`results/citywide/change_summary_trainval.csv`).
- **Vegetation:** "and point classes in 2025, flown as leaves emerged" added (`results/citywide/veg2025_fix.md`).
- **Remaining work, item 1:** citywide LiDAR (five years) and imagery (nine years) are now processed.
- Rebuilt with xelatex: still 2 pages, no overfull boxes.
