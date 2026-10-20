# %% [markdown]
## Full-resolution NDVI and vegetation cover, processed in windows
# Based on Lab 3-2 (bands, NDVI, classification, GeoTIFF export) and
# Lab 3-3 (Window and polygon masking). Restart the kernel first, then run
# from this section: the existing clipped TIFF is sufficient. Do not rerun
# the earlier full-array mosaic/clip cells when resuming this section.
# Only map previews are downsampled; all statistics use original pixels.

# %%
from pathlib import Path
import math
import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio.windows import Window
from rasterio.features import geometry_mask, geometry_window
from rasterio.errors import WindowError
from shapely.geometry import mapping
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

# Work from either the notebook folder or the repository root.
work = Path.cwd()
if not (work / 'sf_boundary').exists():
    work = work / 'assignment' / 'assignment_1'
data = work / 'data'
data.mkdir(exist_ok=True)
naip_file = data / 'maskSF-naip.tif'
ndvi_file = data / 'ndviSF.tif'
veg_file = data / 'vegSF.tif'
city_file = work / 'sf_boundary' / 'sf_place_tl_2024_06.shp'
threshold = 0.1
block_size = 1024
ndvi_nodata = -9999.0
veg_nodata = 255  # 0 = non-vegetation, 1 = vegetation; never use 0 as NoData.

def windows_in(width, height, col_start=0, row_start=0):
    # Lab 3-3 Window approach, including partial right and bottom edges.
    for row in range(row_start, row_start + height, block_size):
        for col in range(col_start, col_start + width, block_size):
            yield Window(col, row,
                         min(block_size, col_start + width - col),
                         min(block_size, row_start + height - row))

city = gpd.read_file(city_file)
with rio.open(naip_file) as src:
    if src.crs is None or src.count < 4:
        raise ValueError('The input needs a CRS and four RGB+NIR bands.')
    city_proj = city.to_crs(src.crs)  # Lab 2: transform vectors, not raster.
    city_shapes = [mapping(g) for g in city_proj.geometry]
    print('Raster:', src.crs, src.width, src.height, 'resolution:', src.res)

# %% [markdown]
### Calculate and save NDVI and vegetation together
# Read Red=1 and NIR=4 as float32, as in Lab 3-2. Use NDVI > 0.1.
# A city mask is reapplied because the earlier clip may not have saved a
# NoData value. Exclude source masks and zero denominators; retain valid
# NDVI=0 and negative NDVI. Water is included if it has valid observations.
# Outputs may take time and disk space; memory is limited to small windows.

# %%
with rio.Env(GDAL_CACHEMAX=128 * 1024 * 1024):
    with rio.open(naip_file) as src:
        # Construct a single-band profile; do not inherit RGB photometric settings.
        profile = dict(driver='GTiff', width=src.width, height=src.height,
                       count=1, crs=src.crs, transform=src.transform,
                       tiled=True, blockxsize=512, blockysize=512,
                       compress='lzw', BIGTIFF='YES')
        total = math.ceil(src.width / block_size) * math.ceil(src.height / block_size)
        with rio.open(ndvi_file, 'w', **profile, dtype='float32',
                      nodata=ndvi_nodata) as ndvi_dst, \
             rio.open(veg_file, 'w', **profile, dtype='uint8',
                      nodata=veg_nodata) as veg_dst:
            for index, window in enumerate(windows_in(src.width, src.height), 1):
                red = src.read(1, window=window, masked=True).astype('float32')
                nir = src.read(4, window=window, masked=True).astype('float32')
                inside = geometry_mask(city_shapes, out_shape=red.shape,
                                       transform=src.window_transform(window),
                                       invert=True, all_touched=False)
                denominator = nir.data + red.data
                valid = (inside & ~np.ma.getmaskarray(red) & ~np.ma.getmaskarray(nir)
                         & np.isfinite(denominator) & (denominator > 0))
                ndvi = np.full(red.shape, ndvi_nodata, dtype='float32')
                np.divide(nir.data - red.data, denominator, out=ndvi, where=valid)
                veg = np.full(red.shape, veg_nodata, dtype='uint8')
                veg[valid] = (ndvi[valid] > threshold).astype('uint8')
                ndvi_dst.write(ndvi, 1, window=window)
                veg_dst.write(veg, 1, window=window)
                if index % 100 == 0 or index == total:
                    print(f'Processed {index}/{total} windows')
            ndvi_dst.update_tags(description='NDVI=(NIR-Red)/(NIR+Red); NAIP 2022')
            veg_dst.update_tags(threshold=str(threshold), classes='0=non-vegetation;1=vegetation;255=NoData')
print('Saved:', ndvi_file, veg_file)

# %% [markdown]
### Raster map previews
# Lab 3-1 out_shape approach: preview only, not the source of tract statistics.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for path, ax, title, cmap, low, high in [
    (ndvi_file, axes[0], 'San Francisco NDVI (NAIP 2022)', 'RdYlGn', -1, 1),
    (veg_file, axes[1], 'Vegetation (NDVI > 0.1)', ListedColormap(['#dddddd', '#238b45']), 0, 1)
]:
    with rio.open(path) as src:
        scale = max(src.width, src.height) / 1600
        preview = src.read(1, out_shape=(max(1, int(src.height / max(1, scale))),
                                         max(1, int(src.width / max(1, scale)))), masked=True)
    im = ax.imshow(preview, cmap=cmap, vmin=low, vmax=high)
    ax.set_title(title)
    ax.set_axis_off()
    cb = fig.colorbar(im, ax=ax, shrink=0.65)
    if path == veg_file:
        cb.set_ticks([0, 1], labels=['Non-vegetation', 'Vegetation'])
fig.tight_layout()
fig.savefig(data / 'sf_raster_maps.png', dpi=180, bbox_inches='tight')
plt.show()
plt.close(fig)

# %% [markdown]
### Census tract boundaries
# Official 2024 TIGER/Line California tracts, matching the city boundary vintage.
# Source: https://www2.census.gov/geo/tiger/TIGER2024/TRACT/tl_2024_06_tract.zip
# This cell downloads the ZIP once (about 31 MB). It does not download NAIP.
# Select San Francisco County (STATEFP=06, COUNTYFP=075), reproject with
# Lab 2 to_crs(), and intersect with the city boundary for city-only statistics.

# %%
import urllib.request

tract_zip = data / 'tl_2024_06_tract.zip'
if not tract_zip.exists():
    url = 'https://www2.census.gov/geo/tiger/TIGER2024/TRACT/tl_2024_06_tract.zip'
    temporary = tract_zip.with_suffix('.zip.part')
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(tract_zip)  # A failed download is not mistaken for a complete ZIP.
tracts = gpd.read_file('zip://' + tract_zip.resolve().as_posix())
tracts = tracts.loc[(tracts['STATEFP'] == '06') & (tracts['COUNTYFP'] == '075')].copy()
tracts = tracts.to_crs(city_proj.crs)
city_geometry = city_proj.geometry.union_all()
tracts.geometry = tracts.geometry.intersection(city_geometry)
tracts = tracts.loc[~tracts.geometry.is_empty & tracts.geometry.area.gt(0)].copy()
tracts = tracts.reset_index(drop=True)
print('City-intersecting tracts:', len(tracts))

# %% [markdown]
### Tract-level mean NDVI and vegetation percentage
# Retain Lab 3-3's polygon-by-polygon accumulation and output attributes.
# For each tract, read its bounding window in 1024-pixel subwindows and mask
# by polygon (pixel-center rule). This also handles large coastal tracts.
# Vegetation cover = vegetation pixels / valid observed pixels * 100.
# Both statistics exclude NoData; tracts with no observations remain NaN.
# valid_frac reports observed pixels / tract pixels inside the raster extent;
# it is NOT a whole-tract coverage fraction for tracts extending beyond the raster.

# %%
tracts['mean_ndvi'] = np.nan
tracts['veg_pct'] = np.nan
tracts['valid_n'] = np.int64(0)
tracts['zone_n'] = np.int64(0)
tracts['valid_frac'] = np.nan

with rio.Env(GDAL_CACHEMAX=128 * 1024 * 1024):
    with rio.open(ndvi_file) as src:
        for index, feature in tracts.iterrows():
            geom = mapping(feature.geometry)
            try:
                zone_window = geometry_window(src, [geom])
            except WindowError:
                continue
            count = 0
            vegetation_count = 0
            zone_count = 0
            ndvi_sum = 0.0
            for window in windows_in(int(zone_window.width), int(zone_window.height),
                                     int(zone_window.col_off), int(zone_window.row_off)):
                values = src.read(1, window=window, masked=True)
                inside = geometry_mask([geom], out_shape=values.shape,
                                       transform=src.window_transform(window),
                                       invert=True, all_touched=False)
                valid = inside & ~np.ma.getmaskarray(values) & np.isfinite(values.data)
                pixels = values.data[valid]
                zone_count += int(inside.sum())
                count += pixels.size
                ndvi_sum += pixels.sum(dtype='float64')
                vegetation_count += int(np.count_nonzero(pixels > threshold))
            tracts.loc[index, 'valid_n'] = count
            tracts.loc[index, 'zone_n'] = zone_count
            if zone_count:
                tracts.loc[index, 'valid_frac'] = count / zone_count
            if count:
                tracts.loc[index, 'mean_ndvi'] = ndvi_sum / count
                tracts.loc[index, 'veg_pct'] = 100 * vegetation_count / count
            if (index + 1) % 25 == 0 or index + 1 == len(tracts):
                print(f'Processed {index + 1}/{len(tracts)} tracts')

tracts.to_file(data / 'sf_tract_vegetation.gpkg', layer='tracts', driver='GPKG')
tracts[['GEOID', 'mean_ndvi', 'veg_pct', 'valid_n', 'zone_n', 'valid_frac']].to_csv(
    data / 'sf_tract_vegetation.csv', index=False)
display(tracts[['GEOID', 'mean_ndvi', 'veg_pct', 'valid_n', 'valid_frac']].head())
print('Tracts with no valid pixels:', int(tracts['mean_ndvi'].isna().sum()))

# %% [markdown]
### Census tract maps
# Lab 2 GeoPandas plot(), colored by the statistics calculated above.
# Gray tracts have no valid observations; water is not separately excluded.

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 8))
for ax, column, title, cmap, low, high in [
    (axes[0], 'mean_ndvi', 'Mean NDVI', 'RdYlGn', -1, 1),
    (axes[1], 'veg_pct', 'Vegetation cover (%)', 'YlGn', 0, 100)
]:
    tracts.plot(column=column, ax=ax, cmap=cmap, vmin=low, vmax=high,
                edgecolor='#555555', linewidth=0.25, legend=True,
                legend_kwds={'shrink': 0.6, 'label': title},
                missing_kwds={'color': '#dddddd', 'label': 'No valid data'})
    ax.set_title('San Francisco: ' + title)
    ax.set_axis_off()
fig.suptitle('NAIP 2022 | Census tracts 2024 | Vegetation: NDVI > 0.1')
fig.text(0.5, 0.02, 'City-clipped tracts; valid observed pixels only. Water included; gray = no data.',
         ha='center', fontsize=9)
fig.tight_layout(rect=[0, 0.04, 1, 0.95])
fig.savefig(data / 'sf_tract_maps.png', dpi=200, bbox_inches='tight')
plt.show()
plt.close(fig)
