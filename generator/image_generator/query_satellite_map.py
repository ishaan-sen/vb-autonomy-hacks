# save as gee_exact_bbox.py, then: python gee_exact_bbox.py
import ee, requests, os
from io import BytesIO

# --- Initialize (optionally: ee.Initialize(project='your-project-id')) ---
ee.Initialize(project = 'map-screenshot-474419')

# --- Your exact bbox (lon/lat, EPSG:4326) ---
x_left  = -86.846293
x_right = -86.756262
y_bottom = 36.124444
y_top    = 36.174170

region = ee.Geometry.Rectangle([x_left, y_bottom, x_right, y_top],
                               proj='EPSG:4326', geodesic=False)

# --- Try NAIP (≈1 m, USA). Fallback to Sentinel-2 (10 m) ---
def get_naip():
    ic = (ee.ImageCollection('USDA/NAIP/DOQQ')
          .filterBounds(region)
          .filterDate('2015-01-01', '2030-01-01')
          .sort('system:time_start', False))  # newest first
    if ic.size().getInfo() == 0:
        return None
    # Mosaic and select RGB; clip to exact region
    return ic.mosaic().select(['R','G','B']).clip(region)

def get_s2():
    def mask(img):
        qa = img.select('QA60')
        cloud = 1 << 10
        cirrus = 1 << 11
        m = qa.bitwiseAnd(cloud).eq(0).And(qa.bitwiseAnd(cirrus).eq(0))
        return img.updateMask(m).divide(10000)
    ic = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(region)
          .filterDate('2024-01-01', '2025-12-31')
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
          .map(mask))
    # median composite, select RGB, clip exact region
    return ic.median().select(['B4','B3','B2']).clip(region)

img = get_naip() or get_s2()
use_naip = img.bandNames().getInfo()[0] in ['R','G','B']
vis = {'bands': (['R','G','B'] if use_naip else ['B4','B3','B2']),
       'min': (0 if use_naip else 0.03),
       'max': (255 if use_naip else 0.3),
       **({} if use_naip else {'gamma':1.1})}

task = ee.batch.Export.image.toDrive(
    image=img.visualize(**vis),
    description='bbox_satellite_nativePNG',
    fileNamePrefix='bbox_satellite_native',
    region=region,
    scale=(1 if use_naip else 10),  # native pixel size
    crs='EPSG:4326',
    fileFormat='GEO_TIFF',
    maxPixels=1e13
)

task.start()
print(task.status())
print("Drive export started (full native scale). Open Tasks in EE or check your Google Drive.")