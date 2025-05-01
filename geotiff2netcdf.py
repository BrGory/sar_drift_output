# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 11:10:23 2025

@author: Owner
"""
def convert(user_args, gdf_points, gdf_lines):
    import os
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.plot import plotting_extent
    import numpy as np
    import xarray as xr
    from pyproj import Transformer
    
    # 1. Input GeoTIFF path
    geotiff_path = os.path.normpath(user_args['geotiff_filename'])
    
    # 1. Load raw pixel values from GeoTIFF
    with rasterio.open(geotiff_path) as src:
        geotiff_data = src.read(1)
        height, width = geotiff_data.shape
        extent = plotting_extent(src)
        
        transform = src.transform
        height, width = src.height, src.width
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
        xs = np.array(xs, dtype='float64').reshape((height, width))
        ys = np.array(ys, dtype='float64').reshape((height, width))
    
    # # 2. Convert known top-left lon/lat to EPSG:3413
    # top_left_lon = -135.89
    # top_left_lat = 73.60
    # transformer = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
    # top_left_x, top_left_y = transformer.transform(top_left_lon, top_left_lat)
    
    # # 3. Build the affine transform
    # pixel_size = 1000  # 1 km resolution
    # affine = from_origin(top_left_x, top_left_y, pixel_size, pixel_size)
    
    # # 4. Build x and y coordinate arrays from affine
    # cols, rows = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
    # xs, ys = affine * (cols, rows)
    
    # 5. Create DataArray with proper geolocation
    data_array = xr.DataArray(
        geotiff_data,
        dims=("y", "x"),
        coords={
            "x": (("y", "x"), xs),
            "y": (("y", "x"), ys)
        },
        attrs={
            "long_name": "Normalized Radar Cross Section",
            "units": "dB"
        }
    )
    
    
    # Sanity Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Raw GeoTIFF
    axes[0].imshow(geotiff_data, cmap='gray')
    axes[0].set_title('Original GeoTIFF (pixel space)')
    axes[0].axis('off')
 
    
    # Plot 2: DataArray with coordinates
    # data_array.plot(ax=axes[1], cmap='gray', robust=True)
    axes[1].imshow(
        geotiff_data,
        cmap='gray',
        origin='upper',
        extent=extent
        )
    axes[1].set_title('DataArray (polar stereographic meters)')
    
    # Reproject vectors to EPSG:3413
    gdf_points_3413 = gdf_points.to_crs(epsg=3413)
    gdf_lines_3413 = gdf_lines.to_crs(epsg=3413)
    
    # Overlay vectors (no need for Cartopy if using x/y coords)
    gdf_points_3413.plot(ax=axes[1], color='red', markersize=20, label='Points')
    gdf_lines_3413.plot(ax=axes[1], color='blue', linewidth=1, label='Lines')
    
    plt.tight_layout()
    plt.show()
    exit()
    
    # show the x.y coordinate in meters
    # fig, ax = plt.subplots()
    # data_array.plot(ax=ax, cmap='gray', robust=True)
    
    # def onclick(event):
    #     if event.inaxes == ax:
    #         x_click, y_click = event.xdata, event.ydata
    #         print(f"Clicked at x={x_click:.2f} meters, y={y_click:.2f} meters")
    
    # fig.canvas.mpl_connect('button_press_event', onclick)
    # plt.show()
    
    
    # 9. Wrap into a Dataset and define CRS metadata following CF Conventions
    # ds = data_array.to_dataset(name='nrcs')
    # ds['crs'] = xr.DataArray(0, attrs={
    #     'grid_mapping_name': 'polar_stereographic',
    #     'straight_vertical_longitude_from_pole': -45.0,
    #     'latitude_of_projection_origin': 90.0,
    #     'standard_parallel': 70.0,
    #     'false_easting': 0.0,
    #     'false_northing': 0.0,
    #     'semi_major_axis': 6378137.0,
    #     'inverse_flattening': 298.257223563,
    #     'epsg_code': 'EPSG:3413'
    # })
    
    ds = data_array.to_dataset(name='nrcs')
    ds['crs'] = xr.DataArray(0, attrs={'epsg_code': 'EPSG:3413'})
    
    # 10. Optional: Global attributes
    ds.attrs['title'] = 'SAR-derived NRCS Product'
    ds.attrs['institution'] = 'Generated via Python processing'
    ds.attrs['source'] = 'GeoTIFF file from ESA Sentinel-1B SAR'
    ds.attrs['history'] = 'Converted from GeoTIFF to NetCDF with corrected geolocation'
    ds.attrs['references'] = 'https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-1'
    
    # 11. Optional: Set encoding (compression)
    encoding = {
        'nrcs': {
            'zlib': True,
            'complevel': 5,
            'shuffle': True,
            'chunksizes': (512, 512),
            'dtype': 'uint8',
        }
    }
    
    # 12. Save as NetCDF
    output_netcdf = os.path.join(user_args['output_dir'], 'from_geotiff.nc')
    ds.to_netcdf(output_netcdf, encoding=encoding)
    
    
    print("✅ NetCDF created successfully at:", output_netcdf)
    
    
def read_geotiff(user_args):
    """
        Affine(
            1000.0,     0.0,   -1788091.7358,  # a, b, c
            0.0,    -1000.0,      27777.4326   # d, e, f
        )

        x=1000⋅col+0⋅row+(−1788091.74)
        y=0⋅col+(−1000)⋅row+27777.43

    Parameters
    ----------
    user_args : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    import os
    import numpy as np
    from rasterio.transform import xy
    from rasterio import open as rio_open
    import rasterio 
    
    geotiff_path = os.path.normpath(user_args['geotiff_filename'])
    with rio_open(geotiff_path) as src:
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width
    print(transform)    
        
    
    # Get the (x, y) coordinates for every pixel center
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
    
    
def vectors_on_geotiff(user_args, gdf_points, gdf_lines):
    import os
    import matplotlib.pyplot as plt
    import rasterio
    import geopandas as gpd
    import cartopy.crs as ccrs
    
    geotiff_path = os.path.normpath(user_args['geotiff_filename'])
    with rasterio.open(geotiff_path) as src:
        data = src.read(1)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    
    # Set Cartopy projection
    crs_proj = ccrs.epsg(3413)
    
    # Load your real vector data here (replace with your actual GeoDataFrames)
    gdf_point = gpd.GeoDataFrame(geometry=gpd.points_from_xy([-135.89], [73.60]), crs="EPSG:4326")
    gdf_line = gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_wkt(["LINESTRING (-136 73.5, -135.5 73.7)"]), crs="EPSG:4326")
    
    # Reproject vectors to match raster CRS
    gdf_point_3413 = gdf_point.to_crs(epsg=3413)
    gdf_line_3413 = gdf_line.to_crs(epsg=3413)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': crs_proj})
    ax.imshow(data, extent=extent, transform=crs_proj, origin='upper', cmap='gray')
    gdf_point_3413.plot(ax=ax, transform=crs_proj, color='red', markersize=50)
    gdf_line_3413.plot(ax=ax, transform=crs_proj, color='blue', linewidth=2)
    ax.set_title("GeoTIFF with Projected Vectors Overlay")
    ax.gridlines(draw_labels=True)
    plt.show()

    
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
from rasterio.crs import CRS
import geopandas as gpd
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
import os
from rasterio.transform import Affine

def plot_shapes_on_geotiff(geotiff_path, gdf_points, gdf_lines):
    """
    Robust function to plot geographic shapes on GeoTIFF with proper alignment.
    Handles:
    - Missing CRS in GeoTIFF
    - CRS mismatches
    - Missing geotransforms
    """
    # First handle the GeoTIFF's coordinate system
    temp_path = None
    try:
        with rasterio.open(geotiff_path) as src:
            # Case 1: GeoTIFF has no CRS but we know it's WGS84
            if src.crs is None or src.transform.is_identity:
                print("GeoTIFF has no CRS or geotransform - creating proper referencing")
                
                # Create a temporary file with proper georeferencing
                temp_path = geotiff_path + '.fixed'
                height, width = src.shape
                
                # Calculate transform based on vector data extent
                xmin, ymin, xmax, ymax = gdf_points.total_bounds
                transform = Affine.translation(xmin, ymax) * Affine.scale(
                    (xmax - xmin) / width,
                    -(ymax - ymin) / height
                )
                
                meta = src.meta.copy()
                meta.update({
                    'transform': transform,
                    'crs': CRS.from_epsg(4326)
                })
                
                with rasterio.open(temp_path, 'w', **meta) as dst:
                    dst.write(src.read())
                
                geotiff_path = temp_path

    except Exception as e:
        print(f"Error processing GeoTIFF: {e}")
        return

    # Now create the plot
    try:
        with rasterio.open(geotiff_path) as src:
            fig, ax = plt.subplots(figsize=(15, 15))
            
            # Ensure vector data has proper CRS
            if gdf_points.crs is None:
                gdf_points = gdf_points.set_crs(src.crs)
            if gdf_lines.crs is None:
                gdf_lines = gdf_lines.set_crs(src.crs)
                
            # Transform vectors if needed
            if gdf_points.crs != src.crs:
                gdf_points = gdf_points.to_crs(src.crs)
            if gdf_lines.crs != src.crs:
                gdf_lines = gdf_lines.to_crs(src.crs)

            # Show raster
            show(src, ax=ax, cmap='gray')
            
            # Plot vectors
            gdf_lines.plot(ax=ax, color='cyan', linewidth=2, alpha=0.8, label='Drift Paths')
            gdf_points.plot(ax=ax, color='red', markersize=100, alpha=0.9, label='Start Points',
                          edgecolor='yellow', linewidth=1)
            
            # Add coordinate display
            def format_coord(x, y):
                return f"Lon: {x:.5f}, Lat: {y:.5f}"
            ax.format_coord = format_coord
            
            # Add map elements
            ax.grid(True, linestyle=':', alpha=0.5)
            ax.legend()
            plt.title(f"SAR Drift Visualization\nCRS: {src.crs}")
            
            plt.show()
            
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)