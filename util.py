# -*- coding: utf-8 -*-
"""
******************************************************************************

 Project:    SAR Drift Output Generator
 Purpose:    Utility functions for sar_drift_output
 Author:     Brendon Gory, brendon.gory@noaa.gov
                           brendon.gory@colostate.edu
             Data Science Application Specialist (Research Associate II)
             at CSU CIRA
 Supervisor: Dr. Prasanjit Dash, prasanjit.dash@noaa.gov
                               prasanjit.dash@colostate.edu
             CSU CIRA Research Scientist III
             (Program Innovation Scientist)
******************************************************************************
Copyright notice
         NOAA STAR SOCD and Colorado State Univ CIRA
         2025, Version 1.0.0
         POC: Brendon Gory (brendon.gory@noaa.gov)

 Permission is hereby granted, free of charge, to any person obtaining a
 copy of this software and associated documentation files (the "Software"),
 to deal in the Software without restriction, including without limitation
 the rights to use, copy, modify, merge, publish, distribute, sublicense,
 and/or sell copies of the Software, and to permit persons to whom the
 Software is furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included
 in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 DEALINGS IN THE SOFTWARE.
"""

def error_msg(msg, rc):
    print(f"  ⚠️ {msg}")
    exit(rc)
    
def compute_distance_meters(x1, y1, x2, y2, precision):
    """
    Compute planar bearing (clockwise from north) and Euclidean distance
    between two points in a projected CRS (e.g., EPSG:3413).
    
    Parameters:
        x1 (float): Starting longitude
        y1 (float): Starting latitude
        x2 (float): Ending longitude
        y2 (float): Ending latitude
        precision: Round significant digits
        
    Returns:
        distance (float): Computed Euclidean distance
    """
    import numpy as np
    
    dx = x2 - x1
    dy = y2 - y1

    # Distance in kilometers --> np.hypot=(dx^2+dy^2)^.5
    distance = np.round(np.hypot(dx, dy) / 1000, precision)

    return distance

    
def read_geotiff_rasterio(geotiff_path):
    """
    Reads a GeoTIFF image using GCP-based reprojection to EPSG:3413
    (NSIDC Sea Ice Polar Stereographic North) and returns a masked
    array with coordinate information.
    
    This function:
        - Opens a GeoTIFF file using rasterio
        - Extracts Ground Control Points (GCPs) to reproject the image
        to a target CRS (EPSG:3413)
        - Uses nearest-neighbor resampling to regrid the data
        - Constructs an xarray.DataArray with spatial coordinates in meters
        - Masks background values (zeros) to allow clean visualization
        - Computes the image extent for use in plotting (e.g., with imshow)
    
    Parameters:
        geotiff_path (str): Path to the input GeoTIFF file containing
                            GCPs and raster data.
    
    Returns:
        tuple:
            masked_xr (np.ma.MaskedArray): Masked 2D array of image data
                                           with background set to NaN.
            extent (list): [xmin, xmax, ymin, ymax] extent of the image
                           in meters (EPSG:3413) for use with plotting.
    Coauthor:
        Rachael Lazzaro, rachel.lazzaro@noaa.gov
    """
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.warp import calculate_default_transform
    import xarray as xr
    import numpy as np
   

    with rasterio.open(geotiff_path) as src:
        gcps, gcps_crs = src.get_gcps()
        dst_crs = "EPSG:3413"
        dst_transform, width, height = calculate_default_transform(
            gcps_crs, dst_crs, src.width, src.height, gcps=gcps
        )

        dst_array = np.empty(
            (src.count, height, width), 
            dtype=src.dtypes[0]
        )

        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array[0],
            src_crs=gcps_crs,
            src_transform=None, # None triggers GCP-based warping
            gcps=gcps,          # Let rasterio warp based on GCPs
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest
        )
        
        # Construct xarray.DataArray with coordinates
        x_coords = dst_transform[2] + dst_transform[0] * np.arange(width)
        y_coords = dst_transform[5] + dst_transform[4] * np.arange(height)
        
        geotiff_xr = xr.DataArray(
            dst_array[0],
            dims=("y", "x"),
            coords={"x": x_coords, "y": y_coords},
            attrs={"crs": dst_crs}
        )
        
        # change backround to white
        masked_xr = np.ma.masked_equal(geotiff_xr.values, 0)
        
        extent = [
            dst_transform[2],
            dst_transform[2] + dst_transform[0] * width,
            dst_transform[5] + dst_transform[4] * height,
            dst_transform[5],
        ]

    return masked_xr, extent


def set_metadata(user_args, cdl_file):
    """
    Generate a NetCDF metadata template using a CDL file and load
    it as an xarray.Dataset.
    
    This function takes the user-defined CDL (Common Data Language)
    file path from the `user_args` dictionary, runs the `ncgen` 
    command-line tool to convert it into a NetCDF (.nc) file,
    and then loads that file into memory using `xarray`.
    
    The function is typically used to extract metadata
    (attributes and structure) from a CDL file so that it can be applied
    to a data-driven NetCDF file.
    
    Parameters:
        user_args (dict): Dictionary containing user-provided arguments,
        including:
            - 'metadata_dir' (str): Path to the directory where CDL file
                                    is stored.
    
    Returns:
        xarray.Dataset: A dataset containing only metadata
                        from the generated NetCDF file.
    
    Raises:
        SystemExit: If the `ncgen` command fails or
                    returns a non-zero status code.
    """


    import os
    import glob
    import subprocess
    from datetime import datetime
    import xarray as xr
    
    # clear previous .nc files
    metadata_dir = user_args['metadata_dir']
    for nc_file in glob.glob(os.path.join(metadata_dir, '*.nc')):
        os.remove(nc_file)
        
    # Prepare ncgen input and output filenames
    now = datetime.now()
    ncgen_ofile_nc = os.path.join(
        f"{metadata_dir}/metadata{now:%Y%m%d}.nc"
        )
    
    
    # Run ncgen command to generate the netCDF file from CDL
    myCmd1 = " ".join(
        [
            "ncgen",
            "-o",
            ncgen_ofile_nc,
            cdl_file,
        ]
    )
    print(f'  {myCmd1}')
    rc = subprocess.call(myCmd1, shell=True)
    print(f'  ncgen return code: {rc}')
    
    if rc != 0:
        error_msg('Error in `ncgen` call. Cannot continue.', 10)
        
    return xr.open_dataset(ncgen_ofile_nc, decode_times=False)