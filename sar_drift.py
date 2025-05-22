# -*- coding: utf-8 -*-
"""
******************************************************************************

 Project:    SAR Drift Output Generator
 Purpose:    Create shape file package (.gpkg) and NetCDF file (.nc) from the
             SAR drift daily file. This script allows the data to be visualized
             in QGIS or any program that can read NetCDF
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

def read_input_file(sar_file, precision, transformer, user_args):
    """
    Load and preprocess SAR drift data from a CSV file.

    This function performs the following:
        - Reads a SAR drift CSV file with start/end positions and times
        - Cleans the dataset by removing invalid records
        - Converts time values (Julian seconds since 2000-01-01) to 
          readable timestamps
        - Computes observation durations in both datetime and seconds
        - Rounds and converts lat/lon coordinates to a specified decimal
          precision
        - Projects coordinates from WGS84 to EPSG:3413 using a pyproj
          Transformer
        - Computes zonal (dx) and meridional (dy) displacements in meters
        - Converts U/V velocity from m/s to km/day
        - Calculates total drift distance in kilometers using projected
          coordinates
        - Writes a formatted intermediate CSV to the output directory

    Parameters:
        sar_file (str): Path to the SAR drift input CSV file.
        precision (int): Number of decimal places to round lat/lon coordinates.
        transformer (pyproj.Transformer): Transformer from WGS84 to EPSG:3413.
        user_args (dict): Dictionary of user-defined arguments including:
            - 'delimiter': Delimiter used in the CSV file (e.g., ',' or ';')
            - 'output_dir': Path to output directory for saving formatted CSV

    Returns:
        pandas.DataFrame: Cleaned and enriched DataFrame with added columns:
            - Date1, Date2: Human-readable timestamps
            - Duration, JS_Duration: Observation durations
            - Lon1, Lat1, Lon2, Lat2: Rounded geographic coordinates
            - X1, Y1, X2, Y2: Projected coordinates (EPSG:3413, meters)
            - dx, dy: Displacements (meters)
            - U_kmdy, V_kmdy: Velocities in km/day
            - total_distance_km: Great-circle distance in kilometers
    """

    import util
    import os
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    

    # Read the SAR drift data file
    df = pd.read_csv(
        sar_file, delimiter=user_args['delimiter'],
        header=0, engine='python'
    )
    df.columns = df.columns.str.strip()
    
    # Check of delimiter worked or input file features changed
    needed_cols = ['Time1_JS', 'Time2_JS',
                   'Lon1', 'Lat1', 'Lon2', 'Lat2',
                   'U_vel_ms', 'V_vel_ms', 'Speed_kmdy', 'Bear_deg'
    ]
    if not any(col in df.columns for col in needed_cols):
        util.error_msg(
            f"The delimiter `{user_args['delimiter']}` did not properly parse "
            "the input file. Check documentation for proper syntax if tab "
            "character used as the delimiter. Or, the structure of the input "
            f"file changed.\nExpected features:\n{needed_cols}.",
            9
        )
    
    # Add the appropriate input file to a data frame
    # Julian seconds start from date 01-01-2000
    base_time = datetime(2000, 1, 1)

    # Remove rows from Data Frame where orig_bearing = 0
    # The values for these observations are incorrect
    df = df[df['Bear_deg'] != 0]

    # Create new Date* columnc by converting Time_JS* columns to datetime
    df['Date1'] = df["Time1_JS"].apply(
        lambda x: base_time + timedelta(seconds=x)
        )
    df['Date1'] = df['Date1'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df['Date2'] = df["Time2_JS"].apply(
        lambda x: base_time + timedelta(seconds=x)
        )
    df['Date2'] = df['Date2'].dt.strftime('%Y-%m-%d %H:%M:%S')


    # Calculate duration of observations
    # 1. Date time
    # 2. Raw Julian seconds
    df['Duration'] = pd.to_timedelta(
        df['Time2_JS'] - df['Time1_JS'], unit='s'
        ).astype(str)
    df['JS_Duration'] = (
        df['Time2_JS'] - df['Time1_JS']
        )


    # Convert lon/lat to float and round
    df['Lon1'] = np.round(df['Lon1'].astype(float), precision)
    df['Lat1'] = np.round(df['Lat1'].astype(float), precision)
    df['Lon2'] = np.round(df['Lon2'].astype(float), precision)
    df['Lat2'] = np.round(df['Lat2'].astype(float), precision)
    

    # transform lon/lat to polarstereographic meters
    df['X1'], df['Y1'] = transformer.transform(
        df['Lon1'].values, df['Lat1'].values
        )
    df['X2'], df['Y2'] = transformer.transform(
        df['Lon2'].values, df['Lat2'].values
        )
    
    # Get the zonal and meridional displacment used by NetCDF
    df['U_kmdy'] = (df['U_vel_ms'] * 60 * 60 * 24) / 1000 # in km
    df['V_kmdy'] = (df['V_vel_ms'] * 60 * 60 * 24) / 1000 # in km
    

    # set dX and dY to plot quivers
    df['dx'] = df['X2'] - df['X1']
    df['dy'] = df['Y2'] - df['Y1']
    df['total_distance_km'] = util.compute_distance_meters(
        df['X1'].values, df['Y1'].values,
        df['X2'].values, df['Y2'].values,
        precision
    )
    
    # save intermediary file with added columns
    formatted_sar_filename = os.path.join(
        user_args['output_dir'], 'formatted_sar_drift.csv'
    )
    df.to_csv(formatted_sar_filename, index=False)
    
    print("  Read and formatted input file")
    
    return df


def create_shape_package(user_args, df, transformer, timestamp):
    """
    Generate a GeoPackage (.gpkg) file with point and line geometries
    derived from SAR drift data, suitable for visualization in GIS software.

    This function processes a SAR drift DataFrame and performs the following:
        - Transforms geographic coordinates (lon/lat) to Polar Stereographic
          (EPSG:3413)
        - Creates start and end point geometries for each drift vector
        - Creates line geometries connecting start and end points
        - Saves all geometries into a single multi-layer GeoPackage file with
          three layers: 'start_points', 'end_points', and 'drift_lines'

    Each output layer:
        - Uses the EPSG:3413 coordinate reference system
        - Includes the original date and position metadata
        - Can be opened in GIS software such as QGIS

    Parameters:
        user_args (dict): Dictionary of user-defined arguments, including:
            - 'output_dir' (str): Path to the directory for saving
                                  the .gpkg file
        df (pandas.DataFrame): DataFrame containing SAR drift data, including:
            - 'Date1', 'Date2': Observation timestamps
            - 'Lon1', 'Lat1', 'Lon2', 'Lat2': Geographic coordinates
        transformer (pyproj.Transformer): Transformer to convert WGS84
                                          to EPSG:3413
        timestamp (str): Timestamp string used for naming the output file

    Returns:
        tuple:
            gdf_start (geopandas.GeoDataFrame): GeoDataFrame of start point
                                                geometries 
                (EPSG:3413) including metadata fields and 'geometry_type' =
                'point'.
            gdf_line (geopandas.GeoDataFrame): GeoDataFrame of drift line
                                               geometries connecting start and
                                               end points, with 'geometry_type'
                                               = 'line'.

    Output:
        Writes a GeoPackage file named "sar_drift_<timestamp>.gpkg"
        with three layers:
            - 'start_points': Point geometries at drift start locations
            - 'end_points': Point geometries at drift end locations
            - 'drift_lines': Line geometries connecting start to end

    Notes:
        - All geometries are tagged with a 'geometry_type' field
          ('point' or 'line')
        - Geometry is projected in meters (EPSG:3413)
        - Useful for visualizing individual drifts or overlaying
          with SAR data in GIS
    """

    import os
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    from pyproj import CRS
    
    # reduce data frame to needed features
    cols = ['Date1', 'Date2', 'Lon1', 'Lat1', 'Lon2', 'Lat2']
    df_reduced = df[cols].copy()
    
    # transform projection
    df_reduced['X1'], df_reduced['Y1'] = transformer.transform(
        df_reduced['Lon1'].values, df_reduced['Lat1'].values)
    df_reduced['X2'], df_reduced['Y2'] = transformer.transform(
        df_reduced['Lon2'].values, df_reduced['Lat2'].values)

    # Create Point and Line Geometries
    df_reduced['geometry_start'] = df_reduced.apply(
        lambda row: Point((row['X1'], row['Y1'])), axis=1
        )
    
    df_reduced['geometry_end'] = df_reduced.apply(
        lambda row: Point((row['X2'], row['Y2'])), axis=1
        )
    df_reduced['geometry_line'] = df_reduced.apply(
        lambda row: LineString(
            [(row['X1'], row['Y1']), (row['X2'], row['Y2'])]
            ),
        axis=1)

    # Create GeoDataFrame for start points (points only)
    gdf_start = gpd.GeoDataFrame(
        df_reduced, geometry='geometry_start'
        )
    # Add a column to distinguish geometry type    
    gdf_start['geometry_type'] = 'point'  
    
    # Create GeoDataFrame for end points (points only)
    gdf_end = gpd.GeoDataFrame(
        df_reduced, geometry='geometry_end'
        )
    # Add a column to distinguish geometry type    
    gdf_end['geometry_type'] = 'point'  
    
    # Create GeoDataFrame for lines (lines only)
    gdf_line = gpd.GeoDataFrame(
        df_reduced, geometry='geometry_line'
        )
    # Add a column to distinguish geometry type    
    gdf_line['geometry_type'] = 'line'  
    
    # Combine the two GeoDataFrames while retaining original fields
    gdf_combined = pd.concat([
        gdf_start.rename(columns={'geometry_start': 'geometry'}),
        gdf_end.rename(columns={'geometry_end': 'geometry'}),
        gdf_line.rename(columns={'geometry_line': 'geometry'})
    ], ignore_index=True)
    
    # Recreate the GeoDataFrame with the common geometry column
    gdf_combined = gpd.GeoDataFrame(
        gdf_combined, geometry='geometry'
        )
    
    # Save as a single GeoPackage file (supports mixed geometries)
    geopackage_file = f"sar_drift_{timestamp}.gpkg"
    output_file_path = os.path.join(
        user_args['output_dir'], f"{geopackage_file}"
        )
    
    gdf_start = gdf_start.rename(
        columns={'geometry_start': 'geometry'}
    ).set_geometry('geometry')
    gdf_start.crs = CRS.from_epsg(3413)
    gdf_start.to_file(output_file_path, layer='start_points', driver='GPKG')
    
    gdf_end = gdf_end.rename(
        columns={'geometry_end': 'geometry'}
    ).set_geometry('geometry')
    gdf_end.crs = CRS.from_epsg(3413)
    gdf_end.to_file(output_file_path, layer='end_points', driver='GPKG')
    
    
    gdf_line = gdf_line.rename(
        columns={'geometry_line': 'geometry'}
    ).set_geometry('geometry')
    gdf_line.crs = CRS.from_epsg(3413)
    gdf_line.to_file(output_file_path, layer='drift_lines', driver='GPKG')
    
    gdf_start = gdf_start['geometry']
    gdf_line = gdf_line['geometry']
    
    print("  GeoPackage created")
    
    return gdf_start, gdf_line


def create_netcdf(user_args, df, transformer, timestamp, cdl_file):
    """
    Generate a CF/ACDD-compliant NetCDF file from SAR drift data.

    This function performs the following operations:
        - Projects geographic coordinates (longitude/latitude) into 
          EPSG:3413 (Polar Stereographic) using a pyproj Transformer
        - Defines a 2D spatial grid at 1 km resolution based on the spatial
          extent of the drift data
        - Initializes NetCDF variables for:
              - `Speed_kmdy` (drift speed in km/day)
              - `dx`, `dy` (zonal/meridional displacement in meters/day)
              - `Bear_deg` (bearing in degrees from true north)
        - Computes the observation time range and stores it as a CF-compliant
          time coordinate (seconds since Unix epoch)
        - Loads metadata from a CDL file and populates standard global 
          attributes in the NetCDF file
        - Maps each drift observation to the nearest grid cell, 
          skipping duplicates with a warning
        - Writes the result to a compressed `.nc` file (NetCDF4 format)

    Parameters:
        user_args (dict): Dictionary containing script arguments, including:
            - 'input_filename': Path to the SAR drift input file
            - 'metadata_dir': Path to directory containing metadata templates
            - 'output_dir': Path to directory save NetCDF file
        df (pandas.DataFrame): Cleaned SAR drift data containing the
                               following fields:
            - 'X1', 'Y1', 'X2', 'Y2': Projected coordinates in meters
              (EPSG:3413)
            - 'dx', 'dy': Displacements in meters/day
            - 'Speed_kmdy': Speed in kilometers per day
            - 'Bear_deg': Bearing angle in degrees
            - 'Date1': Observation datetime string 
                      (used to extract time coverage)
        transformer (pyproj.Transformer): Transformer used for coordinate
                                          projection
        timestamp (str): Timestamp string used to name the output NetCDF file
        cdl_file (str): Name of the file with the NetCDF metadata standards

    Returns:
        None

    Output:
        Writes a NetCDF file named `sar_drift_<timestamp>.nc` to the
        `output/` directory.

    Notes:
        - The grid is defined in projected meters (EPSG:3413) with 1 km
          resolution
        - Metadata placeholders in the CDL template (e.g., `FILL_DATE_CREATED`)
          are replaced with actual values at runtime
        - Observations mapped to the same grid cell will emit a warning
          and be skipped
        - The resulting NetCDF is compatible with QGIS and other
          CF-compliant tools
    """

    import util
    import os
    import numpy as np
    import pandas as pd
    from datetime import datetime
    import xarray as xr
   
    # Define grid resolution and bounds
    resolution_km = 1  # Resolution in km
    

    # reduce data frame to needed features   
    cols = [
        'Date1', 'X1', 'Y1', 'X2', 'Y2', 'dx', 'dy', 'Speed_kmdy', 'Bear_deg'
    ]
    df_reduced = df[cols].copy()
    df_reduced['Date1'] = pd.to_datetime(df_reduced['Date1'])
    
    
    # Get the absolute minimum and maximum for lat (Y) and lon (X)
    min_x, max_x = (
        df_reduced[['X1', 'X2']].min().min(),
        df_reduced[['X1', 'X2']].max().max()
    )
    min_y, max_y = (
        df_reduced[['Y1', 'Y2']].min().min(),
        df_reduced[['Y1', 'Y2']].max().max()
    )
    
    
    # Build the X and Y coordinates based on maximum and minimum
    # with steps of resolution multiplied by 1000 km
    x_coords = np.arange(min_x, max_x, resolution_km * 1000)
    y_coords = np.arange(min_y, max_y, resolution_km * 1000)
    
    try:
        # Create an empty grid
        # (time, y, x)
        grid_shape = (1, len(y_coords), len(x_coords))  
        
        # time defaults
        epoch = datetime(1970, 1, 1)
        mean_time = df_reduced['Date1'].mean()
        min_time = df_reduced['Date1'].min()
        max_time = df_reduced['Date1'].max()
        
        # convert to "seconds since 1970-01-01 00:00:00"
        time_sec = (mean_time - epoch).total_seconds()
        
        # time array for NetCDF
        time_array = np.array([time_sec], dtype='float64')
        
        # shell of NetCDF
        netcdf_grid = xr.Dataset(
            {
                'Speed_kmdy': (('time', 'y', 'x'),
                                np.full(grid_shape, np.nan),
                                {
                                    'long_name': "Speed in km/day",
                            		'standard_name': "sea_ice_speed",
                                    'ioos_category': (
                                        "SAR daily sea-ice drift"
                                        ),
                                    'units': "km/day",
                                    'grid_mapping': "spatial_ref"
                                    }
                                ),
                'dx': (('time', 'y', 'x'),
                              np.full(grid_shape, np.nan),
                              {
                                  'long_name': 'Zonal Velocity',
                                  'standard_name': 'movement_in_x_direction',
                                  'ioos_category': 'SAR daily sea-ice drift',
                                  'units': 'm/day',
                                  'grid_mapping': 'spatial_ref'
                                  }
                              ),
                'dy': (('time', 'y', 'x'),
                              np.full(grid_shape, np.nan),
                              {
                                  'long_name': 'Meridional Velocity',
                                  'standard_name': 'movement_in_y_direction',
                                  'ioos_category': 'SAR daily sea-ice drift',
                                  'units': 'm/day',
                                  'grid_mapping': 'spatial_ref'
                                  }
                              ),
                'Bear_deg': (('time', 'y', 'x'),
                              np.full(grid_shape, np.nan),
                              {
                                  'long_name': 'Bearing',
                                  'standard_name': "direction_true_north",
                                  'ioos_category': 'SAR daily sea-ice drift',
                                  'units': 'degrees',
                                  'grid_mapping': 'spatial_ref'
                                  }
                              )               
            },
            # Add metadata to coords so QGIS can properly scale the map
            coords={
                'x': (('x',), x_coords,
                      {
                          'actual_range': (
                              [float(x_coords.min()), float(x_coords.max())]
                              ),
                          'axis': 'X',
                          'comment': (
                              'x values are the centers of the grid cells'
                              ),
                          'ioos_category': 'Location',
                          'long_name': 'x coordinate of projection',
                          'standard_name': 'projection_x_coordinate',
                          'units': 'm'
                          }),
                'y': (('y',), y_coords,
                      {
                          'actual_range': (
                              [float(y_coords.min()), float(y_coords.max())]
                              ),
                          'axis': 'Y',
                          'comment': (
                              'y values are the centers of the grid cells'
                              ),                          
                          'ioos_category': 'Location',
                          'long_name': 'y coordinate of projection',
                          'standard_name': 'projection_x_coordinate',
                          'units': 'm'
                          }),
                'time': (('time',), time_array, {
                    'actual_range': (
                        [float(time_array.min()), float(time_array.max())]
                        ),
                    'axis': 'T',
                    'comment': (
                        'This is the 00Z reference time. '
                        'Note that products are nowcasted to be valid '
                        'specifically at the time given here.'
                        ),
                    'CoordinateAxisType': 'Time',
                    'ioos_category': 'Time',
                    'long_name': 'Centered Time',
                    'standard_name': 'time',
                    'time_origin': '01-Jan-1970 0:00:00',
                    'units': "seconds since 1970-01-01 00:00:00 UTC"
                })
            }
        )
        
        
        # Set NetCDF standard attributes
        metadata_nc = util.set_metadata(user_args, cdl_file)
        
        
        # Replace placeholders with real values
        netcdf_grid.attrs.update(metadata_nc.attrs)
        netcdf_grid.attrs['date_created'] = (
            datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
            )
        netcdf_grid.attrs['time_coverage_start'] = (
            min_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            )
        netcdf_grid.attrs['time_coverage_end'] = (
            max_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            )
        
        
        
        # Mapping data to the grid
        index_mapping = {}
        for _, row in df_reduced.iterrows():
            # To get the i, j indices, we
            x_idx = np.argmin(np.abs(x_coords - row['X1']))
            y_idx = np.argmin(np.abs(y_coords - row['Y1']))
                
            
            # Create a unique key for each (i, j) pair
            index_key = (y_idx, x_idx)
            
            # Check if this index already exists
            if index_key in index_mapping:
                print(f"Duplicate index detected at (i, j): {index_key}")
            else:
                # Store data in the grid
                netcdf_grid['Speed_kmdy'][0, y_idx, x_idx] = row['Speed_kmdy']
                netcdf_grid['dx'][0, y_idx, x_idx] = row['dx']
                netcdf_grid['dy'][0, y_idx, x_idx] = row['dy']
                netcdf_grid['Bear_deg'][0, y_idx, x_idx] = row['Bear_deg']
        
        
        # Save to NetCDF with compression level 4
        
        output_file_path = os.path.join(
            user_args["output_dir"], f"sar_drift_{timestamp}.nc"
            )
    
        # Save to NetCDF with compression level 4
        netcdf_grid.to_netcdf(output_file_path, mode='w', encoding={
            'Speed_kmdy': {'zlib': True, 'complevel': 4},
            'dx': {'zlib': True, 'complevel': 4},
            'dy': {'zlib': True, 'complevel': 4},
            'Bear_deg': {'zlib': True, 'complevel': 4}
        })
    
        
        
        
    finally:
        # Ensure that these lines are executed even if an error occurs
        netcdf_grid.close()
        del netcdf_grid
        
        print('  NetCDF file created')        
        
        
def overlay_sar_drift_on_geotiff(
        geotiff_path, gdf_lines, df_sar, timestamp, sar_basename, user_args
    ):
    """
    Create a two-panel visualization of SAR sea-ice drift data overlaid 
    on a GeoTIFF image, with both a regional overview map and a detailed 
    drift vector plot.
    
    This function:
        - Loads and displays the SAR backscatter GeoTIFF image
        (projected in EPSG:3413)
        - Plots drift vectors (`dx`, `dy`) as quivers based on line geometries
        - Draws a 50–100 km scale bar for spatial reference
        - Annotates a True North arrow using geodetic conversion
        - Includes a left panel showing a North Polar overview with a red
          rectangle indicating the region of interest
        - Adds axis labels, rotated tick labels, and custom titles
        - Saves the result as a high-resolution PNG image
    
    Parameters:
        geotiff_path (str): Path to the GeoTIFF file representing SAR
                            backscatter imagery.
        gdf_lines (GeoSeries): GeoSeries or list of LineString geometries
                               representing SAR-derived drift vectors.
        df_sar (pandas.DataFrame): DataFrame containing start/end projected
                                   coordinates:
            - 'X1', 'Y1', 'X2', 'Y2': EPSG:3413 coordinates in meters.
        timestamp (str): Timestamp string (e.g., "20250521_1530") for naming
                         the output file.
        sar_basename (str): Short name of the SAR input file,
                            used in the plot title.
        user_args (dict): Dictionary containing script arguments, including:
            - 'output_dir': Path to the save png
    
    Returns:
        matplotlib.figure.Figure: The generated figure with two subplots:
            - Left: Arctic overview with red bounding box
            - Right: Drift vectors overlaid on SAR GeoTIFF
    
    Output:
        A PNG file named `sar_drift_<timestamp>.png` is saved in the current
        working directory.
    
    Notes:
        - The right subplot uses raw Polar Stereographic x/y coordinates
          in meters.
        - The left subplot uses Cartopy’s North Polar Stereographic projection.
        - Only LineString geometries are used for drift vector plotting.
        - The GeoTIFF image must include GCPs or valid transform info
          to be reprojected.
    """
    
    import util
    import os
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from shapely.geometry import LineString, Polygon
    from pyproj import Transformer
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    # SAR drift bounds for map extent
    xmin = df_sar[['X1', 'X2']].min().min()
    xmax = df_sar[['X1', 'X2']].max().max()
    ymin = df_sar[['Y1', 'Y2']].min().min()
    ymax = df_sar[['Y1', 'Y2']].max().max()
    

    # initialize plot
    fig = plt.figure(figsize=(18, 10))
    

    # --- overview map with land and coastlines---
    # transform 3413 to 4326 to draw True North arrow
    to_lonlat = Transformer.from_crs("EPSG:3413", "EPSG:4326", always_xy=True)
    
    # Convert all 4 corners of the SAR extent
    corner_coords = [
        (xmin, ymin),
        (xmax, ymin),
        (xmax, ymax),
        (xmin, ymax),
        (xmin, ymin)  # close the loop
    ]
    
    # transform meters to degrees
    corner_lonlat = [to_lonlat.transform(x, y) for x, y in corner_coords]
    
    # Create a shapely Polygon and extract x/y separately
    poly = Polygon(corner_lonlat)
    inset_lon, inset_lat = poly.exterior.xy
    
    main_ax = fig.add_subplot(1, 2, 1, projection=ccrs.NorthPolarStereo())
    main_ax.add_feature(cfeature.LAND, zorder=0, facecolor='lightgray')
    main_ax.add_feature(cfeature.COASTLINE, zorder=1)
    main_ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())
    gl = main_ax.gridlines(
        draw_labels=True, crs=ccrs.PlateCarree(), linestyle='--', alpha=0.5
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    
    
    # Plot red box on main overview map
    main_ax.plot(
        inset_lon, inset_lat, color='red',
        linewidth=2, transform=ccrs.PlateCarree()
    )
    
    
    # --- geotiff with overlays ---
    ax = fig.add_subplot(1, 2, 2)
    
    # read SAR geotiff
    masked_xr, map_extent_xr = util.read_geotiff_rasterio(geotiff_path)
    
    # plot geotiff    
    ax.imshow(
        masked_xr,
        extent=map_extent_xr,
        origin="upper",
        cmap="gray"
    )
    
    # Extract quiver vector data from LineStrings
    lon_start = []
    lat_start = []
    dx = []
    dy = []
    
    for line in gdf_lines:
        if isinstance(line, LineString):
            x0, y0 = line.coords[0]     # start point
            x1, y1 = line.coords[-1]    # end point
            lon_start.append(x0)
            lat_start.append(y0)
            dx.append(x1 - x0)
            dy.append(y1 - y0)
            
    
    # Plot drift vectors as quivers
    ax.quiver(
        lon_start, lat_start,
        dx, dy,
        angles='xy',
        scale_units='xy',
        scale=1.0,       # adjust arrow length scaling
        width=0.003,
        color='orange',
        alpha=0.8
    )
    
    # create buffer around geotiff
    buffer_deg = 10_000 # 10km
    map_extent = [
        xmin - buffer_deg,
        xmax + buffer_deg,
        ymin - buffer_deg,
        ymax + buffer_deg
    ]
    
    
    # scale bar (bottom-left placement)
    segment_length_m = 50_000  # 50 km
    bar_height = 0.005 * (map_extent[3] - map_extent[2])
    x_start = (
        map_extent[1] - (2 * segment_length_m) - .65 * 
        (map_extent_xr[1] - map_extent[0])
    )
    y_start = map_extent[2] + 0.05 * (map_extent[3] - map_extent[2])
    
    # Draw black and white rectangles
    for i in range(2):  # two segments: 0–50, 50–100
        color = 'black' if i % 2 == 0 else 'white'
        ax.add_patch(plt.Rectangle(
            (x_start + i * segment_length_m, y_start),  # bottom-left corner
            segment_length_m, bar_height,               # width, height
            facecolor=color,
            edgecolor='black'
        ))
    
    # Draw border
    ax.plot(
        [x_start, x_start + 2 * segment_length_m],
        [y_start, y_start],
        color='black', linewidth=1
    )
    ax.plot(
        [x_start, x_start + 2 * segment_length_m],
        [y_start + bar_height, y_start + bar_height],
        color='black', linewidth=1
    )
    
    # Vertical ticks
    for i in range(3):  # 0, 50, 100
        xpos = x_start + i * segment_length_m
        ax.plot(
            [xpos, xpos],
            [y_start, y_start + bar_height],
            color='black', linewidth=1
        )
        ax.text(
            xpos,
            y_start - 0.01 * (map_extent_xr[3] - map_extent_xr[2]),
            f'{i * 50}',
            ha='center',
            va='top',
            fontsize=10,
            color='black'
        )
    
    # Label above the bar
    ax.text(
        x_start + segment_length_m,
        y_start + bar_height + 0.005 * (map_extent[3] - map_extent[2]),
        'km',
        ha='center',
        va='bottom',
        fontsize=10,
        color='black'
    )

        
    # True North arrow
    # create transformer for EPSG:3413 to EPSG:4326 and back
    to_latlon = Transformer.from_crs("EPSG:3413", "EPSG:4326", always_xy=True)
    to_xy = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
    
    # bottom-right corner of the plot as reference point
    x_ref = xmax - 0.05 * (xmax - xmin)
    y_ref = ymin + 0.05 * (ymax - ymin)
    
    # convert meters to degrees
    lon_ref, lat_ref = to_latlon.transform(x_ref, y_ref)
    
    # move a small distance north
    lat_north = lat_ref + 0.5
    lon_north = lon_ref
    
    # convert degrees back to meters
    x_north, y_north = to_xy.transform(lon_north, lat_north)
    
    # arrow
    ax.annotate(
        '', xy=(x_north, y_north), xytext=(x_ref, y_ref),
        arrowprops=dict(
            facecolor='black', edgecolor='black', width=2, headwidth=10
        ),
    )
    ax.text(
        x_ref, y_ref - 20000, 'N', color='black',
        fontsize=16, ha='center', va='top'
    )        
        
    # tick label adjustments
    # Use plain formatting for large x-axis numbers
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x:.0f}')
    )
    # Rotate x-axis tick labels
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    # axes labels 
    ax.set_xlabel("Longitude (meters)")
    ax.set_ylabel("Latitude (meters)")
    
    # titles
    ax.set_title('dX, dY Vectors with Magnitude', fontsize=12)
    main_ax.set_title('Observation region', fontsize=12)
    fig.suptitle(
        f'Vector Overlay on GeoTiff:\n{sar_basename}',
        fontsize=14
    )
    
    
    plt.tight_layout()
    
    # save plot as .png
    png_file = os.path.join(
        user_args['output_dir'], f'sar_drift_{timestamp}.png'
    )
    fig.savefig(png_file, bbox_inches='tight', dpi=300)
    
    print('  Overlay plot created')
    
    return fig
