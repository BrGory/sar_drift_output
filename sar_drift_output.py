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


def read_arguments():
    import argparse
    import os
    import sys
    

    parser = argparse.ArgumentParser(
        description='Converts SAR drift data to .gpkg an .nc files.'
        )
    
    # Input filename argument
    parser.add_argument(
        '-i', '--input_filename',
        action='store',
        help='SAR drift file to be converted'
        )
    
    # Output directory argument
    parser.add_argument(
        '-o', '--output_dir',
        action='store',
        help='Directory on file sys to store convert shape files '
             'and NetCDF files',        
        )
    
    # Precison argument
    parser.add_argument(
        '-p', '--precision',
        default=3, type=int,
        action='store',
        help="digits after decimal point"
        )

    # Compute distance and bearing argument
    parser.add_argument(
        '-c', '--compute',
        action='store_true',
        help='Compute distance and bearing',        
        )  
    
    # Verbosity argument
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Increase output verbosity'
    )    
        
    
    # Read command line args.
    args = parser.parse_args()

    if args.input_filename is None:
        print(
            'Please pass SAR drift file when calling script. '
            'Use -h for help. <sar_drift_output.py -h>'
            )
        sys.exit(1)

    if args.output_dir is None:
        print(
            'Please pass output directory when calling script. '
            'Use -h for help. <sar_drift_output.py -h>'
            )
        sys.exit(1)
            
    # Check the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check input file exists
    if not os.path.exists((args.input_filename)):
        raise ValueError(f"Cannot find {args.input_filename}")
        
    
    user_args = {
        'input_filename': args.input_filename,
        'output_dir': os.path.normpath(os.path.join(args.output_dir)),
        'precision': args.precision,
        'compute': args.compute,
        'verbose': args.verbose
    }
        
    
    param_string = (
        "CONF PARAMS:\n"
        "  input file -i:                    "
        f"{user_args['input_filename']}\n"
        "  output directory -o:              "
        f"{user_args['output_dir']}\n"
        "  precision (-p):                   "
        f"{user_args['precision']}\n"
        "  compute distance and bearing -c:  "
        f"{user_args['compute']}\n"

    )
    if user_args['verbose'] is True:
        print(param_string)
    
    
    return user_args        


def read_input_file(user_args):
    import helper
    import pandas as pd
    from datetime import datetime, timedelta
    

    # Read the SAR drift data file
    df = pd.read_csv(
        user_args['input_filename'], delimiter=',', header=0
        )
    df.columns = df.columns.str.strip()

    
    
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


    # Convert lon/lat to float
    df['Lon1'] = df['Lon1'].astype(float)
    df['Lat1'] = df['Lat1'].astype(float)
    df['Lon2'] = df['Lon2'].astype(float)
    df['Lat2'] = df['Lat2'].astype(float)
    
    # Standardize naming conventions if azimuth and distance
    # are calculated or not
    if user_args['compute']:
        azimuth, distance = helper.compute_bearing(
            df['Lat1'], df['Lon1'], df['Lat2'], df['Lon2']
        )
        df['Azimuth'] = azimuth
        df['Distance'] = distance / 1000 # in km
    else:
        df['Azimuth'] = df['Bear_deg'].astype(float)
        df['Distance'] = df['Speed_kmdy'].astype(float)
        

    # Get the zonal and meridional displacment used by NetCDF
    df['U'] = (df['U_vel_ms'] * df['JS_Duration']) / 1000 # in km
    df['V'] = (df['V_vel_ms'] * df['JS_Duration']) / 1000 # in km
        
        
    return df


def create_shape_package(user_args, df):
    import os
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import Point, LineString


    # Create Point and Line Geometries
    df['geometry_start'] = df.apply(
        lambda row: Point((row['Lon1'], row['Lat1'])), axis=1
        )
    df['geometry_line'] = df.apply(
        lambda row: LineString(
            [(row['Lon1'], row['Lat1']), (row['Lon2'], row['Lat2'])]
            ),
        axis=1)
    
    # Create GeoDataFrame for start points (points only)
    gdf_start = gpd.GeoDataFrame(
        df, geometry='geometry_start', crs='EPSG:4326'
        )
    # Add a column to distinguish geometry type    
    gdf_start['geometry_type'] = 'point'  
    
    # Create GeoDataFrame for lines (lines only)
    gdf_line = gpd.GeoDataFrame(
        df, geometry='geometry_line', crs='EPSG:4326'
        )
    # Add a column to distinguish geometry type    
    gdf_line['geometry_type'] = 'line'  
    
    # Combine the two GeoDataFrames while retaining original fields
    gdf_combined = pd.concat([
        gdf_start.rename(columns={'geometry_start': 'geometry'}),
        gdf_line.rename(columns={'geometry_line': 'geometry'})
    ], ignore_index=True)
    
    # Recreate the GeoDataFrame with the common geometry column and the CRS
    gdf_combined = gpd.GeoDataFrame(
        gdf_combined, geometry='geometry', crs='EPSG:4326'
        )
    
    # Save as a single GeoPackage file (supports mixed geometries)
    inputfile_rootname = os.path.basename(user_args['input_filename'])
    output_file_path = os.path.join(
        "output", f"{inputfile_rootname}.gpkg"
        )
    gdf_combined.to_file(output_file_path, driver='GPKG')
    
    print(f"GeoPackage <{output_file_path}> created")


def create_netcdf(user_args, df):
    import helper
    import os
    import numpy as np
    from datetime import datetime, timedelta
    from pyproj import Transformer
    import xarray as xr
   
    # Define grid resolution and bounds
    resolution_km = 12.5  # Resolution in km
    
    # Define the polar stereographic projection (EPSG:3413)
    transformer_to_meters = Transformer.from_crs(
        "EPSG:4326", "EPSG:3413", always_xy=True
        )
    
    # Convert lon/lat to x/y coordinates in meters
    df['X1'], df['Y1'] = transformer_to_meters.transform(
        df['Lon1'].values, df['Lat1'].values
        )
    df['X2'], df['Y2'] = transformer_to_meters.transform(
        df['Lon2'].values, df['Lat2'].values
        )
    
    # Get the absolute minimum and maximum for lat (Y) and lon (X)
    min_x, max_x = df[['X1', 'X2']].min().min(), df[['X1', 'X2']].max().max()
    min_y, max_y = df[['Y1', 'Y2']].min().min(), df[['Y1', 'Y2']].max().max()
    
    # Build the X and Y coordinates based on maximum and minimum
    # with steps of resolution multiplied by 1000 km
    x_coords = np.arange(min_x, max_x, resolution_km * 1000)
    y_coords = np.arange(min_y, max_y, resolution_km * 1000)
    
    try:
        # Create an empty grid
        # (time, y, x)
        grid_shape = (1, len(y_coords), len(x_coords))  
        
        # Time defaults
        base_time = datetime(2000, 1, 1)
        mean_time_js = df['Time1_JS'].mean()
        time_val = base_time + timedelta(seconds=mean_time_js)
        epoch = datetime(1970, 1, 1)
        time_sec = (time_val - epoch).total_seconds()
        
        # time_array becomes:
        time_array = np.array([time_sec], dtype='float64')
        
        netcdf_grid = xr.Dataset(
            {
                'Speed_kmdy': (('time', 'y', 'x'),
                               np.full(grid_shape, np.nan),
                               {
                                   'long_name': 'Speed in km/day',
                                   'ioos_category': 'SAR daily sea-ice drift',
                                   'units': 'km/day',
                                   'grid_mapping': 'spatial_ref'
                                   }
                               ),
                'U_vel_ms': (('time', 'y', 'x'),
                             np.full(grid_shape, np.nan),
                             {
                                 'long_name': 'Zonal Velocity',
                                 'ioos_category': 'SAR daily sea-ice drift',
                                 'units': 'm/s',
                                 'grid_mapping': 'spatial_ref'
                                  }
                             ),
                'V_vel_ms': (('time', 'y', 'x'),
                             np.full(grid_shape, np.nan),
                             {
                                 'long_name': 'Meridional Velocity',
                                 'ioos_category': 'SAR daily sea-ice drift',
                                 'units': 'm/s',
                                 'grid_mapping': 'spatial_ref'
                                 }
                             ),
                'Bear_deg': (('time', 'y', 'x'),
                             np.full(grid_shape, np.nan),
                             {
                                 'long_name': 'Bearing',
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
                })
            }
        )
        
        
        # Add NetCDF standard attributes
        netcdf_grid = helper.add_netcdf_attrs(netcdf_grid, df)
        
        
        # Mapping data to the grid
        index_mapping = {}
        for _, row in df.iterrows():
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
                netcdf_grid['U_vel_ms'][0, y_idx, x_idx] = row['U_vel_ms']
                netcdf_grid['V_vel_ms'][0, y_idx, x_idx] = row['V_vel_ms']
                netcdf_grid['Bear_deg'][0, y_idx, x_idx] = row['Bear_deg']
        
        
        # Save to NetCDF with compression level 4
        inputfile_rootname = os.path.basename(user_args['input_filename'])
        output_file_path = os.path.join(
            "output", f"{inputfile_rootname}.nc"
            )
    
        # Save to NetCDF with compression level 4
        netcdf_grid.to_netcdf(output_file_path, mode='w', encoding={
            'Speed_kmdy': {'zlib': True, 'complevel': 4},
            'U_vel_ms': {'zlib': True, 'complevel': 4},
            'V_vel_ms': {'zlib': True, 'complevel': 4},
            'Bear_deg': {'zlib': True, 'complevel': 4}
        })
    
        print(f'NetCDF <{output_file_path}> created')
        
        
    finally:
        # Ensure that these lines are executed even if an error occurs
        netcdf_grid.close()
        del netcdf_grid


def main():
       
    # parse user arguments
    user_args = read_arguments()
    
    # Read SAR drift data file
    df = read_input_file(user_args)
    
    # Create shape file package for QGIS    
    create_shape_package(user_args, df)

    # Create NetCDF file for QGIS    
    create_netcdf(user_args, df)
    
   
    print('done')

    
if __name__ == "__main__":
    main()