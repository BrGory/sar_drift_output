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


def read_arguments():
    """
    Parse and validate command-line arguments for the 
    SAR Drift Output Generator.

    This function uses the argparse module to define and parse 
    command-line options needed for converting SAR drift data into NetCDF
    and GeoPackage formats. It also validates the existence of required
    input files and directories and returns a dictionary of user arguments.

    Supported arguments:
        -i, --input_dir: Path to the input files (required).
        -g, --geotiff file
        -proj, --projection: EPSG projection to apply to data (3413 or 4326)
        -o, --output_dir: Directory where output files (.nc, .gpkg) 
                          will be stored (default: "output").
        -m, --metadata_dir: Directory containing the metadata CDL file
                            (default: "meta").
        -cdl, --cdl_filename: Name of the CDL file used to generate NetCDF
                              metadata (default: "sar_drift_output.cdl").
        -d, --delimiter: Character that separates the fields.
                         (default: ',')
        -p, --precision: Number of decimal places to retain in calculations
                         (default: 3).
        -c, --compute: Flag to recalculate azimuth and distance instead
                       of using original values.
        -v, --verbose: Enable verbose output of parameter settings.

    Returns:
        dict: A dictionary (`user_args`) containing all validated and
              normalized user inputs.

    Exits:
        The program will exit with status 1 if required inputs are missing
        or paths do not exist.
    """

    import util
    import argparse
    import os
    import shutil
    

    parser = argparse.ArgumentParser(
        description='Converts SAR drift data to .gpkg an .nc files.'
        )
    
    # Input filename argument
    parser.add_argument(
        '-i', '--input_dir',
        default='input', type=str,
        action='store',
        help='Directory with all input files.'
        )
       
    # Output directory argument
    parser.add_argument(
        '-o', '--output_dir',
        default='output', type=str,
        action='store',
        help='Directory on file sys to store convert shape files '
             'and NetCDF files.',        
        )
    
    # Metadata directory
    parser.add_argument(
        '-m', '--metadata_dir',
        default='meta', type=str,
        action='store',
        help='Directory where CDL file (metadata) is stored.')
            
    # Input file tye argument
    parser.add_argument(
        '-t', '--input_file_type',
        default='txt', type=str,
        action='store',
        help="Delimited file type, e.g. CSV or TXT."
        )
    
    # Precison argument
    parser.add_argument(
        '-d', '--delimiter',
        default=',', type=str,
        action='store',
        help="Delimiter character. For tab delimiter, enter \\t."
        )
    
    # Precison argument
    parser.add_argument(
        '-p', '--precision',
        default=3, type=int,
        action='store',
        help="digits after decimal point"
        )
    
    # Verbosity argument
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Increase output verbosity'
    )    
        
    
    # Read command line args.
    args = parser.parse_args()

    # Check input file exists
    input_dir = os.path.normpath(os.path.join(args.input_dir))
    if not os.path.exists(input_dir):
        util.error_msg(f"Cannot find input directory `{args.input_dir}`", 1)
        
    # Check metadata dir exists
    metadata_dir = os.path.normpath(os.path.join(args.metadata_dir))
    if not os.path.exists(metadata_dir):
        util.error_msg(f"Cannot find meatdata directory `{metadata_dir}`", 2)
        
        
    # create output directory if needed
    output_dir = os.path.normpath(os.path.join(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)
        
    
    # initialize dictionary
    user_args = {
        'input_dir': input_dir,
        'output_dir': os.path.normpath(os.path.join(args.output_dir)),
        'metadata_dir': metadata_dir,
        'input_file_type': args.input_file_type,
        'delimiter': args.delimiter,
        'precision': args.precision,
        'verbose': args.verbose
    }
            
    # log settings
    param_string = (
        "CONF PARAMS:\n"
        "  input directory -i:               "
        f"{user_args['input_dir']}\n"
        "  output directory -o:              "
        f"{user_args['output_dir']}\n"
        "  metadata directory -m:            "
        f"{user_args['metadata_dir']}\n"
        "  input file type (-t):             "
        f"{user_args['input_file_type']}\n"         
        "  delimiter (-d):                   "
        f"{user_args['delimiter']}\n" 
        "  precision (-p):                   "
        f"{user_args['precision']}\n"
    )
    if user_args['verbose'] is True:
        print(param_string)
    
    
    return user_args        


def main():
    """
    Main execution workflow for converting SAR drift data to GeoPackage
    and NetCDF formats.

    This function:
    - Parses command-line arguments
    - Loads and preprocesses the SAR drift input file
    - Generates a GeoPackage file containing point and line geometries for QGIS
    - Generates a CF-compliant NetCDF file using metadata from a CDL template

    The output files are saved to the specified output directory.

    This function is intended to be executed when the script is run
    as a standalone program.
    """

    import util
    import sar_drift as sd
    import glob
    from datetime import datetime
    
       
    # set timestamp for output files
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    
    # parse user arguments
    user_args = read_arguments()
    
    from pyproj import Transformer
    transformer = Transformer.from_crs(
        "EPSG:4326", 
        "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +x_0=0 +y_0=0"
        " +datum=WGS84 +units=m +no_defs +type=crs",
        always_xy=True
    )
    
    import os        
    # geotiff_path = os.path.normpath(user_args['geotiff_filename'])


    # find SAR input file
    input_dir = user_args['input_dir']
    file_type = user_args['input_file_type']
    input_files = glob.glob(os.path.join(input_dir, f"*.{file_type}"))
    if not input_files:
        util.error_msg(
            f"No .{file_type} file found in input directory `{input_dir}`",
            3
        )
    if len(input_files) > 1:
        util.error_msg(
            f"More than one .{file_type} file found in input directory"
            f" `{input_dir}`:\n{input_files}",
            4
        )
    sar_file = input_files[0]
    sar_basename = os.path.basename(sar_file)
    
    
    # find getiff file
    tif_files = glob.glob(os.path.join(input_dir, '*.tif'))
    if not tif_files:
        util.error_msg(
            f"No .tif file found in input directory `{input_dir}`",
            5
        )
    if len(tif_files) > 1:
        util.error_msg(
            f"More than one .tif file found in input directory:\{tif_files}",
            6
        )
    geotiff_file = tif_files[0]
    
    
    # find cdl file
    metadata_dir = user_args['metadata_dir']
    cdl_files = glob.glob(os.path.join(metadata_dir, '*.cdl'))    
    if not cdl_files:
        util.error_msg(
            f"No .cdl file found in metadata driectory `{metadata_dir}`",
            7
        )
    if len(cdl_files) > 1:
        util.error_msg(
            "More than one .cdl file found in metadata directory:\n"
            f"{cdl_files}",
            8
        )
    cdl_file = cdl_files[0]

                              
        
    # Read SAR drift data file
    df_sar = sd.read_input_file(
        sar_file, user_args['precision'], transformer, user_args
    )

    
    # Create shape file package for QGIS    
    gdf_points, gdf_lines = sd.create_shape_package(
        user_args, df_sar, transformer, timestamp
    )


    # Create NetCDF file for QGIS    
    sd.create_netcdf(user_args, df_sar, transformer, timestamp, cdl_file)

    
    # Overlay SAR drift data vectors on geotiff image
    sd.overlay_sar_drift_on_geotiff(
        geotiff_file, gdf_lines, df_sar, timestamp, sar_basename, user_args
    )
    
    
if __name__ == "__main__":
    main()