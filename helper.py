"""
******************************************************************************

 Project:    SAR Drift Output Generator
 Purpose:    Helper function for sar_drift_output
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
from typing import Tuple

  
def compute_bearing(
    lat1: float, lon1: float,
    lat2: float, lon2: float 
    ) -> Tuple[float, float]: # Returns a tuple: (fwd_azimuth, distance)

    """
    Calculate the daily drift between two points (start and end) based on 
    their latitude and longitude.

    Parameters:
        lat1 (float): Starting latitude(s), can be a single float
                      or a list of floats
        lon1 (float): Starting longitude(s), can be a single float
                      or a list of floats 
        lat2 (float): Ending latitude(s), can be a single float
                      or a list of floats
        lon2 (float): Ending longitude(s), can be a single float
                      or a list of floats  
    Returns: 
        fwd_azimuth (float): Forward azimuth in degrees (0 to 360),
                             measured clockwise from true north
        distance (float): Great circle distance between the two points
                          in meters
    Reference:
        https://pyproj4.github.io/pyproj/stable/api/geod.html#pyproj.Geod.inv
        
    Author:
        Sun Bak-Hospital, sun.bak-hospital@noaa.gov
    """
    from pyproj import Geod 

    # Initialize a geodetic object using the WGS84 ellipsoid
    geod = Geod(ellps='WGS84')

    # Calculate azimuth and distance using geod.inv method
    # Note: Arguments order is (lon1, lat1, lon2, lat2) 
    # as required by pyproj.Geod.inv
    fwd_azimuth, _ , distance = geod.inv(lon1, lat1, lon2, lat2)
    
    # Return the calculated forward azimuth, back azimuth, and distance
    return fwd_azimuth, distance 
    
def compute_r_earth(latitude):
    """Computes the local Earth radius at a given latitude in degrees.

    Parameters:
        latitude (float): Latitude in degrees where the Earth's radius
                          should be calculated.

    Returns:
        float: Radius of Earth in meters.
    """
    
    import numpy as np
    
    # Define Earth's semi-major and semi-minor axes (meters)
    r_equator = 6.378137E6  # Equatorial radius (m)
    r_pole = 6.356752E6     # Polar radius (m)

    # Convert latitude to radians
    latitude_rad = np.radians(latitude)

    # Precompute the cos and sin of latitude
    cos_lat = np.cos(latitude_rad)
    sin_lat = np.sin(latitude_rad)

    # Compute the numerator and denominator separately
    numerator = (r_equator**2 * cos_lat)**2 + (r_pole**2 * sin_lat)**2
    denominator = (r_equator * cos_lat)**2 + (r_pole * sin_lat)**2

    # Calculate Earth radius
    r_earth = np.sqrt(numerator / denominator)

    return r_earth


def get_haversine(lat1, lon1, lat2, lon2, r_earth):
    """
    Calculate the great-circle distance and bearing between two points
    on the Earth using the Haversine formula and the spherical law of cosines.

    Parameters:
        lat1 (float): Latitude of the first point in degrees.
        lon1 (float): Longitude of the first point in degrees.
        lat2 (float): Latitude of the second point in degrees.
        lon2 (float): Longitude of the second point in degrees
        r_earth (float): Radius of the Earth (in meters),
                         used for calculating the distance.

    Returns:
        dX (float): The east-west component of the displacement (in meters), 
                    calculated based on the bearing and distance.
        dY (float): The north-south component of the displacement (in meters), 
                    calculated based on the bearing and distance.
    """
        
    import numpy as np
    
    # Convert degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Compute differences in latitude and longitude
    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1

    # Haversine formula components
    sin_delta_lat = np.sin(delta_lat / 2)
    sin_delta_lon = np.sin(delta_lon / 2)
    cos_lat1 = np.cos(lat1)
    cos_lat2 = np.cos(lat2)

    # Compute the 'a' term in the Haversine formula
    a = sin_delta_lat ** 2 + cos_lat1 * cos_lat2 * sin_delta_lon ** 2

    # Compute the 'c' term (angular distance in radians)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Calculate the great-circle distance
    distance = r_earth * c

    # Bearing calculation components
    sin_delta_lon_cos_lat2 = np.sin(delta_lon) * cos_lat2
    cos_lat1_sin_lat2 = np.cos(lat1) * np.sin(lat2)
    sin_lat1_cos_lat2_cos_delta_lon = (
        np.sin(lat1) * cos_lat2 * np.cos(delta_lon)
        )

    # Calculate the initial bearing (azimuth)
    bearing = np.arctan2(
        sin_delta_lon_cos_lat2, cos_lat1_sin_lat2 - 
        sin_lat1_cos_lat2_cos_delta_lon
        )

    # Calculate dX and dY
    # (East-West and North-South components of the displacement)
    dX = distance * np.cos(bearing)  # East-West displacement
    dY = distance * np.sin(bearing)  # North-South displacement

    return dX, dY


def add_netcdf_attrs(netcdf_grid, df):
    from datetime import datetime, timedelta
    
    # Time measurements
    base_time = datetime(2000, 1, 1)
    min_time_js = df['Time1_JS'].min()
    max_time_js = df['Time1_JS'].max()
    
    # Convert to datetime
    min_time = base_time + timedelta(seconds=float(min_time_js))
    max_time = base_time + timedelta(seconds=float(max_time_js))
    
    # Format as 'YYYY-MM-DDT00:00:00Z'
    min_time_str = min_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    max_time_str = max_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # Create standard attributes for NetCDF metadata
    netcdf_grid.attrs.update({
        #'_NCProperties': netcdf_grid.attrs.get('_NCProperties'),
        'acknowledgement': 'Produced by NOAA using IABP data.',
        'cdm_data_type': 'Grid',
        'contributor_name': (
            'NOAA PolarWatch, Southwest Fisheries Science Center, '
            'NESDIS STAR, U.S. National Ice Center'
            ),
        'contributor_role': 'Producer, Publisher, Advisor, Originator',
        'Conventions': 'CF-1.8, ACDD-1.3',
        'creator_email': 'brendon.gory@noaa.gov',
        'creator_institution': 'CIRA/NOAA',
        'creator_name': 'Brendon Gory',
        'creator_type': 'Institution',
        'creator_url': 'https://noaa.gov/',
        'date_created': datetime.utcnow().strftime('%Y-%m-%dT00:00:00Z'),
        'grid_mapping__ChunkSizes': 1,
        'grid_mapping_false_easting': 0,
        'grid_mapping_false_northing': 0,
        'grid_mapping_latitude_of_projection_origin': 90,
        'grid_mapping_name': 'ploar_stereographic',
        'grid_mapping_proj4text': (
            '+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-80 +k=1 '
            '+x_0=0 +y_0=0 +a=6378137 +b=6356257 +units=m +no_defs'
            ),
        'grid_mapping_semi_major_axis': 6378137,
        'grid_mapping_semi_minor_axis': 6356257,
        'grid_mapping_spatial_ref': (
            'PROJCS["unknown",GEOGCS["unknown",DATUM["unknown",'
            'SPHEROID["unknown",6378137,291.505347349177]],'
            'PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
            'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],'
            'PROJECTION["Polar_Stereographic"],'
            'PARAMETER["latitude_of_origin",60],'
            'PARAMETER["central_meridian",-80],PARAMETER["false_easting",0],'
            'PARAMETER["false_northing",0],'
            'UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
            'AXIS["Easting",SOUTH],AXIS["Northing",SOUTH]]'
            ),
        'grid_mapping_standard_parallel': 60,
        'grid_mapping_straight_vertical_longitude_from_pole': -80,
        'grid_mapping_units': 'm',
        'history': (
            '02 Apr 2025: First version of convert SAR dailt drift to NetCDF'
            ),
        'infoUrl': (
            'https://coastwatch.noaa.gov/cwn/products/'
            'sar-composite-arctic-imagery-normalized-radar-cross-section.html'
            ),
        'institution': 'NOAA CoastWatch',
        'instrument': (
                'See website for details'
                'https://coastwatch.noaa.gov/cwn/products/'
                'sar-composite-arctic-imagery-normalized-'
                'radar-cross-section.html'
                ),
        'keywords': (
            'Earth Science > Cryosphere > Sea Ice > Ice Extent, Earth Science '
            '> Cryosphere > Snow/Ice > Snow Cover, GEOGRAPHIC REGION > '
            'ARCTIC, GEOGRAPHIC REGION > NORTHERN HEMISPHERE, '
            'GEOGRAPHIC REGION > POLAR'
            ),
        'keywords_vocabulary': (
            'NASA Global Change Master Directory (GCMD) Keywords, Version 9.0'
            ),
        'license': 'CC-10',
        'metadata_link': (
            'https://coastwatch.noaa.gov/cwn/products/'
            'sar-composite-arctic-imagery-normalized-radar-cross-section.html'
            ),
        'naming_authority': 'gov.noaa.coastwatch',
        'ncei_template_version': 'NCEI_NetCDF_Grid_Template_v2.0',
        'platform': (
            'Earth Observation Satellites, In Situ Land-based Platforms, '
            'Ground Stations, Models/Analyses'
            ),
        'platform_vocabulary': (
            'NASA Global Change Master Directory (GCMD) Keywords, Version 9.0'
            ),
        'processing_level': 'NOAA Level 4',
        'product_version': 'Version 1',
        'project': (
            'SAR Daily Sea Ice Drift'
            ),
        'publisher_email': 'coastwatch at noaa dot gov',
        'publisher_institution': 'NOAA CoastWatch',
        'publisher_name':'NOAA CoastWatch',
        'publisher_type': 'Institutional',
        'publisher_url': 'coastwatch.noaa.gov',
        'references': (
                'See website for details'
                'https://coastwatch.noaa.gov/cwn/products/'
                'sar-composite-arctic-imagery-normalized-'
                'radar-cross-section.html'
                ),
        'source': 'SAR Daily Drift dataset',
        'sourceUrl': (
            'https://www.star.nesdis.noaa.gov/socd/mecb/sar/AKDEMO_products/'
            'COMPOSITE_TIFF/DRIFT_SHAPEFILES/'
            ),
        'standard_name_vocabulary': (
            'CF Standard Name Table (Version 72, 10 March 2020)'
            ),
        'summary': (
            'NetCDF version of SAR daily drift dataset. This includes the '
            'starting longitude and latitude, ending longitude and latitude '
            'start and end date of observation, the distance traveled in the '
            'observation period, and the bearing of the drift.'
            ),
        'title': 'SAR drift data converted from raw text form to NetCDF',
        'time_coverage_duration': 'P1D',
        'time_coverge_end': max_time_str,
        'time_coverage_resolution': 'P1D',
        'time_coverge_start': min_time_str,
            })
    
    return netcdf_grid