# SAR Drift Output Generator

## Project Overview

This project converts daily SAR-derived sea ice drift data into GIS-ready formats:

- **GeoPackage (`.gpkg`)**: Contains point and line geometries representing ice drift observations.
- **NetCDF (`.nc`)**: CF/ACDD-compliant output including spatial grid of drift vectors (`dx`, `dy`), speed, and bearing.
- **PNG Plot (`.png`)**: High-resolution plot showing the SAR backscatter image with drift vectors and an overview map of the Arctic region.

The tool supports data visualization in QGIS and other NetCDF/GIS software, and it aligns SAR imagery with derived vector data using EPSG:3413 (NSIDC Sea Ice Polar Stereographic North). Place the input SAR drift file and geotiff file in the input directory of your choice. The script will locate the GeoTIFF and SAR drift data in the directory and dynamically create a GeoPackage (for GIS software), a NetCDF file and a png file. These files show the SAR drift data as vectors with magnitude and direction disaplying on top of the supplied GeoTIFF image.

---

## Features

- Extracts and cleans drift data from `.txt` files
- Computes duration, distance, and azimuth per observation
- Converts lat/lon to projected `x/y` in meters using EPSG:3413
- Builds gridded NetCDF output with CF/ACDD metadata via `.cdl` files
- Generates GeoPackage output for GIS visualization

---

## Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
```

Or use a Conda environment (recommended):

```bash
conda env create -f environment.yml
conda activate sar_drift
```

### Dependencies

- numpy
- pandas
- xarray
- geopandas
- shapely
- pyproj
- netCDF4
- scipy

You also need `ncgen` from the [NetCDF-C tools](https://www.unidata.ucar.edu/software/netcdf/) for CDL file parsing.
```bash
conda install -c conda-forge netcdf-c nco
ncgen -h
ncdump -h
ncks --version
```
---

## Metadata Injection via CDL

The script supports injecting metadata from a CDL file. It uses:

- `ncgen` to convert a `.cdl` template into a `.nc` file
- `xarray` to read and apply global attributes
- Placeholder replacement for fields like `FILL_DATE_CREATED`, `FILL_MIN_TIME`

Ensure your `.cdl` file lives in a directory (default: `meta/`) and follows CF/ACDD conventions.

---

## Running the Script

```bash
python sar_drift_output.py [options]
```

### Optional arguments:
- `-i`, `--input_filename`: Path to delimited data file and geotiff image (default: `input` in script directory)
- `-o`, `--output_dir`: Directory for `.nc` and `.gpkg` output (default: `output` in script directory)
- `-m`, `--metadata_dir`: Directory containing the CDL file (default: `meta` in script directory)
- `-t`, `--input_file_type`: Extension of delimited input file (e.g. `txt` or `csv [not CaSe SeNsItIvE])
- `-d`, `--delimiter`: Character that separates the fields in the input data file (default: `,` [use `\t` for tab])
- `-p`, `--precision`: Decimal precision for numeric output (default: 3)
- `-v`, `--verbose`: Verbose logging

### Example:

```bash
python sar_drift_output.py -i input -o output -m meta -t txt -d \t -p 2 -v
```

---

## Input Files
- SAR Drift CSV: A .txt or .csv file with columns including:
    -  Lon1, Lat1, Lon2, Lat2: start and end coordinates
	-  Time1_JS, Time2_JS: Julian seconds since 2000-01-01
    -  U_vel_ms, V_vel_ms: velocity components
	-  Bear_deg, Speed_kmdy: bearing and speed
- GeoTIFF Image: Raster product of SAR backscatter with GCPs
- CDL File: Metadata template used to apply CF-compliant global and variable attributes to the NetCDF output
	
## Output
| Type                         | Description                                                         |
| ---------------------------- | ------------------------------------------------------------------- |
| `sar_drift_<timestamp>.gpkg` | QGIS GeoPackage with start points, end points, and drift lines      |
| `sar_drift_<timestamp>.nc`   | CF/ACDD-compliant NetCDF with drift variables (`dx`, `dy`, `Speed_kmdy`, `Bear_deg`)                      |
| `sar_drift_<timestamp>.png`  | Annotated PNG showing SAR image and vector overlays                 |
|							   |	  - Left: Arctic overview (Cartopy, EPSG:4326) with bounding box |
|							   |	  - Right: SAR image overlaid with quiver arrows, True North arrow, and scale bar             |

---

## Notes

- The NetCDF file uses EPSG:3413 with properly defined grid_mapping attributes.
- GeoTIFF is reprojected using Ground Control Points (GCPs) via rasterio.
- Vector drift observations are rendered using LineString geometries with corresponding start points.
- Placeholder CDL values like FILL_DATE_CREATED are auto-filled during export.
- For full documentation, see inline docstrings in sar_drift.py and util.py.

---

## Return codes
- 01: Cannot find input directory
- 02: Cannot find metadata directory
- 03: Input file type found in input directory
- 04: More than one input file type found in input directory
- 05: No .tif file found in input directory
- 06: More than one .tif file found in input directory
- 07: No .cdl file found in metadata directory
- 08: More than one .cdl file found in metadata directory
- 09: The delimiter character did not properly split the fields or the input file structure changed
- 10: Error running `ncgen` utility

---

## License

This project is licensed under the MIT License.

---

## Contact

- Brendon Gory — [brendon.gory@noaa.gov](mailto:brendon.gory@noaa.gov)
