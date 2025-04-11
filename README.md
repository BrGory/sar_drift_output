# SAR Drift Output Generator

## Project Overview

This project provides a tool to convert SAR drift data into:
- `.gpkg` (GeoPackage) files for visualization in GIS tools like QGIS
- `.nc` (NetCDF) files for scientific analysis and compliance with metadata standards

It supports Polar Stereographic projection (EPSG:3413) and includes metadata injection from CDL templates using `ncgen`.

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
conda install -c conda-forge netcdf-c netcdf4 nco
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
python sar_drift_output.py -i <input_file> [options]
```

### Required arguments:
- `-i`, `--input_filename`: Path to SAR drift `.txt` file

### Optional arguments:
- `-o`, `--output_dir`: Directory for `.nc` and `.gpkg` output (default: `output`)
- `-m`, `--metadata_dir`: Directory containing the CDL file (default: `meta`)
- `-cdl`, `--cdl_filename`: CDL file name (default: `sar_drift_output.cdl`)
- `-p`, `--precision`: Decimal precision for numeric output (default: 3)
- `-c`, `--compute`: Compute azimuth and distance using `pyproj` (vs. using original fields)
- `-v`, `--verbose`: Verbose logging

### Example:

```bash
python sar_drift_output.py -i data/sample_drift.txt -c -v
```

---

## Output

You will find:
- A `.nc` file in the output directory, CF/ACDD-compliant
- A `.gpkg` file with both point and line geometries for QGIS

---

## Notes

- Metadata in the `.cdl` file must use valid CDL syntax and declare dimensions and attributes cleanly.
- Placeholders like `FILL_DATE_CREATED` are automatically replaced based on the input file contents.

---

## License

This project is licensed under the MIT License.

---

## Contact

- Brendon Gory — [brendon.gory@noaa.gov](mailto:brendon.gory@noaa.gov)
- Dr. Prasanjit Dash — [prasanjit.dash@noaa.gov](mailto:prasanjit.dash@noaa.gov)