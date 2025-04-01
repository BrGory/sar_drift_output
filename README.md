
# SAR Drift Output Generator

## Project Overview
This project provides a tool for converting SAR drift data into `.gpkg` (GeoPackage) and `.nc` (NetCDF) files, which can be visualized in programs like QGIS or any tool that supports NetCDF format.

## Requirements

This project has several dependencies that are listed in the `requirements.txt` file. You can use the `requirements.txt` file to install all necessary packages using either `pip` or `conda`.

### Python Dependencies
- numpy
- pandas
- xarray
- geopandas
- shapely
- pyproj
- netCDF4
- scipy

## Setting Up the Conda Environment

To set up the environment for this project, follow these steps:

### 1. Create a New Conda Environment

First, create a new Conda environment called `sar_drift` and install Python. You can choose the version of Python you want (e.g., `python=3.9`):

```bash
conda create --name sar_drift python=3.9
```

### 2. Activate the Environment

Activate the newly created environment:

```bash
conda activate sar_drift
```

### 3. Install the Required Packages

Once the environment is activated, you can install the dependencies listed in the `requirements.txt` file using `pip`:

```bash
pip install -r requirements.txt
```

Alternatively, if you want to use a Conda environment file, you can create an `environment.yml` file. Here's an example `environment.yml`:

```yaml
name: sar_drift
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - numpy
  - pandas
  - xarray
  - geopandas
  - shapely
  - pyproj
  - netCDF4
  - scipy
```

To create the environment using the YAML file:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate sar_drift
```

### 4. Run the Script

Once the environment is set up, you can run the `sar_drift_output.py` script by providing the necessary arguments:

```bash
python sar_drift_output.py -i <input_file> -o <output_directory> -p <precision> -v
```

Where:
- `-i` or `--input_filename` specifies the SAR drift data file (either `.txt` or `.nc`).
- `-o` or `--output_dir` specifies the output directory where `.gpkg` and `.nc` files will be saved.
- `-p` or `--precision` specifies the number of digits after the decimal point for precision (default is `3`).
- `-v` or `--verbose` enables verbose output for additional information during execution.

### 5. Generate the Output Files

The script will process the input SAR drift data, compute necessary parameters, and generate:
- A `.gpkg` file (GeoPackage) containing the processed drift data for visualization in QGIS.
- A `.nc` file (NetCDF) containing the drift data in a format suitable for use with other tools.

## License

This software is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For more information, please contact:
- Brendon Gory (brendon.gory@noaa.gov)
- Dr. Prasanjit Dash (prasanjit.dash@noaa.gov)
