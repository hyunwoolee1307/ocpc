# Atmospheric Analysis Environment Setup (Project Standard)

This document defines the standard environment and workflow for atmospheric-variable analysis in the OCPC project.

## Scope

Applies to:
- Reanalysis/forecast wind analysis (ugrd/vgrd)
- Streamfunction/velocity potential diagnostics (ψ/χ)
- Regional subsets (e.g., East Asia 20–55.5°N, 110–150°E)

## Directory Layout

```
data/      # raw inputs (obs/reanalysis/model)
proc/      # preprocessed intermediates
clim/      # climatologies (monthly/seasonal)
fig/       # plots/figures
scripts/   # analysis scripts
notebooks/ # exploratory notebooks
env/       # environment files
docs/      # project documentation
```

## Preferred Tooling (Project Standard)

- **Statistics/EDA**: R  
- **Numerical computing (esp. linear algebra)**: Julia  
- **Visualization**: Python  

Use these defaults unless there is a clear technical reason to deviate (e.g., required library or performance constraints).

## Python Environment (Conda Recommended)

### Supported Python
- 3.11–3.13

### Required/Recommended Packages
- Core: `xarray`, `numpy`, `scipy`, `pandas`
- IO: `netCDF4`, `cfgrib`, `cftime`, `dask`
- Plotting: `matplotlib`, `cartopy`, `cmocean`
- Diagnostics: `windspharm`, `xgcm`, `xesmf`
- Utilities: `tqdm`, `rich`, `typer`

### Example Install
```
conda create -n ocpc_py313 python=3.13 -y
conda activate ocpc_py313
conda install -c conda-forge \
  xarray dask netcdf4 cfgrib cftime \
  numpy scipy pandas \
  matplotlib cartopy cmocean \
  windspharm xgcm xesmf \
  tqdm rich typer -y
```

## Data Standards

### Coordinates
- `lat`, `lon`
- `time` (cftime)
- `level` or `isobaricInhPa`

### Units (Recommended)
- Wind: m/s
- Temperature: K
- Pressure: Pa or hPa
- Precipitation: kg/m²/s or mm/day

### Longitude Convention
- Use either 0–360 or -180–180 consistently per workflow.

## Baseline Variables

Required:
- `ugrd`, `vgrd`

Recommended:
- `pres`/`mslp`, `tmp`/`tmp2m`, `q`/`rh`, `prate`
- Upper-air levels: 850/500/200 hPa

## Standard Pipeline

1. Quality checks (NaNs, ranges, units)
2. Regridding and alignment
3. Climatology computation (monthly/seasonal)
4. Anomaly calculation
5. Diagnostics (ψ/χ, vorticity, divergence)
6. Visualization (global + regional)

## Helmholtz Decomposition (ψ/χ)

Use `windspharm` for consistent ψ/χ and rotational/divergent wind components:
- ψ contours should align with rotational streamlines.
- Use the same grid/coordinates for both fields.

## R and Julia (Optional)

Some workflows may benefit from R or Julia. Use them as optional, secondary tools while keeping data formats consistent (NetCDF/GRIB) and reusing the same grid conventions.

### R (Suggested)
- Packages: `terra`, `stars`, `ncdf4`, `raster`, `sf`, `ggplot2`, `metR`, `fields`

### Julia (Suggested)
- Packages: `NCDatasets`, `NetCDF`, `ClimateBase`, `Interpolations`, `Plots`, `PyPlot`, `Cartopy` (via PyCall if needed)

### Interoperability
- Prefer NetCDF for cross-language exchange.
- Keep coordinate names `lat/lon/time/level` consistent.
- Document any unit conversions in scripts or metadata.

## Reproducibility

- Keep input/output paths in a config file.
- Embed metadata: period, grid, version, method.
- Name outputs with date/lead-time.
- Log script version + parameters.

## Recommended Templates

- `env/ocpc.yml`
- `scripts/preprocess.py`
- `scripts/compute_clim.py`
- `scripts/compute_anom.py`
- `scripts/plot_psi_chi.py`
