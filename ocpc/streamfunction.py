"""Compute a simple streamfunction from gridded winds.

Assumptions
-----------
This module treats the wind field as approximately non-divergent and
computes a streamfunction using the kinematic relationships:

    u =  dψ/dy
    v = -dψ/dx

The streamfunction is estimated by integrating v along longitude and u
along latitude on a sphere and then averaging the two integrals. This is
*not* a full Poisson solver; it is a lightweight diagnostic intended for
quick-look analysis and synthetic test cases.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

EARTH_RADIUS_M = 6_371_000.0


def _integrate_along_lon(v: np.ndarray, lat_rad: np.ndarray, lon_rad: np.ndarray) -> np.ndarray:
    """Integrate v along longitude to estimate streamfunction.

    Parameters
    ----------
    v
        Meridional wind array with shape (sfc, lat, lon).
    lat_rad
        Latitude values in radians, shape (lat,).
    lon_rad
        Longitude values in radians, shape (lon,).
    """
    dlon = np.diff(lon_rad)
    cos_lat = np.cos(lat_rad)
    dx = EARTH_RADIUS_M * cos_lat[:, np.newaxis] * dlon[np.newaxis, :]
    increments = -v[:, :, :-1] * dx[np.newaxis, :, :]
    psi = np.zeros_like(v, dtype=float)
    psi[:, :, 1:] = np.cumsum(increments, axis=2)
    return psi


def _integrate_along_lat(u: np.ndarray, lat_rad: np.ndarray) -> np.ndarray:
    """Integrate u along latitude to estimate streamfunction.

    Parameters
    ----------
    u
        Zonal wind array with shape (sfc, lat, lon).
    lat_rad
        Latitude values in radians, shape (lat,).
    """
    dlat = np.diff(lat_rad)
    dy = EARTH_RADIUS_M * dlat
    increments = u[:, :-1, :] * dy[np.newaxis, :, np.newaxis]
    psi = np.zeros_like(u, dtype=float)
    psi[:, 1:, :] = np.cumsum(increments, axis=1)
    return psi


def compute_streamfunction(ugrd: np.ndarray, vgrd: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Compute a simple streamfunction from gridded winds.

    Parameters
    ----------
    ugrd
        Zonal wind component with shape (sfc, lat, lon).
    vgrd
        Meridional wind component with shape (sfc, lat, lon).
    lat
        Latitude array in degrees, shape (lat,), ranging from -90 to 90.
    lon
        Longitude array in degrees, shape (lon,), ranging from 0 to 360.

    Returns
    -------
    numpy.ndarray
        Streamfunction estimate with shape (sfc, lat, lon).
    """
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    psi_x = _integrate_along_lon(vgrd, lat_rad, lon_rad)
    psi_y = _integrate_along_lat(ugrd, lat_rad)
    return 0.5 * (psi_x + psi_y)


def load_streamfunction(path: str | Path) -> xr.DataArray:
    """Load winds from a NetCDF file and compute the streamfunction.

    The NetCDF file must contain variables named ``ugrd`` and ``vgrd`` with
    dimensions (sfc, lat, lon).
    """
    dataset = xr.open_dataset(path)
    ugrd = dataset["ugrd"].values
    vgrd = dataset["vgrd"].values
    lat = dataset["lat"].values
    lon = dataset["lon"].values
    stream = compute_streamfunction(ugrd, vgrd, lat, lon)
    return xr.DataArray(
        stream,
        coords={"sfc": dataset["sfc"].values, "lat": lat, "lon": lon},
        dims=("sfc", "lat", "lon"),
        name="streamfunction",
        attrs={"long_name": "simple streamfunction", "units": "m^2/s"},
    )


def main() -> None:
    """CLI entry point for quick-look streamfunction computation."""
    parser = argparse.ArgumentParser(description="Compute a simple streamfunction from ugrd/vgrd.")
    parser.add_argument("input", help="Path to NetCDF file containing ugrd/vgrd.")
    parser.add_argument("--output", help="Optional output NetCDF path to save streamfunction.")
    args = parser.parse_args()

    stream = load_streamfunction(args.input)
    if args.output:
        stream.to_dataset(name="streamfunction").to_netcdf(args.output)
    else:
        print(stream)


if __name__ == "__main__":
    main()
