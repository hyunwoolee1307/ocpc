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
        Meridional wind array with shape (..., lat, lon).
    lat_rad
        Latitude values in radians, shape (lat,).
    lon_rad
        Longitude values in radians, shape (lon,).
    """
    dlon = np.diff(lon_rad)
    cos_lat = np.cos(lat_rad)
    dx = EARTH_RADIUS_M * cos_lat[:, np.newaxis] * dlon[np.newaxis, :]
    increments = -v[..., :-1] * dx
    increments = np.nan_to_num(increments, nan=0.0)

    psi_fwd = np.zeros_like(v, dtype=float)
    psi_fwd[..., 1:] = np.cumsum(increments, axis=-1)

    psi_bwd = np.zeros_like(v, dtype=float)
    psi_bwd[..., :-1] = -np.cumsum(increments[..., ::-1], axis=-1)[..., ::-1]

    return 0.5 * (psi_fwd + psi_bwd)


def _integrate_along_lat(u: np.ndarray, lat_rad: np.ndarray) -> np.ndarray:
    """Integrate u along latitude to estimate streamfunction.

    Parameters
    ----------
    u
        Zonal wind array with shape (..., lat, lon).
    lat_rad
        Latitude values in radians, shape (lat,).
    """
    dlat = np.diff(lat_rad)
    dy = EARTH_RADIUS_M * dlat
    dy_shape = (1,) * (u.ndim - 2) + (dy.size, 1)
    increments = u[..., :-1, :] * dy.reshape(dy_shape)
    increments = np.nan_to_num(increments, nan=0.0)

    psi_fwd = np.zeros_like(u, dtype=float)
    psi_fwd[..., 1:, :] = np.cumsum(increments, axis=-2)

    psi_bwd = np.zeros_like(u, dtype=float)
    psi_bwd[..., :-1, :] = -np.cumsum(increments[..., ::-1, :], axis=-2)[..., ::-1, :]

    return 0.5 * (psi_fwd + psi_bwd)


def _compute_vorticity(ugrd: np.ndarray, vgrd: np.ndarray, lat_rad: np.ndarray, lon_rad: np.ndarray) -> np.ndarray:
    """Compute relative vorticity on a sphere from wind components."""
    dlon = float(np.gradient(lon_rad).mean())
    dlat = float(np.gradient(lat_rad).mean())
    cos_lat = np.cos(lat_rad)
    cos_lat = np.where(np.abs(cos_lat) < 1e-6, 1e-6, cos_lat)

    dv_dlon = np.gradient(vgrd, dlon, axis=-1)
    du_dlat = np.gradient(ugrd, dlat, axis=-2)

    dv_dx = dv_dlon / (EARTH_RADIUS_M * cos_lat[..., np.newaxis])
    du_dy = du_dlat / EARTH_RADIUS_M
    return dv_dx - du_dy


def _compute_divergence(ugrd: np.ndarray, vgrd: np.ndarray, lat_rad: np.ndarray, lon_rad: np.ndarray) -> np.ndarray:
    """Compute divergence on a sphere from wind components."""
    dlon = float(np.gradient(lon_rad).mean())
    dlat = float(np.gradient(lat_rad).mean())
    cos_lat = np.cos(lat_rad)
    cos_lat = np.where(np.abs(cos_lat) < 1e-6, 1e-6, cos_lat)

    du_dlon = np.gradient(ugrd, dlon, axis=-1)
    v_cos = vgrd * cos_lat[..., np.newaxis]
    dvcos_dlat = np.gradient(v_cos, dlat, axis=-2)

    return (du_dlon + dvcos_dlat) / (EARTH_RADIUS_M * cos_lat[..., np.newaxis])


def _solve_poisson_on_sphere(zeta: np.ndarray, lat_rad: np.ndarray, lon_rad: np.ndarray) -> np.ndarray:
    """Solve ∇²ψ = ζ on a sphere with periodic lon and Neumann at the poles."""
    nlat, nlon = zeta.shape[-2], zeta.shape[-1]
    dphi = float(np.gradient(lat_rad).mean())
    cos_lat = np.cos(lat_rad)
    cos_lat = np.where(np.abs(cos_lat) < 1e-6, 1e-6, cos_lat)
    cos_half = 0.5 * (cos_lat[:-1] + cos_lat[1:])

    # FFT along longitude (periodic).
    zeta_hat = np.fft.fft(zeta, axis=-1)
    psi_hat = np.zeros_like(zeta_hat, dtype=complex)

    # Flatten leading dimensions for batched solves.
    lead_shape = zeta_hat.shape[:-2]
    nlead = int(np.prod(lead_shape)) if lead_shape else 1
    zeta_hat_flat = zeta_hat.reshape((nlead, nlat, nlon))
    psi_hat_flat = psi_hat.reshape((nlead, nlat, nlon))

    a2 = EARTH_RADIUS_M * EARTH_RADIUS_M

    for m in range(nlon):
        k = m if m <= nlon // 2 else m - nlon
        k2 = float(k * k)

        A = np.zeros(nlat, dtype=float)
        B = np.zeros(nlat, dtype=float)
        C = np.zeros(nlat, dtype=float)

        for i in range(1, nlat - 1):
            A[i] = cos_half[i - 1] / (a2 * cos_lat[i] * dphi * dphi)
            C[i] = cos_half[i] / (a2 * cos_lat[i] * dphi * dphi)
            B[i] = -(A[i] + C[i]) - (k2 / (a2 * cos_lat[i] * cos_lat[i]))

        # Neumann boundaries via mirrored ghost points.
        A0 = cos_lat[0] / (a2 * cos_lat[0] * dphi * dphi)
        C0 = cos_half[0] / (a2 * cos_lat[0] * dphi * dphi)
        B0 = -(A0 + C0) - (k2 / (a2 * cos_lat[0] * cos_lat[0]))
        C0 += A0
        A0 = 0.0

        A1 = cos_half[-1] / (a2 * cos_lat[-1] * dphi * dphi)
        C1 = cos_lat[-1] / (a2 * cos_lat[-1] * dphi * dphi)
        B1 = -(A1 + C1) - (k2 / (a2 * cos_lat[-1] * cos_lat[-1]))
        A1 += C1
        C1 = 0.0

        A[0], B[0], C[0] = A0, B0, C0
        A[-1], B[-1], C[-1] = A1, B1, C1

        if k == 0:
            B += 1e-12  # regularize nullspace

        D = zeta_hat_flat[:, :, m]
        cp = np.zeros(nlat, dtype=complex)
        dp = np.zeros((nlead, nlat), dtype=complex)

        cp[0] = C[0] / B[0]
        dp[:, 0] = D[:, 0] / B[0]
        for i in range(1, nlat):
            denom = B[i] - A[i] * cp[i - 1]
            cp[i] = C[i] / denom
            dp[:, i] = (D[:, i] - A[i] * dp[:, i - 1]) / denom

        sol = np.zeros((nlead, nlat), dtype=complex)
        sol[:, -1] = dp[:, -1]
        for i in range(nlat - 2, -1, -1):
            sol[:, i] = dp[:, i] - cp[i] * sol[:, i + 1]

        psi_hat_flat[:, :, m] = sol

    psi = np.fft.ifft(psi_hat, axis=-1).real
    return psi


def _solve_poisson_on_sphere_scipy(zeta: np.ndarray, lat_rad: np.ndarray, lon_rad: np.ndarray) -> np.ndarray:
    """Solve ∇²ψ = ζ on a sphere using SciPy banded solvers."""
    from scipy import fft as sp_fft
    from scipy import linalg as sp_linalg

    nlat, nlon = zeta.shape[-2], zeta.shape[-1]
    dphi = float(np.gradient(lat_rad).mean())
    cos_lat = np.cos(lat_rad)
    cos_lat = np.where(np.abs(cos_lat) < 1e-6, 1e-6, cos_lat)
    cos_half = 0.5 * (cos_lat[:-1] + cos_lat[1:])

    # FFT along longitude (periodic).
    zeta_hat = sp_fft.rfft(zeta, axis=-1)
    psi_hat = np.zeros_like(zeta_hat, dtype=complex)

    lead_shape = zeta_hat.shape[:-2]
    nlead = int(np.prod(lead_shape)) if lead_shape else 1
    zeta_hat_flat = zeta_hat.reshape((nlead, nlat, zeta_hat.shape[-1]))
    psi_hat_flat = psi_hat.reshape((nlead, nlat, zeta_hat.shape[-1]))

    a2 = EARTH_RADIUS_M * EARTH_RADIUS_M

    for m in range(zeta_hat.shape[-1]):
        k2 = float(m * m)

        A = np.zeros(nlat, dtype=float)
        B = np.zeros(nlat, dtype=float)
        C = np.zeros(nlat, dtype=float)

        for i in range(1, nlat - 1):
            A[i] = cos_half[i - 1] / (a2 * cos_lat[i] * dphi * dphi)
            C[i] = cos_half[i] / (a2 * cos_lat[i] * dphi * dphi)
            B[i] = -(A[i] + C[i]) - (k2 / (a2 * cos_lat[i] * cos_lat[i]))

        A0 = cos_lat[0] / (a2 * cos_lat[0] * dphi * dphi)
        C0 = cos_half[0] / (a2 * cos_lat[0] * dphi * dphi)
        B0 = -(A0 + C0) - (k2 / (a2 * cos_lat[0] * cos_lat[0]))
        C0 += A0
        A0 = 0.0

        A1 = cos_half[-1] / (a2 * cos_lat[-1] * dphi * dphi)
        C1 = cos_lat[-1] / (a2 * cos_lat[-1] * dphi * dphi)
        B1 = -(A1 + C1) - (k2 / (a2 * cos_lat[-1] * cos_lat[-1]))
        A1 += C1
        C1 = 0.0

        A[0], B[0], C[0] = A0, B0, C0
        A[-1], B[-1], C[-1] = A1, B1, C1

        if m == 0:
            B += 1e-12

        ab = np.zeros((3, nlat), dtype=float)
        ab[0, 1:] = C[:-1]
        ab[1, :] = B
        ab[2, :-1] = A[1:]

        for lead in range(nlead):
            D = zeta_hat_flat[lead, :, m]
            sol = sp_linalg.solve_banded((1, 1), ab, D)
            psi_hat_flat[lead, :, m] = sol

    psi = sp_fft.irfft(psi_hat, n=nlon, axis=-1).real
    return psi


def compute_streamfunction(
    ugrd: np.ndarray, vgrd: np.ndarray, lat: np.ndarray, lon: np.ndarray, method: str = "scipy"
) -> np.ndarray:
    """Compute a simple streamfunction from gridded winds.

    Parameters
    ----------
    ugrd
        Zonal wind component with shape (..., lat, lon).
    vgrd
        Meridional wind component with shape (..., lat, lon).
    lat
        Latitude array in degrees, shape (lat,), ranging from -90 to 90.
    lon
        Longitude array in degrees, shape (lon,), ranging from 0 to 360.

    Returns
    -------
    numpy.ndarray
        Streamfunction estimate with shape (..., lat, lon).
    """
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    mask = np.isnan(ugrd) | np.isnan(vgrd)
    u_clean = np.nan_to_num(ugrd, nan=0.0)
    v_clean = np.nan_to_num(vgrd, nan=0.0)

    if method == "integral":
        psi_x = _integrate_along_lon(v_clean, lat_rad, lon_rad)
        psi_y = _integrate_along_lat(u_clean, lat_rad)
        stream = 0.5 * (psi_x + psi_y)
        stream = stream - np.nanmean(stream, axis=-1, keepdims=True)
        stream = stream - np.nanmean(stream, axis=-2, keepdims=True)
    elif method == "poisson":
        zeta = _compute_vorticity(u_clean, v_clean, lat_rad, lon_rad)
        stream = _solve_poisson_on_sphere(zeta, lat_rad, lon_rad)
        stream = stream - np.nanmean(stream, axis=-1, keepdims=True)
    elif method == "scipy":
        zeta = _compute_vorticity(u_clean, v_clean, lat_rad, lon_rad)
        stream = _solve_poisson_on_sphere_scipy(zeta, lat_rad, lon_rad)
        stream = stream - np.nanmean(stream, axis=-1, keepdims=True)
    else:
        raise ValueError(f"Unknown method '{method}'. Expected 'integral', 'poisson', or 'scipy'.")

    return np.where(mask, np.nan, stream)


def compute_wind_decomposition(
    ugrd: np.ndarray, vgrd: np.ndarray, lat: np.ndarray, lon: np.ndarray, method: str = "scipy"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Helmholtz decomposition of winds on a sphere.

    Returns
    -------
    tuple of numpy.ndarray
        (psi, chi, u_rot, v_rot, u_div, v_div)
    """
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    mask = np.isnan(ugrd) | np.isnan(vgrd)
    u_clean = np.nan_to_num(ugrd, nan=0.0)
    v_clean = np.nan_to_num(vgrd, nan=0.0)

    zeta = _compute_vorticity(u_clean, v_clean, lat_rad, lon_rad)
    div = _compute_divergence(u_clean, v_clean, lat_rad, lon_rad)

    if method == "poisson":
        psi = _solve_poisson_on_sphere(zeta, lat_rad, lon_rad)
        chi = _solve_poisson_on_sphere(div, lat_rad, lon_rad)
    elif method == "scipy":
        psi = _solve_poisson_on_sphere_scipy(zeta, lat_rad, lon_rad)
        chi = _solve_poisson_on_sphere_scipy(div, lat_rad, lon_rad)
    else:
        raise ValueError(f"Unknown method '{method}'. Expected 'poisson' or 'scipy'.")

    dlon = float(np.gradient(lon_rad).mean())
    dlat = float(np.gradient(lat_rad).mean())
    cos_lat = np.cos(lat_rad)
    cos_lat = np.where(np.abs(cos_lat) < 1e-6, 1e-6, cos_lat)

    dpsi_dlat = np.gradient(psi, dlat, axis=-2)
    dpsi_dlon = np.gradient(psi, dlon, axis=-1)
    dchi_dlat = np.gradient(chi, dlat, axis=-2)
    dchi_dlon = np.gradient(chi, dlon, axis=-1)

    u_rot = -dpsi_dlat / EARTH_RADIUS_M
    v_rot = dpsi_dlon / (EARTH_RADIUS_M * cos_lat[..., np.newaxis])
    u_div = dchi_dlon / (EARTH_RADIUS_M * cos_lat[..., np.newaxis])
    v_div = dchi_dlat / EARTH_RADIUS_M

    psi = np.where(mask, np.nan, psi)
    chi = np.where(mask, np.nan, chi)
    u_rot = np.where(mask, np.nan, u_rot)
    v_rot = np.where(mask, np.nan, v_rot)
    u_div = np.where(mask, np.nan, u_div)
    v_div = np.where(mask, np.nan, v_div)

    return psi, chi, u_rot, v_rot, u_div, v_div


def _build_streamfunction_da(stream: np.ndarray, template: xr.DataArray) -> xr.DataArray:
    """Build a streamfunction DataArray with consistent metadata."""
    return xr.DataArray(
        stream,
        coords=template.coords,
        dims=template.dims,
        name="streamfunction",
        attrs={"long_name": "simple streamfunction", "units": "m^2/s"},
    )


def load_streamfunction(path: str | Path, method: str = "scipy") -> xr.DataArray:
    """Load winds from a NetCDF file and compute the streamfunction.

    The NetCDF file must contain variables named ``ugrd`` and ``vgrd`` with
    dimensions (sfc, lat, lon).
    """
    dataset = xr.open_dataset(path)
    u_da = dataset["ugrd"]
    v_da = dataset["vgrd"]
    lat = u_da["lat"].values
    lon = u_da["lon"].values
    stream = compute_streamfunction(u_da.values, v_da.values, lat, lon, method=method)
    return _build_streamfunction_da(stream, u_da)


def load_streamfunction_from_files(
    ugrd_path: str | Path, vgrd_path: str | Path, method: str = "scipy"
) -> xr.DataArray:
    """Load winds from separate NetCDF files and compute the streamfunction."""
    u_dataset = xr.open_dataset(ugrd_path)
    v_dataset = xr.open_dataset(vgrd_path)
    if "ugrd" not in u_dataset:
        raise KeyError(f"Expected 'ugrd' in {ugrd_path}")
    if "vgrd" not in v_dataset:
        raise KeyError(f"Expected 'vgrd' in {vgrd_path}")
    u_da, v_da = xr.align(u_dataset["ugrd"], v_dataset["vgrd"], join="exact")
    lat = u_da["lat"].values
    lon = u_da["lon"].values
    stream = compute_streamfunction(u_da.values, v_da.values, lat, lon, method=method)
    return _build_streamfunction_da(stream, u_da)


def main() -> None:
    """CLI entry point for quick-look streamfunction computation."""
    parser = argparse.ArgumentParser(description="Compute a simple streamfunction from ugrd/vgrd.")
    parser.add_argument(
        "input",
        nargs="+",
        help="Path to NetCDF file containing ugrd/vgrd, or two paths for ugrd then vgrd.",
    )
    parser.add_argument(
        "--method",
        choices=["poisson", "integral", "scipy"],
        default="scipy",
        help="Computation method (default: scipy).",
    )
    parser.add_argument("--output", help="Optional output NetCDF path to save streamfunction.")
    args = parser.parse_args()

    if len(args.input) == 1:
        stream = load_streamfunction(args.input[0], method=args.method)
    elif len(args.input) == 2:
        stream = load_streamfunction_from_files(args.input[0], args.input[1], method=args.method)
    else:
        parser.error("Expected one input file or two input files (ugrd then vgrd).")
    if args.output:
        stream.to_dataset(name="streamfunction").to_netcdf(args.output)
    else:
        print(stream)


if __name__ == "__main__":
    main()
