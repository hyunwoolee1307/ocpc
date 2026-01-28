#!/usr/bin/env python3
"""Compute and plot streamfunction (psi) and velocity potential (chi) from winds."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from windspharm.standard import VectorWind

ROOT = Path("/home/hyunwoo/ocpc")
FIG_DIR = ROOT / "results" / "figures"

DATASETS = {
    "ctrl": {
        "u": "/mnt/d/Data/processed/CFSv2/202601_FMA/forecast/ugrd.2026012100.2026020304.nc",
        "v": "/mnt/d/Data/processed/CFSv2/202601_FMA/forecast/vgrd.2026012100.2026020304.nc",
    },
    "anom": {
        "u": "/mnt/d/Data/processed/CFSv2/202601_FMA/forecast/a.ugrd.2026012100.2026020304.nc",
        "v": "/mnt/d/Data/processed/CFSv2/202601_FMA/forecast/a.vgrd.2026012100.2026020304.nc",
    },
}

EA_BOUNDS = dict(lat_min=20.0, lat_max=55.0, lon_min=110.0, lon_max=150.0)


def _open_uv(u_path: str, v_path: str) -> tuple[xr.DataArray, xr.DataArray]:
    u_ds = xr.open_dataset(u_path)
    v_ds = xr.open_dataset(v_path)
    u_da, v_da = xr.align(u_ds["ugrd"], v_ds["vgrd"], join="exact")
    if "sfc" in u_da.dims:
        u_da = u_da.isel(sfc=0)
    if "sfc" in v_da.dims:
        v_da = v_da.isel(sfc=0)
    return u_da, v_da


def _wrap_field(field: xr.DataArray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon = field["lon"].values
    lat = field["lat"].values
    lon_wrapped = ((lon + 180) % 360) - 180
    sort_idx = np.argsort(lon_wrapped)
    lon_wrapped = lon_wrapped[sort_idx]
    data_sorted = field.values[:, sort_idx]
    data_cyc, lon_cyc = add_cyclic_point(data_sorted, coord=lon_wrapped)
    return lat, lon_cyc, data_cyc


def _wrap_stream(u: np.ndarray, v: np.ndarray, lat: np.ndarray, lon: np.ndarray, step: int = 3):
    u_sub = u[::step, ::step]
    v_sub = v[::step, ::step]
    lon_sub = lon[::step]
    lat_sub = lat[::step]
    lon_wrapped = ((lon_sub + 180) % 360) - 180
    sort_idx = np.argsort(lon_wrapped)
    lon_wrapped = lon_wrapped[sort_idx]
    u_sorted = u_sub[:, sort_idx]
    v_sorted = v_sub[:, sort_idx]
    u_cyc, lon_cyc = add_cyclic_point(u_sorted, coord=lon_wrapped)
    v_cyc, _ = add_cyclic_point(v_sorted, coord=lon_wrapped)
    return lat_sub, lon_cyc, u_cyc, v_cyc


def _plot_pair(
    psi: xr.DataArray,
    chi: xr.DataArray,
    u_rot: np.ndarray,
    v_rot: np.ndarray,
    u_div: np.ndarray,
    v_div: np.ndarray,
    label: str,
    out_path: Path,
    global_view: bool,
    symmetric: bool,
) -> None:
    proj = ccrs.PlateCarree(central_longitude=205) if global_view else ccrs.PlateCarree()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={"projection": proj}, layout="constrained")

    for ax, field, title, u_stream, v_stream in [
        (axes[0], psi, "Streamfunction (ψ)", u_rot, v_rot),
        (axes[1], chi, "Velocity Potential (χ)", u_div, v_div),
    ]:
        if global_view:
            ax.set_global()
        ax.coastlines(linewidth=0.6)

        lat, lon_cyc, data_cyc = _wrap_field(field)
        if symmetric:
            vmax = float(np.nanmax(np.abs(data_cyc)))
            step = 0.25e6
            vmax = np.ceil(vmax / step) * step
            levels = np.arange(-vmax, vmax + step, step)
        else:
            vmin = float(np.nanmin(data_cyc))
            vmax = float(np.nanmax(data_cyc))
            vmin = np.floor(vmin / 1e6) * 1e6
            vmax = np.ceil(vmax / 1e6) * 1e6
            levels = np.arange(vmin, vmax + 1e6, 1e6)

        mesh = ax.contourf(
            lon_cyc,
            lat,
            data_cyc,
            transform=ccrs.PlateCarree(),
            levels=levels,
            cmap="RdBu_r",
            extend="both",
        )

        slat, slon, su, sv = _wrap_stream(u_stream, v_stream, field["lat"].values, field["lon"].values)
        ax.streamplot(slon, slat, su, sv, transform=ccrs.PlateCarree(), density=1.6, linewidth=0.7, color="k")

        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="0.6", alpha=0.6, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)
        cbar.set_label(f"{title} [m$^2$ s$^{-1}$]")
        ax.set_title(f"{label} · {title}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    for label, paths in DATASETS.items():
        u_da, v_da = _open_uv(paths["u"], paths["v"])
        lat = u_da["lat"].values
        lon = u_da["lon"].values

        u_vals = np.nan_to_num(u_da.values, nan=0.0)
        v_vals = np.nan_to_num(v_da.values, nan=0.0)
        vw = VectorWind(u_vals, v_vals)
        psi = vw.streamfunction()
        chi = vw.velocitypotential()
        u_rot, v_rot = vw.rotationalcomponent()
        u_div, v_div = vw.divergentcomponent()

        psi_da = xr.DataArray(psi, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
        chi_da = xr.DataArray(chi, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))

        _plot_pair(
            psi_da,
            chi_da,
            u_rot,
            v_rot,
            u_div,
            v_div,
            label=f"{label.upper()} Global",
            out_path=FIG_DIR / f"psi_chi_{label}_global.png",
            global_view=True,
            symmetric=(label == "anom"),
        )

        lat_min, lat_max = EA_BOUNDS["lat_min"], EA_BOUNDS["lat_max"]
        lon_min, lon_max = EA_BOUNDS["lon_min"], EA_BOUNDS["lon_max"]
        psi_ea = psi_da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        chi_ea = chi_da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

        lat_mask = (lat >= lat_min) & (lat <= lat_max)
        lon_mask = (lon >= lon_min) & (lon <= lon_max)
        u_rot_ea = u_rot[np.ix_(lat_mask, lon_mask)]
        v_rot_ea = v_rot[np.ix_(lat_mask, lon_mask)]
        u_div_ea = u_div[np.ix_(lat_mask, lon_mask)]
        v_div_ea = v_div[np.ix_(lat_mask, lon_mask)]

        _plot_pair(
            psi_ea,
            chi_ea,
            u_rot_ea,
            v_rot_ea,
            u_div_ea,
            v_div_ea,
            label=f"{label.upper()} East Asia",
            out_path=FIG_DIR / f"psi_chi_{label}_east_asia.png",
            global_view=False,
            symmetric=(label == "anom"),
        )


if __name__ == "__main__":
    main()
