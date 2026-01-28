#!/usr/bin/env python3
"""Plot FMA streamfunction contours with rotational wind streamlines (windspharm)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from windspharm.standard import VectorWind

U_PATH = "/mnt/d/Data/processed/CFSv2/202601_FMA/forecast/a.ugrd.2026012100.2026020304.nc"
V_PATH = "/mnt/d/Data/processed/CFSv2/202601_FMA/forecast/a.vgrd.2026012100.2026020304.nc"

OUT_GLOBAL = Path("/home/hyunwoo/ocpc/results/figures/streamfunction.a.2026012100.2026020304.rotparallel.png")
OUT_EA = Path(
    "/home/hyunwoo/ocpc/results/figures/streamfunction.a.2026012100.2026020304.rotparallel.subset_20-55.5N_110-150E.png"
)


def _wrap_field(field: xr.DataArray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon = field["lon"].values
    lat = field["lat"].values
    lon_wrapped = ((lon + 180) % 360) - 180
    sort_idx = np.argsort(lon_wrapped)
    lon_wrapped = lon_wrapped[sort_idx]
    data_sorted = field.values[:, sort_idx]
    data_cyc, lon_cyc = add_cyclic_point(data_sorted, coord=lon_wrapped)
    return lat, lon_cyc, data_cyc


def _wrap_stream(u: np.ndarray, v: np.ndarray, lat: np.ndarray, lon: np.ndarray, step: int = 2):
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


def main() -> None:
    u_ds = xr.open_dataset(U_PATH)
    v_ds = xr.open_dataset(V_PATH)
    u_da, v_da = xr.align(u_ds["ugrd"], v_ds["vgrd"], join="exact")

    if "sfc" in u_da.dims:
        u_da = u_da.isel(sfc=0)
    if "sfc" in v_da.dims:
        v_da = v_da.isel(sfc=0)

    lat = u_da["lat"].values
    lon = u_da["lon"].values

    u_vals = np.nan_to_num(u_da.values, nan=0.0)
    v_vals = np.nan_to_num(v_da.values, nan=0.0)
    vw = VectorWind(u_vals, v_vals)
    psi = vw.streamfunction()
    u_rot, v_rot = vw.rotationalcomponent()

    psi_da = xr.DataArray(psi, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))

    levels = np.linspace(-6e6, 6e6, 13)
    tick_abs = np.array([-6, -4, -2, 0, 2, 4, 6]) * 1e6

    # Global
    lat_g, lon_cyc, psi_cyc = _wrap_field(psi_da)
    slat, slon, su, sv = _wrap_stream(u_rot, v_rot, lat, lon, step=2)

    fig, ax = plt.subplots(
        figsize=(12, 5),
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=205)},
        layout="constrained",
    )
    ax.set_global()
    ax.coastlines(linewidth=0.6)
    mesh = ax.contourf(
        lon_cyc,
        lat_g,
        psi_cyc,
        transform=ccrs.PlateCarree(),
        levels=levels,
        cmap="RdBu_r",
        extend="both",
    )
    ax.streamplot(slon, slat, su, sv, transform=ccrs.PlateCarree(), density=2.0, linewidth=0.7, color="k")
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="0.6", alpha=0.6, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.4)
    cbar.set_label("Streamfunction (m$^2$ s$^{-1}$)")
    cbar.set_ticks(tick_abs)
    ax.set_title("Global Streamfunction FMA")
    OUT_GLOBAL.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_GLOBAL, dpi=150)
    plt.close(fig)

    # East Asia
    lat_min, lat_max = 20.0, 55.5
    lon_min, lon_max = 110.0, 150.0
    psi_sel = psi_da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    lon_mask = (lon >= lon_min) & (lon <= lon_max)
    u_ea = u_rot[np.ix_(lat_mask, lon_mask)]
    v_ea = v_rot[np.ix_(lat_mask, lon_mask)]
    lat_ea = lat[lat_mask]
    lon_ea = lon[lon_mask]

    fig, ax = plt.subplots(figsize=(7, 5), subplot_kw={"projection": ccrs.PlateCarree()}, layout="constrained")
    ax.set_extent([110, 150, 20, 55.5], crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.6)
    mesh = ax.contourf(
        psi_sel["lon"],
        psi_sel["lat"],
        psi_sel.values,
        transform=ccrs.PlateCarree(),
        levels=levels,
        cmap="RdBu_r",
        extend="both",
    )
    step = 2
    ax.streamplot(
        lon_ea[::step],
        lat_ea[::step],
        u_ea[::step, ::step],
        v_ea[::step, ::step],
        transform=ccrs.PlateCarree(),
        density=2.0,
        linewidth=0.7,
        color="k",
    )
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="0.6", alpha=0.6, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.4)
    cbar.set_label("Streamfunction (m$^2$ s$^{-1}$)")
    cbar.set_ticks(tick_abs)
    ax.set_title("East Asia Streamfunction")
    OUT_EA.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_EA, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
