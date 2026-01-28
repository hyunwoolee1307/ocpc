#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

psi_path = "/home/hyunwoo/ocpc/data/psi_anom_a.2026012100.2026020304.nc"
u_path = "/mnt/d/Data/processed/CFSv2/202601_FMA/forecast/a.ugrd.2026012100.2026020304.nc"
v_path = "/mnt/d/Data/processed/CFSv2/202601_FMA/forecast/a.vgrd.2026012100.2026020304.nc"

out_global = "/home/hyunwoo/ocpc/results/figures/psi_anom_a_global.png"
out_ea = "/home/hyunwoo/ocpc/results/figures/psi_anom_a_east_asia.png"

psi = xr.open_dataset(psi_path)["psi"]
u = xr.open_dataset(u_path)["ugrd"]
v = xr.open_dataset(v_path)["vgrd"]

if "sfc" in u.dims:
    u = u.isel(sfc=0)
if "sfc" in v.dims:
    v = v.isel(sfc=0)

# ensure lat/lon order
if tuple(psi.dims) != ("lat", "lon"):
    psi = psi.transpose("lat", "lon")
if tuple(u.dims) != ("lat", "lon"):
    u = u.transpose("lat", "lon")
if tuple(v.dims) != ("lat", "lon"):
    v = v.transpose("lat", "lon")


def wrap_field(field, global_view):
    lon = field["lon"].values
    lat = field["lat"].values
    if global_view:
        data_cyc, lon_cyc = add_cyclic_point(field.values, coord=lon)
        return lat, lon_cyc, data_cyc
    lon_wrapped = ((lon + 180) % 360) - 180
    sort_idx = np.argsort(lon_wrapped)
    lon_wrapped = lon_wrapped[sort_idx]
    data_sorted = field.values[:, sort_idx]
    data_cyc, lon_cyc = add_cyclic_point(data_sorted, coord=lon_wrapped)
    return lat, lon_cyc, data_cyc


def wrap_stream(u_field, v_field, global_view, step=3):
    u_sub = u_field.isel(lat=slice(None, None, step), lon=slice(None, None, step))
    v_sub = v_field.isel(lat=slice(None, None, step), lon=slice(None, None, step))
    lon = u_sub["lon"].values
    lat = u_sub["lat"].values
    if global_view:
        u_cyc, lon_cyc = add_cyclic_point(u_sub.values, coord=lon)
        v_cyc, _ = add_cyclic_point(v_sub.values, coord=lon)
        return lat, lon_cyc, u_cyc, v_cyc
    lon_wrapped = ((lon + 180) % 360) - 180
    sort_idx = np.argsort(lon_wrapped)
    lon_wrapped = lon_wrapped[sort_idx]
    u_sorted = u_sub.values[:, sort_idx]
    v_sorted = v_sub.values[:, sort_idx]
    u_cyc, lon_cyc = add_cyclic_point(u_sorted, coord=lon_wrapped)
    v_cyc, _ = add_cyclic_point(v_sorted, coord=lon_wrapped)
    return lat, lon_cyc, u_cyc, v_cyc


def plot_panel(field, u_field, v_field, title, out_path, global_view=True):
    if global_view:
        lat, lon_cyc, data_cyc = wrap_field(field, global_view=True)
        slat, slon, su, sv = wrap_stream(u_field, v_field, global_view=True)
        plot_lon = lon_cyc
        plot_lat = lat
        plot_data = data_cyc
    else:
        plot_lat = field["lat"].values
        plot_lon = field["lon"].values
        plot_data = field.values
        slat = u_field["lat"].values
        slon = u_field["lon"].values
        su = u_field.values
        sv = v_field.values

    vmax = np.nanmax(np.abs(plot_data))
    levels = np.linspace(-vmax, vmax, 21)

    proj = ccrs.PlateCarree(central_longitude=205) if global_view else ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(12, 5), subplot_kw={"projection": proj}, layout="constrained")
    if global_view:
        ax.set_global()
    ax.coastlines(linewidth=0.6)
    mesh = ax.contourf(
        plot_lon,
        plot_lat,
        plot_data,
        transform=ccrs.PlateCarree(),
        levels=levels,
        cmap="RdBu_r",
        extend="both",
    )
    ax.streamplot(slon, slat, su, sv, transform=ccrs.PlateCarree(), density=1.6, linewidth=0.7, color="k")
    if not global_view:
        ax.set_extent([110, 150, 20, 55], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="0.6", alpha=0.6, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)
    cbar.set_label("Streamfunction (m$^2$ s$^{-1}$)")
    ax.set_title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


plot_panel(psi, u, v, "Anomaly Streamfunction (Global)", Path(out_global), global_view=True)

psi_ea = psi.sel(lat=slice(20, 55), lon=slice(110, 150))
u_ea = u.sel(lat=slice(20, 55), lon=slice(110, 150))
v_ea = v.sel(lat=slice(20, 55), lon=slice(110, 150))
plot_panel(psi_ea, u_ea, v_ea, "Anomaly Streamfunction (East Asia)", Path(out_ea), global_view=False)
