import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from pathlib import Path

from ocpc.streamfunction import compute_streamfunction

U_PATH = "/mnt/d/Data/processed/CFSv2/202601_FMA/forecast/ugrd.2026012100.2026020304.nc"
V_PATH = "/mnt/d/Data/processed/CFSv2/202601_FMA/forecast/vgrd.2026012100.2026020304.nc"

OUT_GLOBAL = Path("/home/hyunwoo/ocpc/results/figures/rotational_poisson_global.png")
OUT_EA = Path("/home/hyunwoo/ocpc/results/figures/rotational_poisson_east_asia.png")

R = 6.371e6

u = xr.open_dataset(U_PATH)["ugrd"]
v = xr.open_dataset(V_PATH)["vgrd"]

if "sfc" in u.dims:
    u = u.isel(sfc=0)
if "sfc" in v.dims:
    v = v.isel(sfc=0)

u = u.transpose("lat", "lon")
v = v.transpose("lat", "lon")

lat = u["lat"].values
lon = u["lon"].values

psi = compute_streamfunction(u.values, v.values, lat, lon, method="scipy")

# u = -dψ/dy, v = dψ/dx on sphere
lat_rad = np.deg2rad(lat)
lon_rad = np.deg2rad(lon)
dphi = float(np.gradient(lat_rad).mean())
dlon = float(np.gradient(lon_rad).mean())
coslat = np.cos(lat_rad)
coslat = np.where(np.abs(coslat) < 1e-6, 1e-6, coslat)

dpsi_dphi = np.gradient(psi, dphi, axis=0)
dpsi_dlon = np.gradient(psi, dlon, axis=1)

u_rot = -(1.0 / R) * dpsi_dphi
v_rot = (1.0 / (R * coslat[:, None])) * dpsi_dlon

psi_da = xr.DataArray(psi, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))


def wrap_field(field):
    lon = field["lon"].values
    lat = field["lat"].values
    data_cyc, lon_cyc = add_cyclic_point(field.values, coord=lon)
    return lat, lon_cyc, data_cyc


def wrap_stream(u_vals, v_vals, step=3):
    u_sub = u_vals[::step, ::step]
    v_sub = v_vals[::step, ::step]
    lon_sub = lon[::step]
    lat_sub = lat[::step]
    u_cyc, lon_cyc = add_cyclic_point(u_sub, coord=lon_sub)
    v_cyc, _ = add_cyclic_point(v_sub, coord=lon_sub)
    return lat_sub, lon_cyc, u_cyc, v_cyc


# Global plot
lat_g, lon_cyc, psi_cyc = wrap_field(psi_da)
slat, slon, su, sv = wrap_stream(u_rot, v_rot)

vmax = np.nanmax(np.abs(psi_cyc))
step = 0.5e6
vmax = np.ceil(vmax / step) * step
levels = np.arange(-vmax, vmax + step, step)

fig, ax = plt.subplots(figsize=(12, 5), subplot_kw={"projection": ccrs.PlateCarree(central_longitude=205)}, layout="constrained")
ax.set_global(); ax.coastlines(linewidth=0.6)
mesh = ax.contourf(lon_cyc, lat_g, psi_cyc, transform=ccrs.PlateCarree(), levels=levels, cmap="RdBu_r", extend="both")
ax.streamplot(slon, slat, su, sv, transform=ccrs.PlateCarree(), density=1.6, linewidth=0.7, color="k")

cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)
cbar.set_label("Streamfunction ψ (m$^2$ s$^{-1}$)")
ax.set_title("Streamfunction & Streamlines [Global]")

OUT_GLOBAL.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_GLOBAL, dpi=150)
plt.close(fig)

# East Asia
lat_min, lat_max = 20.0, 55.5
lon_min, lon_max = 110.0, 150.0
psi_ea = psi_da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

lat_mask = (lat >= lat_min) & (lat <= lat_max)
lon_mask = (lon >= lon_min) & (lon <= lon_max)

u_rot_ea = u_rot[np.ix_(lat_mask, lon_mask)]
v_rot_ea = v_rot[np.ix_(lat_mask, lon_mask)]
lat_ea = lat[lat_mask]
lon_ea = lon[lon_mask]

fig, ax = plt.subplots(figsize=(7, 5), subplot_kw={"projection": ccrs.PlateCarree()}, layout="constrained")
ax.set_extent([110, 150, 20, 55.5], crs=ccrs.PlateCarree())
ax.coastlines(linewidth=0.6)
mesh = ax.contourf(psi_ea["lon"], psi_ea["lat"], psi_ea.values, transform=ccrs.PlateCarree(), levels=levels, cmap="RdBu_r", extend="both")

step = 2
ax.streamplot(lon_ea[::step], lat_ea[::step], u_rot_ea[::step, ::step], v_rot_ea[::step, ::step], transform=ccrs.PlateCarree(), density=1.6, linewidth=0.7, color="k")

cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)
cbar.set_label("Streamfunction ψ (m$^2$ s$^{-1}$)")
ax.set_title("Streamfunction & Streamlines [East Asia]")

OUT_EA.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_EA, dpi=150)
plt.close(fig)

print(OUT_GLOBAL)
print(OUT_EA)
