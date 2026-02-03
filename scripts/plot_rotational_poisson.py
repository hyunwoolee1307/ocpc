import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from pathlib import Path
import argparse
import os
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
import sys

from streamfunction import compute_streamfunction

mpl.rcParams["axes.unicode_minus"] = False


def configure_korean_font():
    target_names = [
        "Noto Sans CJK KR",
        "Noto Sans CJK",
        "Noto Sans CJK JP",
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
    ]
    font_paths = font_manager.findSystemFonts(fontext="ttf")
    font_paths += font_manager.findSystemFonts(fontext="otf")

    for font_path in font_paths:
        try:
            font_manager.fontManager.addfont(font_path)
            props = FontProperties(fname=font_path)
            name = props.get_name()
            if name in target_names:
                mpl.rcParams["font.family"] = name
                mpl.rcParams["font.sans-serif"] = [name]
                return props
        except Exception:
            continue

    print("Warning: Noto Sans CJK not found in system fonts.", file=sys.stderr)
    return None


TITLE_FONT_PROPERTIES = configure_korean_font()


def normalize_year_month(year, month):
    year_str = str(year).strip()
    month_str = str(month).strip().zfill(2)
    if len(year_str) != 4 or not year_str.isdigit():
        raise ValueError(f"Invalid year: {year}")
    if not month_str.isdigit() or int(month_str) < 1 or int(month_str) > 12:
        raise ValueError(f"Invalid month: {month}")
    return year_str, month_str


def compute_preferred_suffix(year, month):
    year_int = int(year)
    month_int = int(month)

    def advance_month(y, m, offset):
        m_new = m + offset
        y_new = y + (m_new - 1) // 12
        m_new = ((m_new - 1) % 12) + 1
        return y_new, m_new

    y1, m1 = advance_month(year_int, month_int, 1)
    _, m2 = advance_month(year_int, month_int, 2)
    _, m3 = advance_month(year_int, month_int, 3)
    return f"{y1:04d}{m1:02d}{m2:02d}{m3:02d}"


def find_forecast_file(variable, forecast_dir, year, month, preferred_suffix=None):
    forecast_dir = Path(forecast_dir)
    candidates = sorted(forecast_dir.glob(f"{variable}.{year}{month}2100.*.nc"))
    if not candidates:
        raise FileNotFoundError(f"No {variable} files found in {forecast_dir} for {year}{month}")

    if preferred_suffix:
        preferred = forecast_dir / f"{variable}.{year}{month}2100.{preferred_suffix}.nc"
        if preferred.exists():
            return preferred

    return candidates[-1]


def plot_rotational_poisson(year, month, u_path=None, v_path=None, output_dir=None, data_root=None, forecast_dir=None, preferred_suffix=None):
    year, month = normalize_year_month(year, month)
    month_season = {
        "01": "FMA",
        "02": "MAM",
        "03": "AMJ",
        "04": "MJJ",
        "05": "JJA",
        "06": "JAS",
        "07": "ASO",
        "08": "SON",
        "09": "OND",
        "10": "NDJ",
        "11": "DJF",
        "12": "JFM"
    }
    season = month_season.get(month, "FMA")
    preferred_suffix = preferred_suffix or compute_preferred_suffix(year, month)
    
    if data_root is None:
        data_root = os.environ.get("OCPC_DATA_ROOT", "/mnt/d/Data/processed/CFSv2")
    if forecast_dir is None:
        forecast_dir = Path(data_root) / f"{year}{month}_{season}" / "forecast"
    if output_dir is None:
        output_dir = os.environ.get("OCPC_OUTPUT_DIR", str(Path(__file__).resolve().parents[1] / "results" / "figures"))

    if u_path is None:
        u_path = find_forecast_file("ugrd", forecast_dir, year, month, preferred_suffix)
    if v_path is None:
        v_path = find_forecast_file("vgrd", forecast_dir, year, month, preferred_suffix)

    OUT_GLOBAL = Path(f"{output_dir}/stream_ccrs_global_{year}_{season}_forecast.png")
    OUT_EA = Path(f"{output_dir}/stream_ccrs_SAK_{year}_{season}_forecast.png")

    R = 6.371e6

    u = xr.open_dataset(u_path)["ugrd"]
    v = xr.open_dataset(v_path)["vgrd"]

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
    ax.set_xticks([80, 160, 240, 320], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -45, 0, 45, 90], crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(top=False, right=False, labeltop=False, labelright=False)

    cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)
    ax.set_title(
        f"전지구 streamfunction 및 streamline \n(forecast, {season}, m$^2$ s$^{{-1}}$)",
        fontproperties=TITLE_FONT_PROPERTIES,
    )

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
    ea_vmax = 4.0e6
    ea_step = 0.5e6
    ea_levels = np.arange(-ea_vmax, ea_vmax + ea_step, ea_step)
    mesh = ax.contourf(psi_ea["lon"], psi_ea["lat"], psi_ea.values, transform=ccrs.PlateCarree(), levels=ea_levels, cmap="RdBu_r", extend="both")

    step = 2
    ax.streamplot(lon_ea[::step], lat_ea[::step], u_rot_ea[::step, ::step], v_rot_ea[::step, ::step], transform=ccrs.PlateCarree(), density=1.6, linewidth=0.7, color="k")
    ax.set_xticks([110, 120, 130, 140, 150], crs=ccrs.PlateCarree())
    ax.set_yticks([20, 30, 40, 50], crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(top=False, right=False, labeltop=False, labelright=False)

    cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)
    ax.set_title(
        f"동아시아해역 streamfunction 및 streamline \n(forecast, {season}, m$^2$ s$^{{-1}}$)",
        fontproperties=TITLE_FONT_PROPERTIES,
    )

    OUT_EA.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_EA, dpi=150)
    plt.close(fig)

    print(OUT_GLOBAL)
    print(OUT_EA)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot rotational Poisson streamfunction")
    parser.add_argument("--year", help="Year (YYYY)")
    parser.add_argument("--month", help="Month (MM)")
    parser.add_argument('--u_path', help='Path to ugrd file')
    parser.add_argument('--v_path', help='Path to vgrd file')
    parser.add_argument('--output_dir', help='Output directory for figures')
    parser.add_argument('--data_root', help='Root directory for processed CFSv2 data')
    parser.add_argument('--forecast_dir', help='Override forecast directory')
    parser.add_argument('--preferred_suffix', help='Preferred forecast suffix (e.g., 2025091011)')

    args = parser.parse_args()
    year = args.year or input("Enter year: ").strip()
    month = args.month or input("Enter month: ").strip()

    plot_rotational_poisson(
        year,
        month,
        u_path=args.u_path,
        v_path=args.v_path,
        output_dir=args.output_dir,
        data_root=args.data_root,
        forecast_dir=args.forecast_dir,
        preferred_suffix=args.preferred_suffix,
    )
