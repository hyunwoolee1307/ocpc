"""Plot Helmholtz wind decomposition from a NetCDF file.

Example
-------
python ocpc/plot_wind_decomposition.py \
    --input data/winds.nc \
    --output wind_decomposition.png \
    --time-index 0 \
    --level-index 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from ocpc.streamfunction import compute_wind_decomposition

try:  # cartopy is optional
    import cartopy.crs as ccrs
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    ccrs = None

import matplotlib.pyplot as plt


U_NAMES = ("ugrd", "u")
V_NAMES = ("vgrd", "v")
LAT_NAMES = ("lat", "latitude")
LON_NAMES = ("lon", "longitude")
LEVEL_NAMES = ("level", "lev", "plev", "pressure", "isobaricInhPa", "height")


def _pick_name(names: tuple[str, ...], candidates: tuple[str, ...]) -> str | None:
    for name in names:
        if name in candidates:
            return name
    return None


def _select_by_index(da: xr.DataArray, dim: str, index: int | None) -> xr.DataArray:
    if index is None:
        return da
    return da.isel({dim: index})


def _select_by_value(da: xr.DataArray, dim: str, value: str | float | None) -> xr.DataArray:
    if value is None:
        return da
    try:
        return da.sel({dim: value})
    except (KeyError, ValueError):
        return da.sel({dim: value}, method="nearest")


def _infer_level_dim(da: xr.DataArray) -> str | None:
    return _pick_name(LEVEL_NAMES, tuple(da.dims))


def _prepare_winds(
    dataset: xr.Dataset,
    time_index: int | None,
    time_value: str | None,
    level_index: int | None,
    level_value: float | None,
) -> tuple[xr.DataArray, xr.DataArray, str, str]:
    u_name = _pick_name(U_NAMES, tuple(dataset.data_vars))
    v_name = _pick_name(V_NAMES, tuple(dataset.data_vars))
    if u_name is None or v_name is None:
        raise ValueError("Dataset must contain u/v or ugrd/vgrd variables.")

    u_da = dataset[u_name]
    v_da = dataset[v_name]

    time_dim = "time" if "time" in u_da.dims else None
    if time_dim:
        u_da = _select_by_index(u_da, time_dim, time_index)
        v_da = _select_by_index(v_da, time_dim, time_index)
        u_da = _select_by_value(u_da, time_dim, time_value)
        v_da = _select_by_value(v_da, time_dim, time_value)

    level_dim = _infer_level_dim(u_da)
    if level_dim:
        u_da = _select_by_index(u_da, level_dim, level_index)
        v_da = _select_by_index(v_da, level_dim, level_index)
        u_da = _select_by_value(u_da, level_dim, level_value)
        v_da = _select_by_value(v_da, level_dim, level_value)

    lat_name = _pick_name(LAT_NAMES, tuple(u_da.coords))
    lon_name = _pick_name(LON_NAMES, tuple(u_da.coords))
    if lat_name is None or lon_name is None:
        raise ValueError("Latitude/longitude coordinates not found (expected lat/lon).")

    return u_da, v_da, lat_name, lon_name


def _build_mesh(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon2d, lat2d = np.meshgrid(lon, lat)
    return lon2d, lat2d


def _stride_for_quiver(nlat: int, nlon: int) -> int:
    return max(1, int(max(nlat / 25, nlon / 35)))


def plot_wind_decomposition(
    u_da: xr.DataArray,
    v_da: xr.DataArray,
    lat_name: str,
    lon_name: str,
    output: Path,
    method: str,
) -> None:
    lat = u_da[lat_name].values
    lon = u_da[lon_name].values
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("Latitude and longitude must be 1D for plotting.")

    psi, chi, u_rot, v_rot, u_div, v_div = compute_wind_decomposition(
        u_da.values, v_da.values, lat, lon, method=method
    )

    lon2d, lat2d = _build_mesh(lat, lon)
    speed = np.sqrt(u_da.values**2 + v_da.values**2)
    use_cartopy = ccrs is not None
    projection = ccrs.PlateCarree() if use_cartopy else None
    subplot_kwargs = {"projection": projection} if use_cartopy else {}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True, subplot_kw=subplot_kwargs)
    panel_specs = [
        ("Total wind speed", speed, u_da.values, v_da.values),
        ("Streamfunction ψ (rotational)", psi, u_rot, v_rot),
        ("Velocity potential χ (divergent)", chi, u_div, v_div),
    ]

    stride = _stride_for_quiver(lat.size, lon.size)

    for ax, (title, scalar, u_comp, v_comp) in zip(axes, panel_specs, strict=True):
        if use_cartopy:
            ax.coastlines(linewidth=0.6)
            ax.set_global()

        mesh = ax.pcolormesh(
            lon2d,
            lat2d,
            scalar,
            shading="auto",
            cmap="viridis",
            transform=ccrs.PlateCarree() if use_cartopy else None,
        )
        ax.set_title(title)
        ax.quiver(
            lon2d[::stride, ::stride],
            lat2d[::stride, ::stride],
            u_comp[::stride, ::stride],
            v_comp[::stride, ::stride],
            transform=ccrs.PlateCarree() if use_cartopy else None,
            scale=500,
            width=0.002,
        )
        fig.colorbar(mesh, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot wind decomposition from a NetCDF file.")
    parser.add_argument("--input", required=True, help="Path to input NetCDF with u/v or ugrd/vgrd.")
    parser.add_argument("--output", required=True, help="Output image path (png/pdf).")
    parser.add_argument("--time-index", type=int, default=None, help="Time index to select.")
    parser.add_argument("--time", dest="time_value", default=None, help="Time coordinate value to select.")
    parser.add_argument("--level-index", type=int, default=None, help="Vertical level index to select.")
    parser.add_argument("--level", dest="level_value", type=float, default=None, help="Vertical level value to select.")
    parser.add_argument(
        "--method",
        choices=("scipy", "poisson"),
        default="scipy",
        help="Poisson solver to use for decomposition.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset = xr.open_dataset(args.input)
    u_da, v_da, lat_name, lon_name = _prepare_winds(
        dataset,
        time_index=args.time_index,
        time_value=args.time_value,
        level_index=args.level_index,
        level_value=args.level_value,
    )

    plot_wind_decomposition(
        u_da=u_da,
        v_da=v_da,
        lat_name=lat_name,
        lon_name=lon_name,
        output=Path(args.output),
        method=args.method,
    )


if __name__ == "__main__":
    main()
