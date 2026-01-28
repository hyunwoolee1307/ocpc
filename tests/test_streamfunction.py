import numpy as np
import xarray as xr

from ocpc.streamfunction import compute_streamfunction, load_streamfunction


def _build_dataset(tmp_path, u_value=0.0, v_value=0.0):
    sfc = np.array([1000.0, 850.0])
    lat = np.array([-90.0, 0.0, 90.0])
    lon = np.array([0.0, 90.0, 180.0, 270.0])
    shape = (sfc.size, lat.size, lon.size)
    ugrd = np.full(shape, u_value)
    vgrd = np.full(shape, v_value)
    dataset = xr.Dataset(
        {
            "ugrd": (("sfc", "lat", "lon"), ugrd),
            "vgrd": (("sfc", "lat", "lon"), vgrd),
        },
        coords={"sfc": sfc, "lat": lat, "lon": lon},
    )
    path = tmp_path / "winds.nc"
    dataset.to_netcdf(path)
    return path, dataset


def test_zero_winds_constant_streamfunction(tmp_path):
    path, dataset = _build_dataset(tmp_path, u_value=0.0, v_value=0.0)
    result = load_streamfunction(path)
    assert result.shape == dataset["ugrd"].shape
    assert np.allclose(result.values, 0.0)


def test_nonzero_v_winds_decrease_along_lon(tmp_path):
    path, dataset = _build_dataset(tmp_path, u_value=0.0, v_value=5.0)
    result = load_streamfunction(path)
    assert result.shape == dataset["ugrd"].shape
    assert np.all(result.values[:, :, -1] <= result.values[:, :, 0])


def test_compute_streamfunction_direct_array():
    lat = np.array([-90.0, 0.0, 90.0])
    lon = np.array([0.0, 180.0, 360.0])
    u = np.zeros((1, lat.size, lon.size))
    v = np.zeros((1, lat.size, lon.size))
    stream = compute_streamfunction(u, v, lat, lon)
    assert stream.shape == u.shape
    assert np.allclose(stream, 0.0)
