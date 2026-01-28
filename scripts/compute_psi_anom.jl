#!/usr/bin/env julia
using NCDatasets
using Statistics

const R = 6.371e6

u_path = "/mnt/d/Data/processed/CFSv2/202601_FMA/forecast/a.ugrd.2026012100.2026020304.nc"
v_path = "/mnt/d/Data/processed/CFSv2/202601_FMA/forecast/a.vgrd.2026012100.2026020304.nc"
out_path = "/home/hyunwoo/ocpc/data/psi_anom_a.2026012100.2026020304.nc"

function _load_var(ds::NCDataset, name::String)
    var = ds[name]
    dims = dimnames(var)
    data = Array(var)

    if "sfc" in dims
        sfc_idx = findfirst(==("sfc"), dims)
        data = selectdim(data, sfc_idx, 1)
        dims = filter(!=("sfc"), dims)
    end

    lat_idx = findfirst(==("lat"), dims)
    lon_idx = findfirst(==("lon"), dims)
    if lat_idx === nothing || lon_idx === nothing
        error("Expected lat/lon dimensions in $name")
    end

    perm = (lat_idx, lon_idx)
    data = permutedims(data, perm)
    return Float64.(coalesce.(data, NaN))
end

function load_uv(u_path::String, v_path::String)
    u_ds = NCDataset(u_path)
    v_ds = NCDataset(v_path)
    lat = Float64.(u_ds["lat"][:])
    lon = Float64.(u_ds["lon"][:])

    u = _load_var(u_ds, "ugrd")
    v = _load_var(v_ds, "vgrd")

    close(u_ds)
    close(v_ds)
    return u, v, lat, lon
end

function compute_psi(u, v, lat, lon)
    lat_rad = deg2rad.(lat)
    lon_rad = deg2rad.(lon)
    dphi = diff(lat_rad)
    dlon = diff(lon_rad)
    coslat = cos.(lat_rad)

    # Integrate along longitude: dpsi = -v * R * cos(phi) dλ
    dx = (R .* coslat) .* dlon'  # (lat, lon-1)
    incr_x = -v[:, 1:end-1] .* dx
    incr_x = replace(incr_x, NaN => 0.0)
    psi_x = zeros(size(v))
    psi_x[:, 2:end] = cumsum(incr_x, dims=2)

    # Integrate along latitude: dpsi = -u * R dφ
    dy = R .* dphi
    incr_y = -u[1:end-1, :] .* dy
    incr_y = replace(incr_y, NaN => 0.0)
    psi_y = zeros(size(u))
    psi_y[2:end, :] = cumsum(incr_y, dims=1)

    psi = 0.5 .* (psi_x .+ psi_y)

    mask = isnan.(u) .| isnan.(v)
    psi = ifelse.(mask, NaN, psi)
    return psi
end

u, v, lat, lon = load_uv(u_path, v_path)
psi = compute_psi(u, v, lat, lon)

ds_out = NCDataset(out_path, "c")
defDim(ds_out, "lat", length(lat))
defDim(ds_out, "lon", length(lon))
lat_var = defVar(ds_out, "lat", Float64, ("lat",))
lon_var = defVar(ds_out, "lon", Float64, ("lon",))
lat_var[:] = lat
lon_var[:] = lon
psi_var = defVar(ds_out, "psi", Float64, ("lat", "lon"))
psi_var[:, :] = psi
psi_var.attrib["long_name"] = "Stokes streamfunction (anomaly)"
psi_var.attrib["units"] = "m^2/s"
close(ds_out)

println(out_path)
