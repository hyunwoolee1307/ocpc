# ocpc
Monthly operational workflow at the Ocean Climate Prediction Center

## Preferred tooling

- Statistics/EDA: R
- Numerical computing (especially linear algebra): Julia
- Visualization: Python

See `docs/setup.md` for the full environment standard.

## Wind decomposition plotting

Use the plotting helper to visualize Helmholtz decomposition from a NetCDF file:

```bash
python ocpc/plot_wind_decomposition.py \
  --input path/to/winds.nc \
  --output wind_decomposition.png \
  --time-index 0 \
  --level-index 0
```
