# LLM Agent Guide

**Purpose**
This guide is for LLM-driven execution of the plotting workflow with consistent outputs.

**Checklist**
1. Read `scripts/plot_rotational_poisson.py` to confirm inputs, outputs, and naming rules.
2. Verify `OCPC_DATA_ROOT` and that `{YYYY}{MM}_{SEASON}/forecast` exists for the target month.
3. Run `neofetch --stdout` and cap thread counts before heavy runs:
   - `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`
4. Ensure the system has `Noto Sans CJK` installed. The script forces it to render Korean titles:
   - If missing, plots may render with fallback fonts or missing glyphs.
5. Prefer non-interactive runs with `--year` and `--month`.
6. If a preferred file is missing, list the forecast directory and pass `--preferred_suffix`.
7. Record output paths under `results/figures`.

**Naming Rules**
- Global: `stream_ccrs_global_{YYYY}_{SEASON}_forecast.png`
- East Asia: `stream_ccrs_SAK_{YYYY}_{SEASON}_forecast.png`

**Font Troubleshooting**
If Korean glyphs are missing or warnings appear:
- Install `Noto Sans CJK` at the system level, then rerun.
- The script also applies `fontproperties` to titles, so once the font exists, warnings should disappear.
