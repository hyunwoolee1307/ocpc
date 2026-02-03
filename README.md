# ocpc
Monthly operational workflow at the Ocean Climate Prediction Center.

**Preferred Tooling**
- Statistics/EDA: R
- Numerical computing (especially linear algebra): Julia
- Visualization: Python

See `docs/setup.md` for the full environment standard.

**Workspace Layout**
- `scripts/`: plotting and analysis scripts
- `results/`: generated figures and outputs
- `docs/`: environment and workflow documentation

**Data Layout**
The rotational Poisson plotting script expects CFSv2 wind data under a forecast directory:

`{OCPC_DATA_ROOT}/{YYYY}{MM}_{SEASON}/forecast/ugrd.{YYYY}{MM}2100.{SUFFIX}.nc`

Defaults:
- `OCPC_DATA_ROOT=/mnt/d/Data/processed/CFSv2`
- `OCPC_OUTPUT_DIR=<repo>/results/figures`

The script computes a preferred suffix based on month (lead-1 to lead-3 months). If the preferred suffix file is missing, it automatically picks the latest matching file in the forecast directory.

**Rotational Poisson Plotting**
Run with explicit year/month to avoid interactive input:

```bash
conda run -n ocpc python scripts/plot_rotational_poisson.py --year 2025 --month 11
```

Override file selection when needed:

```bash
conda run -n ocpc python scripts/plot_rotational_poisson.py \
  --year 2025 --month 08 \
  --preferred_suffix 2025091011
```

```bash
conda run -n ocpc python scripts/plot_rotational_poisson.py \
  --year 2025 --month 09 \
  --preferred_suffix 2025101112
```

Outputs are written to `results/figures` by default:
- `stream_ccrs_global_{YYYY}_{SEASON}_forecast.png`
- `stream_ccrs_SAK_{YYYY}_{SEASON}_forecast.png`

**Font Configuration**
The plotting script forces `Noto Sans CJK` from system fonts for Korean titles. If the font is missing, it prints a warning and may render titles incorrectly. Install the font on the system or in the environment before running plots.

**Reproduction: 2025-08 to 2026-01**
```bash
conda run -n ocpc python scripts/plot_rotational_poisson.py --year 2025 --month 08 --preferred_suffix 2025091011
conda run -n ocpc python scripts/plot_rotational_poisson.py --year 2025 --month 09 --preferred_suffix 2025101112
conda run -n ocpc python scripts/plot_rotational_poisson.py --year 2025 --month 10
conda run -n ocpc python scripts/plot_rotational_poisson.py --year 2025 --month 11
conda run -n ocpc python scripts/plot_rotational_poisson.py --year 2025 --month 12
conda run -n ocpc python scripts/plot_rotational_poisson.py --year 2026 --month 01
```

**System Resource Check**
Always verify available resources before large runs:

```bash
neofetch --stdout
```

Example output from this workspace:
```
OS: Ubuntu 24.04.3 LTS on Windows 10 x86_64
Kernel: 6.6.87.2-microsoft-standard-WSL2
CPU: Intel i9-14900K (32) @ 3.187GHz
Memory: 3933MiB / 31974MiB
```

To avoid exceeding resources, cap numerical library threads before running Python:

```bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
```

If memory or CPU is constrained (for example, available memory < 16 GiB), reduce these to `4` or lower.

**LLM Agent Guide**
See `docs/llm_agent_guide.md`.
