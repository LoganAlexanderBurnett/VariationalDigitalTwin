# Variational Digital Twin

This repository contains benchmark workflows for variational digital twin modeling across three application domains:

- **NASA Battery**: battery aging and voltage prediction
- **HTTF**: thermal forecasting
- **PSML**: renewable power forecasting

## Paper

Published article DOI: https://doi.org/10.1016/j.egyai.2026.100756

![Variational Digital Twin loop](VDT-loop.jpg)

## Environment setup

Create a reproducible conda environment from the provided specification file:

```bash
conda env create -f environment.yml
conda activate variational-digital-twin
```

Optional notebook automation dependency:

```bash
pip install papermill
```

## Reproducing paper artifacts

### 1) Run model training/evaluation scripts
Run the experiment scripts relevant to your study (NASA_Battery, HTTF, and/or PSML) to produce model outputs and figures.

### 2) Collect outputs into `paper_results/`
From the repository root, run:

```bash
python scripts/generate_paper_results.py --clean
```

The collection script performs the following steps:

1. Executes the configured plotting scripts:
   - `NASA_Battery/plot_static_vs_rolling.py`
   - `HTTF/static_training/plot_model_comparisons.py`
2. Collects selected figures and result artifacts from `NASA_Battery/`, `HTTF/`, and `PSML/`.
3. Copies collected files into `paper_results/` using source-relative paths.
4. Writes `paper_results/MANIFEST.md` with an index of included artifacts.

If figures are already generated and you only want to re-collect files:

```bash
python scripts/generate_paper_results.py --skip-plots --clean
```

## Reproducibility note

Some training pipelines are stochastic. Exact numerical values may vary between runs, while overall performance trends should remain similar.
