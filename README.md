# Variational Digital Twin

Reinforcement-learning-free benchmark repository for variational digital twin modeling across three case studies:

- **NASA Battery aging prediction**
- **HTTF thermal forecasting**
- **PSML renewable power forecasting**

This repository now includes an automated workflow to generate a consolidated `paper_results/` folder containing the main paper figures and key result artifacts.

## Environment installation

```bash
# 1) Create and activate the conda environment with all required dependencies
conda env create -f environment.yml
conda activate variational-digital-twin

# 2) (Optional) install papermill for notebook automation
pip install papermill
```

## How to generate the paper results

### Step 1: Run (or re-run) experiment/plot scripts
You can run your normal training and plotting scripts in each module as needed (e.g., NASA battery static/rolling outputs, HTTF comparisons, PSML outputs).

### Step 2: Build the consolidated `paper_results/` directory
From the repository root:

```bash
python scripts/generate_paper_results.py --clean
```

This command:

1. Runs the plot scripts that generate key comparison figures:
   - `NASA_Battery/plot_static_vs_rolling.py`
   - `HTTF/static_training/plot_model_comparisons.py`
2. Collects paper-relevant artifacts from `NASA_Battery/`, `HTTF/`, and `PSML/`.
3. Writes all copied files to `paper_results/` while preserving source-relative paths.
4. Generates `paper_results/MANIFEST.md` listing all included artifacts.

## Notes

- If you already generated figures and only want to re-collect files, use:

  ```bash
  python scripts/generate_paper_results.py --skip-plots --clean
  ```

- Some training procedures are stochastic. Re-running model training can produce slightly different metrics, but trends should remain consistent.
