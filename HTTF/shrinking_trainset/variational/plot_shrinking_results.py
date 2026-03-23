import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os
import json
import pandas as pd

def collect_metrics_from(model_dir):
    """
    Given a directory like 'VGRU_finalrun' or 'VLSTM_finalrun',
    finds all 'shrink_train_*' subdirectories, loads their
    performance_metrics.json, and returns a sorted DataFrame.
    """
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Directory not found: {model_dir}")

    # 1) Discover & sort shrink_train directories inside model_dir
    shrink_dirs = sorted(
        [
            d for d in os.listdir(model_dir)
            if os.path.isdir(os.path.join(model_dir, d)) and d.startswith('shrink_train_')
        ],
        key=lambda x: int(x.split('_')[2])
    )

    # 2) Collect metrics into a list of dicts
    records = []
    for d in shrink_dirs:
        full_path = os.path.join(model_dir, d)
        metrics_path = os.path.join(full_path, 'performance_metrics.json')
        if not os.path.exists(metrics_path):
            print(f"Warning: no metrics in {full_path}, skipping")
            continue

        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        # parse out N_removed from the directory name
        N = int(d.split('_')[2])
        rec = {'N_removed': N}

        # flatten output_1 and output_2 metrics
        for output in ['output_1', 'output_2']:
            for m in ['r2', 'mae', 'mape', 'rmse', 'rmspe']:
                rec[f"{output}_{m}"] = metrics.get(output, {}).get(m, None)

        records.append(rec)

    # 3) Build and sort DataFrame
    df = pd.DataFrame(records)
    df = df.sort_values('N_removed').reset_index(drop=True)
    # Add a column for N_sensors (reverse of N_removed)
    df['N_sensors'] = df['N_removed'].iloc[::-1].values
    return df

# Specify the two model directories
model_dirs = ['VGRU_finalrun', 'VLSTM_finalrun']

# Collect into a dict of DataFrames
dfs = {}
for mdl in model_dirs:
    try:
        dfs[mdl] = collect_metrics_from(mdl)
    except FileNotFoundError as e:
        print(e)
        dfs[mdl] = pd.DataFrame()  # empty DataFrame if directory not found

# Now dfs['VGRU_finalrun'] and dfs['VLSTM_finalrun'] are your two DataFrames:
df_vgru  = dfs['VGRU_finalrun']
df_vlstm = dfs['VLSTM_finalrun']

# (Optional) Display them
print("VGRU metrics:")
print(df_vgru.head(), "\n")
print("VLSTM metrics:")
print(df_vlstm.head())

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# --- Helper to compute average CI width per output for each shrink_train dir ---
def compute_avg_ci_widths_per_output(model_dir):
    """
    For each 'shrink_train_{N}' under model_dir, load 'Ypred.npy' (shape (T, 2, 3)).
    Compute width = (upper_ci − lower_ci) separately for output 1 and output 2:
      widths_TS = Ypred[:, 0, 2] − Ypred[:, 0, 1]
      widths_TF = Ypred[:, 1, 2] − Ypred[:, 1, 1]
    Then average over all T to get two scalars per N. Returns:
      Ns (sorted list), avg_TS, avg_TF  (each a list aligned with Ns).
    """
    shrink_dirs = sorted(
        [
            d for d in os.listdir(model_dir)
            if os.path.isdir(os.path.join(model_dir, d)) and d.startswith('shrink_train_')
        ],
        key=lambda x: int(x.split('_')[2])
    )
    Ns = []
    avg_TS = []
    avg_TF = []
    for d in shrink_dirs:
        N = int(d.split('_')[2])
        pfile = os.path.join(model_dir, d, 'Ypred.npy')
        if not os.path.exists(pfile):
            print(f"Warning: Ypred.npy not found in {os.path.join(model_dir, d)}; skipping")
            continue

        Ypred = np.load(pfile)  # shape (T, 2, 3)
        # width for TS (output 1) and TF (output 2)
        width_TS = Ypred[:, 0, 2] - Ypred[:, 0, 1]  # shape (T,)
        width_TF = Ypred[:, 1, 2] - Ypred[:, 1, 1]  # shape (T,)
        avg_TS.append(width_TS.mean())
        avg_TF.append(width_TF.mean())
        Ns.append(N)

    return Ns, avg_TS, avg_TF

# --- Assume df_vgru and df_vlstm already exist as before ---

# 1) Compute average CI widths per output for both models
vgru_Ns,  vgru_avg_TS,  vgru_avg_TF  = compute_avg_ci_widths_per_output('VGRU_finalrun')
vlstm_Ns, vlstm_avg_TS, vlstm_avg_TF = compute_avg_ci_widths_per_output('VLSTM_finalrun')

# 2) Define column names and y-labels for each metric/output
metrics = {
    'R²':   {'col1': 'output_1_r2',  'col2': 'output_2_r2',  'ylabel': r'$R^2$'},
    'MAE':  {'col1': 'output_1_mae', 'col2': 'output_2_mae', 'ylabel': 'MAE (°C)'},
    'RMSE': {'col1': 'output_1_rmse','col2': 'output_2_rmse','ylabel': 'RMSE (°C)'}
}

# 3) Define a distinct color mapping for (model, output)
line_colors = {
    ('vGRU', 'TS'): 'dodgerblue',
    ('vGRU', 'TF'): 'mediumseagreen',
    ('vLSTM', 'TS'): 'orangered',
    ('vLSTM', 'TF'): 'gold'
}

# 4) Define markers by model
model_markers = {
    'vGRU':  'o',
    'vLSTM': 'D'
}

# 5) Create figure and 2×2 GridSpec
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(
    nrows=2,
    ncols=2,
    height_ratios=[1, 1],
    hspace=0.2,
    wspace=0.2
)

# 6) Top-left: R² plot (in gs[0,0])
ax_r2 = fig.add_subplot(gs[0, 0])
for model_label, df in [('vGRU', df_vgru), ('vLSTM', df_vlstm)]:
    for output_idx, output_name in [(1, 'TS'), (2, 'TF')]:
        colname = metrics['R²'][f'col{output_idx}']
        ax_r2.plot(
            df['N_removed'],
            df[colname],
            color=line_colors[(model_label, output_name)],
            marker=model_markers[model_label],
            linestyle='-',
            alpha=0.5,
            label=f"{model_label} {output_name}"
        )

ax_r2.set_xlabel("Number of Sensors Removed", fontsize=12)
ax_r2.set_ylabel(metrics['R²']['ylabel'], fontsize=12)
ax_r2.set_ylim(0.5, 1.03)
ax_r2.legend(loc='best', fontsize=10)
ax_r2.grid(alpha=0.3)

# 7) Top-right: Avg. CI width per output (in gs[0,1])
ax_ci = fig.add_subplot(gs[0, 1])
# vGRU TS & TF
ax_ci.plot(
    vgru_Ns,
    vgru_avg_TS,
    color=line_colors[('vGRU', 'TS')],
    marker=model_markers['vGRU'],
    linestyle='-',
    alpha=0.5,
    label='vGRU TS'
)
ax_ci.plot(
    vgru_Ns,
    vgru_avg_TF,
    color=line_colors[('vGRU', 'TF')],
    marker=model_markers['vGRU'],
    linestyle='-',
    alpha=0.5,
    label='vGRU TF'
)
# vLSTM TS & TF
ax_ci.plot(
    vlstm_Ns,
    vlstm_avg_TS,
    color=line_colors[('vLSTM', 'TS')],
    marker=model_markers['vLSTM'],
    linestyle='-',
    alpha=0.5,
    label='vLSTM TS'
)
ax_ci.plot(
    vlstm_Ns,
    vlstm_avg_TF,
    color=line_colors[('vLSTM', 'TF')],
    marker=model_markers['vLSTM'],
    linestyle='-',
    alpha=0.5,
    label='vLSTM TF'
)

ax_ci.set_xlabel("Number of Sensors Removed", fontsize=12)
ax_ci.set_ylabel("Avg. CI Width (°C)", fontsize=12)
ax_ci.legend(loc='best', fontsize=10)
ax_ci.grid(alpha=0.3)

# 8) Bottom-left: MAE plot (in gs[1,0])
ax_mae = fig.add_subplot(gs[1, 0])
for model_label, df in [('vGRU', df_vgru), ('vLSTM', df_vlstm)]:
    for output_idx, output_name in [(1, 'TS'), (2, 'TF')]:
        colname = metrics['MAE'][f'col{output_idx}']
        ax_mae.plot(
            df['N_removed'],
            df[colname],
            color=line_colors[(model_label, output_name)],
            marker=model_markers[model_label],
            linestyle='-',
            alpha=0.5,
            label=f"{model_label} {output_name}"
        )

ax_mae.set_xlabel("Number of Sensors Removed", fontsize=12)
ax_mae.set_ylabel(metrics['MAE']['ylabel'], fontsize=12)
ax_mae.legend(loc='best', fontsize=10)
ax_mae.grid(alpha=0.3)

# 9) Bottom-right: RMSE plot (in gs[1,1])
ax_rmse = fig.add_subplot(gs[1, 1])
for model_label, df in [('vGRU', df_vgru), ('vLSTM', df_vlstm)]:
    for output_idx, output_name in [(1, 'TS'), (2, 'TF')]:
        colname = metrics['RMSE'][f'col{output_idx}']
        ax_rmse.plot(
            df['N_removed'],
            df[colname],
            color=line_colors[(model_label, output_name)],
            marker=model_markers[model_label],
            linestyle='-',
            alpha=0.5,
            label=f"{model_label} {output_name}"
        )

ax_rmse.set_xlabel("Number of Sensors Removed", fontsize=12)
ax_rmse.set_ylabel(metrics['RMSE']['ylabel'], fontsize=12)
ax_rmse.legend(loc='best', fontsize=10)
ax_rmse.grid(alpha=0.3)

plt.tight_layout()
fig.savefig('HTTF_shrink_metrics.png', dpi=300, bbox_inches='tight')
plt.show()
