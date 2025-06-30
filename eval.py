# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 14:13:21 2025

@author: vinic
"""

import numpy as np
from sklearn.metrics import mean_absolute_error
import pandas as pd

# Monitored point labels
colors = ['ciano', 'laranja', 'verde', 'amarelo', 'rosa']

# Maximum displacement values for each point and axis
max_shift = {
    'ciano':   {'x': ,   'y':,   'z': },
    'laranja': {'x': , 'y': , 'z': },
    'verde':   {'x': , 'y': , 'z': },
    'amarelo': {'x': , 'y': , 'z': },
    'rosa':    {'x': , 'y': , 'z': },
}

metrics = {}

# Loop through each color to compute errors
for i, color in enumerate(colors):
    x_idx = i
    y_idx = i + 5
    z_idx = i + 10

    # Extract real and predicted values
    real_x = configs_test[:, x_idx]
    pred_x = pred_test[:, x_idx]
    real_y = configs_test[:, y_idx]
    pred_y = pred_test[:, y_idx]
    real_z = configs_test[:, z_idx]
    pred_z = pred_test[:, z_idx]

    # Compute absolute errors
    mae_x_array = np.abs(real_x - pred_x)
    mae_y_array = np.abs(real_y - pred_y)
    mae_z_array = np.abs(real_z - pred_z)

    # Mean absolute error (MAE)
    mae_x = np.mean(mae_x_array)
    mae_y = np.mean(mae_y_array)
    mae_z = np.mean(mae_z_array)

    # Standard deviation of errors
    std_x = np.std(mae_x_array)
    std_y = np.std(mae_y_array)
    std_z = np.std(mae_z_array)

    # Relative MAE (%), if applicable
    if max_shift[color]['x'] > 0:
        rel_mae_x_array = mae_x_array / max_shift[color]['x'] * 100
        rel_mae_x = np.mean(rel_mae_x_array)
        std_rel_x = np.std(rel_mae_x_array)
    else:
        rel_mae_x = 0
        std_rel_x = 0

    if max_shift[color]['y'] > 0:
        rel_mae_y_array = mae_y_array / max_shift[color]['y'] * 100
        rel_mae_y = np.mean(rel_mae_y_array)
        std_rel_y = np.std(rel_mae_y_array)
    else:
        rel_mae_y = 0
        std_rel_y = 0

    if max_shift[color]['z'] > 0:
        rel_mae_z_array = mae_z_array / max_shift[color]['z'] * 100
        rel_mae_z = np.mean(rel_mae_z_array)
        std_rel_z = np.std(rel_mae_z_array)
    else:
        rel_mae_z = 0
        std_rel_z = 0

    # Store all metrics
    metrics[color] = {
        'mae_x': mae_x, 'std_x': std_x, 'rel_mae_x(%)': rel_mae_x, 'std_rel_x(%)': std_rel_x,
        'mae_y': mae_y, 'std_y': std_y, 'rel_mae_y(%)': rel_mae_y, 'std_rel_y(%)': std_rel_y,
        'mae_z': mae_z, 'std_z': std_z, 'rel_mae_z(%)': rel_mae_z, 'std_rel_z(%)': std_rel_z,
    }

# Build summary table for MAE and relative MAE
tabela_mae_std = pd.DataFrame({
    'mae_x (±)': [f"{metrics[c]['mae_x']:.1f} ± {metrics[c]['std_x']:.1f}" for c in colors],
    'rel_mae_x(%) (±)': [f"{round(metrics[c]['rel_mae_x(%)'])} ± {round(metrics[c]['std_rel_x(%)'])}" for c in colors],
    'mae_y (±)': [f"{metrics[c]['mae_y']:.1f} ± {metrics[c]['std_y']:.1f}" for c in colors],
    'rel_mae_y(%) (±)': [f"{round(metrics[c]['rel_mae_y(%)'])} ± {round(metrics[c]['std_rel_y(%)'])}" for c in colors],
    'mae_z (±)': [f"{metrics[c]['mae_z']:.1f} ± {metrics[c]['std_z']:.1f}" for c in colors],
    'rel_mae_z(%) (±)': [f"{round(metrics[c]['rel_mae_z(%)'])} ± {round(metrics[c]['std_rel_z(%)'])}" for c in colors]
}, index=colors)


