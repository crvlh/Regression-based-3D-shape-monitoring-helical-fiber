# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 14:13:21 2025

@author: vinic
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd

colors = ['ciano', 'laranja', 'verde', 'amarelo', 'rosa']

# Range, point, axis
max_shift = {
    'ciano':   {'x': 0,   'y': 0,   'z': 0},
    'laranja': {'x': 0.2, 'y': 0.2, 'z': 0},
    'verde':   {'x': 2.2, 'y': 2.2, 'z': 0},
    'amarelo': {'x': 3.8, 'y': 3.8, 'z': 0.3},
    'rosa':    {'x': 6.2, 'y': 6.2, 'z': 0.4},
}

metrics = {}

for i, color in enumerate(colors):
    x_idx = i
    y_idx = i + 5
    z_idx = i + 10

    real_x = configs_test[:, x_idx]
    pred_x = pred_test[:, x_idx]
    real_y = configs_test[:, y_idx]
    pred_y = pred_test[:, y_idx]
    real_z = configs_test[:, z_idx]
    pred_z = pred_test[:, z_idx]

    # MAE amostral
    mae_x_array = np.abs(real_x - pred_x)
    mae_y_array = np.abs(real_y - pred_y)
    mae_z_array = np.abs(real_z - pred_z)

    mae_x = np.mean(mae_x_array)
    mae_y = np.mean(mae_y_array)
    mae_z = np.mean(mae_z_array)

    std_x = np.std(mae_x_array)
    std_y = np.std(mae_y_array)
    std_z = np.std(mae_z_array)

    # Relative error (%)
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

    metrics[color] = {
        'mae_x': mae_x, 'std_x': std_x, 'rel_mae_x(%)': rel_mae_x, 'std_rel_x(%)': std_rel_x,
        'mae_y': mae_y, 'std_y': std_y, 'rel_mae_y(%)': rel_mae_y, 'std_rel_y(%)': std_rel_y,
        'mae_z': mae_z, 'std_z': std_z, 'rel_mae_z(%)': rel_mae_z, 'std_rel_z(%)': std_rel_z,
        'r2_x': r2_score(real_x, pred_x),
        'r2_y': r2_score(real_y, pred_y),
        'r2_z': r2_score(real_z, pred_z),
    }

# Table MAE ± STD (%)
tabela_mae_std = pd.DataFrame({
    'mae_x (±)': [f"{metrics[c]['mae_x']:.5f} ± {metrics[c]['std_x']:.2f}" for c in colors],
    'rel_mae_x(%) (±)': [f"{round(metrics[c]['rel_mae_x(%)'])} ± {round(metrics[c]['std_rel_x(%)'])}" for c in colors],
    'mae_y (±)': [f"{metrics[c]['mae_y']:.5f} ± {metrics[c]['std_y']:.2f}" for c in colors],
    'rel_mae_y(%) (±)': [f"{round(metrics[c]['rel_mae_y(%)'])} ± {round(metrics[c]['std_rel_y(%)'])}" for c in colors],
    'mae_z (±)': [f"{metrics[c]['mae_z']:.5f} ± {metrics[c]['std_z']:.2f}" for c in colors],
    'rel_mae_z(%) (±)': [f"{round(metrics[c]['rel_mae_z(%)'])} ± {round(metrics[c]['std_rel_z(%)'])}" for c in colors]
}, index=colors)

# Euclidean MAE
erros_euclidianos_por_cor = {}

for i, color in enumerate(colors):
    x_idx = i
    y_idx = i + 5
    z_idx = i + 10

    diff_x = configs_test[:, x_idx] - pred_test[:, x_idx]
    diff_y = configs_test[:, y_idx] - pred_test[:, y_idx]
    diff_z = configs_test[:, z_idx] - pred_test[:, z_idx]

    erros = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)

    erros_euclidianos_por_cor[color] = {
        'vetor_erros': erros,
        'erro_medio': np.mean(erros)
    }

for cor, dados in erros_euclidianos_por_cor.items():
    print(f"Erro euclidiano médio para {cor}: {dados['erro_medio']:.4f} cm")
