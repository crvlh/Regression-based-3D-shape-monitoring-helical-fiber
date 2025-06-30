# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 12:53:56 2025

@author: vinic
"""

import os
import glob
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error

# Set random seed for reproducibility
SEED = 0
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Input configuration
n_wavelength_effective = # Number of wavelengths used
skip_rows = # Number of header lines to skip
n_labels =  # Number of target labels (5 labels × 3 axes)
colors = ['ciano', 'laranja', 'verde', 'amarelo', 'rosa'] # Color names of target points

# Load target values
data_folder = "PATH_TO_FOLDER"
config_file = "TARGET_FILE.xlsx"
df = pd.read_excel(os.path.join(data_folder, config_file), header=None)
num_samples = df.shape[0]
configs = np.array(df.iloc[:, 1:n_labels + 1].values)

# Load spectra
data_folder_spectra = os.path.join(data_folder, "SPECTRA_FOLDER")
file_list = sorted(glob.glob(os.path.join(data_folder_spectra, "*.txt")), key=lambda x: int(os.path.basename(x).split(".")[0]))
data = [pd.read_csv(file, delimiter='\t', skiprows=skip_rows, nrows=n_wavelength_effective, usecols=range(1, 2)) for file in file_list]
data = np.reshape(np.array(data), (num_samples, n_wavelength_effective))

# Normalize inputs and outputs
data = MinMaxScaler().fit_transform(data)
configs = MinMaxScaler().fit_transform(configs)

# Apply PCA
pca = PCA(n_components=)
data = pca.fit_transform(data)
print(f"Variância explicada pelo PCA: {sum(pca.explained_variance_ratio_):.4f}")

# Grid search configuration
param_grid = {
    'kernel': ['rbf', 'poly'],
    'C': [],
    'epsilon': [],
    'degree': []  
}

n_splits = 
results = {}

best_params_x_list = []
best_params_y_list = []
best_params_z_list = []

# Loop over each color (point)
for i, color in enumerate(colors):
    mae_x_folds, mae_y_folds, mae_z_folds = [], [], []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    # Grid search for each axis
    grid_x = GridSearchCV(SVR(), param_grid, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_x.fit(data, configs[:, i])
    best_x = grid_x.best_params_

    grid_y = GridSearchCV(SVR(), param_grid, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_y.fit(data, configs[:, i + 5])
    best_y = grid_y.best_params_

    grid_z = GridSearchCV(SVR(), param_grid, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_z.fit(data, configs[:, i + 10])
    best_z = grid_z.best_params_

    best_params_x_list.append(best_x)
    best_params_y_list.append(best_y)
    best_params_z_list.append(best_z)

    print(f"\nMelhores parâmetros para {color.capitalize()} (x): {best_x}")
    print(f"Melhores parâmetros para {color.capitalize()} (y): {best_y}")
    print(f"Melhores parâmetros para {color.capitalize()} (z): {best_z}")

    # Train best models
    svr_x = SVR(**best_x)
    svr_y = SVR(**best_y)
    svr_z = SVR(**best_z)
    svr_x.fit(data, configs[:, i])
    svr_y.fit(data, configs[:, i + 5])
    svr_z.fit(data, configs[:, i + 10])

    # Evaluate using K-Fold CV
    for train_idx, val_idx in kf.split(data):
        X_train, X_val = data[train_idx], data[val_idx]
        yx_train, yx_val = configs[train_idx, i], configs[val_idx, i]
        yy_train, yy_val = configs[train_idx, i + 5], configs[val_idx, i + 5]
        yz_train, yz_val = configs[train_idx, i + 10], configs[val_idx, i + 10]

        pred_x = svr_x.predict(X_val)
        pred_y = svr_y.predict(X_val)
        pred_z = svr_z.predict(X_val)

        mae_x_folds.append(mean_absolute_error(yx_val, pred_x))
        mae_y_folds.append(mean_absolute_error(yy_val, pred_y))
        mae_z_folds.append(mean_absolute_error(yz_val, pred_z))

    # Store results
    results[color] = {
        'mae_x': mae_x_folds,
        'mae_y': mae_y_folds,
        'mae_z': mae_z_folds,
        'mean_mae_x': np.mean(mae_x_folds),
        'std_mae_x': np.std(mae_x_folds),
        'mean_mae_y': np.mean(mae_y_folds),
        'std_mae_y': np.std(mae_y_folds),
        'mean_mae_z': np.mean(mae_z_folds),
        'std_mae_z': np.std(mae_z_folds),
    }

    # Fold results for current color
    print(f"\n Resultados para {color.capitalize()} ---")
    for fold, (mx, my, mz) in enumerate(zip(mae_x_folds, mae_y_folds, mae_z_folds)):
        print(f"Fold {fold + 1}: MAE x = {mx:.4f}, y = {my:.4f}, z = {mz:.4f}")

    print(f"Média MAE x: {results[color]['mean_mae_x']:.4f} ± {results[color]['std_mae_x']:.4f}")
    print(f"Média MAE y: {results[color]['mean_mae_y']:.4f} ± {results[color]['std_mae_y']:.4f}")
    print(f"Média MAE z: {results[color]['mean_mae_z']:.4f} ± {results[color]['std_mae_z']:.4f}")

# Summary of all results
print("\n RESUMO ")
for color in colors:
    print(f"{color.capitalize()} - MAE x: {results[color]['mean_mae_x']:.1f} ± {results[color]['std_mae_x']:.1f}")
    print(f"{color.capitalize()} - MAE y: {results[color]['mean_mae_y']:.1f} ± {results[color]['std_mae_y']:.1f}")
    print(f"{color.capitalize()} - MAE z: {results[color]['mean_mae_z']:.1f} ± {results[color]['std_mae_z']:.1f}")

