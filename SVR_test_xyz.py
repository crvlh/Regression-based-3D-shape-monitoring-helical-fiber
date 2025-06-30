# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 16:13:49 2025

@author: vinic
"""

import os
import glob
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import matplotlib.patches as mpatches
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from joblib import dump
import time

# Random seed configuration
SEED = 0
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Input setup
n_wavelength_effective = # Number of wavelengths used
skip_rows = # Number of header lines to skip
n_labels = # Number of target labels (5 colors × 3 axes)

data_folder_train = "PATH_TO_TRAIN_FOLDER"
config_file_train = "TRAIN_FILE.xlsx"
df_train = pd.read_excel(os.path.join(data_folder_train, config_file_train), header=None)
num_samples_train = df_train.shape[0]
configs_train = np.array(df_train.iloc[:, 1:n_labels + 1].values)

data_folder_spectra_train = "PATH_TO_TRAIN_SPECTRA"
file_list_train = glob.glob(os.path.join(data_folder_spectra_train, "*.txt"))
file_list_train = sorted(file_list_train, key=lambda x: int(os.path.basename(x).split(".")[0]))
data_train = [pd.read_csv(file, delimiter='\t', skiprows=skip_rows, nrows=n_wavelength_effective, usecols=range(1, 2))
              for file in file_list_train]
data_train = np.reshape(np.array(data_train), (num_samples_train, n_wavelength_effective))

data_folder_test = "PATH_TO_TEST_FOLDER"  
config_file_test = "TEST_FILE.xlsx"
df_test = pd.read_excel(os.path.join(data_folder_test, config_file_test), header=None)
num_samples_test = df_test.shape[0]
configs_test = np.array(df_test.iloc[:, 1:n_labels + 1].values)

# Load test spectra
data_folder_spectra_test = "PATH_TO_TEST_SPECTRA"
file_list_test = glob.glob(os.path.join(data_folder_spectra_test, "*.txt"))
file_list_test = sorted(file_list_test, key=lambda x: int(os.path.basename(x).split(".")[0]))
data_test = [pd.read_csv(file, delimiter='\t', skiprows=skip_rows, nrows=n_wavelength_effective, usecols=range(1, 2))
             for file in file_list_test]
data_test = np.reshape(np.array(data_test), (num_samples_test, n_wavelength_effective))

# Normalize spectra input
scaler_input = MinMaxScaler()
data_train = scaler_input.fit_transform(data_train.reshape(-1, data_train.shape[-1])).reshape(data_train.shape)
data_test = scaler_input.transform(data_test.reshape(-1, data_test.shape[-1])).reshape(data_test.shape)

# Apply PCA
n_components = 8


pca = PCA(n_components=n_components)
data_train = pca.fit_transform(data_train)
data_test = pca.transform(data_test)

print(f"Variância explicada pelo PCA: {sum(pca.explained_variance_ratio_):.4f}")
print(f"Dimensão após PCA: {data_train.shape[1]} componentes")

# Normalize targets
scaler_output = MinMaxScaler()
configs_train = scaler_output.fit_transform(configs_train)
configs_test = scaler_output.transform(configs_test)

colors = ['ciano', 'laranja', 'verde', 'amarelo', 'rosa']
svr_models_x, svr_models_y, svr_models_z = [], [], []

# Train SVR models
start_time = time.time()

for i, color in enumerate(colors):
    svr_x = SVR(kernel='rbf', C=10, epsilon=0.01)
    svr_y = SVR(kernel='rbf', C=10, epsilon=0.1)
    svr_z = SVR(kernel='rbf', C=10, epsilon=0.1)

    svr_x.fit(data_train, configs_train[:, i])
    svr_y.fit(data_train, configs_train[:, i + 5])
    svr_z.fit(data_train, configs_train[:, i + 10])

    svr_models_x.append(svr_x)
    svr_models_y.append(svr_y)
    svr_models_z.append(svr_z)
    
print(f"Tempo de treinamento: {time.time() - start_time:.2f} segundos")

# Predict test set
pred_test = np.zeros((num_samples_test, n_labels))
for i in range(5):
    pred_test[:, i] = svr_models_x[i].predict(data_test)
    pred_test[:, i + 5] = svr_models_y[i].predict(data_test)
    pred_test[:, i + 10] = svr_models_z[i].predict(data_test)

pred_test = scaler_output.inverse_transform(pred_test)
configs_test = scaler_output.inverse_transform(configs_test)

# Setup for visualization
z_base = np.array([11, 8, 5, 2, 0])
mostrar_circulos = True

# 3D shape reconstruction and visualization
for i in range(num_samples_test):
    x = pred_test[i, :5]
    y = pred_test[i, 5:10]
    z = z_base - pred_test[i, 10:15]

    x_real = configs_test[i, :5]
    y_real = configs_test[i, 5:10]
    z_real = z_base - configs_test[i, 10:15]

    if len(x) == len(y) == len(z):
        tck, u = splprep([x, y, z], s=0.05, k=2)
        x_new, y_new, z_new = splev(np.linspace(u.min(), u.max(), 1000), tck)

        tck_real, u_real = splprep([x_real, y_real, z_real], s=0.05, k=2)
        x_new_real, y_new_real, z_new_real = splev(np.linspace(u_real.min(), u_real.max(), 1000), tck_real)

        # fig = plt.figure(figsize=(12, 6))  
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot(x_new, y_new, z_new, color='b', label='Predicted shape', linewidth=1.0, alpha=0.99)
        # ax.plot(x_new_real, y_new_real, z_new_real, color='r', label='Actual shape', linewidth=1.0, alpha=0.99)

        # if mostrar_circulos:
        #     num_circles = 20
        #     dist_circles = (u.max() - u.min()) / num_circles
        #     u_vals = np.arange(u.min(), u.max(), dist_circles)
        #     x_c, y_c, z_c = splev(u_vals, tck)
        #     x_r, y_r, z_r = splev(u_vals, tck_real)

        #     for j in range(len(x_c)):
        #         angle = np.linspace(0, 2 * np.pi, 100)
        #         circ_x = np.cos(angle)
        #         circ_y = np.sin(angle)
        #         circ_z = np.zeros_like(angle)
        #         ax.plot(x_c[j] + circ_x, y_c[j] + circ_y, z_c[j] + circ_z, color='b', alpha=0.15, linewidth=1.5)
        #         ax.plot(x_r[j] + circ_x, y_r[j] + circ_y, z_r[j] + circ_z, color='r', alpha=0.15, linewidth=1.5)

        # ax.set_xlim([7, -7])
        # ax.set_ylim([-7, 7])
        # ax.set_zlim([0, 12])
        # ax.set_xlabel('x (cm)', fontsize=14)
        # ax.set_ylabel('y (cm)', fontsize=14)
        # ax.set_zlabel('z (cm)', fontsize=14, labelpad=-4.5)  
        # ax.zaxis.set_tick_params(pad=0)       
        # ax.tick_params(labelsize=14)
        # ax.view_init(elev=45, azim=135)
        # # ax.view_init(elev=5, azim=135)
        # # ax.view_init(elev=90, azim=135)
        # ax.legend(fontsize=14, loc='upper right', bbox_to_anchor=(1, 0.85))

        # plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        # plt.show()

# 3D shape reconstruction and visualization       
        fig = plt.figure(figsize=(18, 6)) 

        view_angles = [(0, 'Lateral View'), (45, 'Oblique View'), (90, 'Top View')]
        
        for idx, (elev, titulo) in enumerate(view_angles):
            ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
            
            ax.plot(x_new, y_new, z_new, color='b', label='Predicted shape', linewidth=1.0, alpha=0.99)
            ax.plot(x_new_real, y_new_real, z_new_real, color='r', label='Actual shape', linewidth=1.0, alpha=0.99)
            
            if mostrar_circulos:
                num_circles = 20
                dist_circles = (u.max() - u.min()) / num_circles
                u_vals = np.arange(u.min(), u.max(), dist_circles)
                x_c, y_c, z_c = splev(u_vals, tck)
                x_r, y_r, z_r = splev(u_vals, tck_real)
        
                for j in range(len(x_c)):
                    angle = np.linspace(0, 2 * np.pi, 100)
                    circ_x = np.cos(angle)
                    circ_y = np.sin(angle)
                    circ_z = np.zeros_like(angle)
                    ax.plot(x_c[j] + circ_x, y_c[j] + circ_y, z_c[j] + circ_z, color='b', alpha=0.15, linewidth=1.5)
                    ax.plot(x_r[j] + circ_x, y_r[j] + circ_y, z_r[j] + circ_z, color='r', alpha=0.15, linewidth=1.5)
        
            ax.set_xlim([7, -7])
            ax.set_ylim([-7, 7])
            ax.set_zlim([0, 12])
            ax.set_xlabel('x (cm)', fontsize=14)
            ax.set_ylabel('y (cm)', fontsize=14)
            ax.set_zlabel('z (cm)', fontsize=14, labelpad=-4.5)
            ax.zaxis.set_tick_params(pad=0)
            ax.tick_params(labelsize=14)
            ax.view_init(elev=elev, azim=135)
            ax.set_title(titulo, fontsize=14)

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=14, loc='upper center', ncol=2)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.show()

# Compute MAE per axis
mae_x = np.zeros(5)
mae_y = np.zeros(5)
mae_z = np.zeros(5)

for i in range(num_samples_test):
    posx_pred = pred_test[i, :5]
    posy_pred = pred_test[i, 5:10]
    posz_pred = pred_test[i, 10:15]
    
    posx_real = configs_test[i, :5]
    posy_real = configs_test[i, 5:10]
    posz_real = configs_test[i, 10:15]

    mae_x += np.abs(posx_pred - posx_real)
    mae_y += np.abs(posy_pred - posy_real)
    mae_z += np.abs(posz_pred - posz_real)

mae_x /= num_samples_test
mae_y /= num_samples_test
mae_z /= num_samples_test

# Print final MAE
for i, color in enumerate(colors):
    print(f"{color.capitalize()} - MAE x: {mae_x[i]:.4f}, y: {mae_y[i]:.4f}, z: {mae_z[i]:.4f}")
