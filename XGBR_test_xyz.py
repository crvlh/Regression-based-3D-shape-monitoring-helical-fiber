# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 12:41:29 2025

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
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import time

# Fixar seeds
SEED = 0
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Configurações iniciais
n_wavelength_effective = 1063
skip_rows = 1062
n_labels = 15
colors = ['ciano', 'laranja', 'verde', 'amarelo', 'rosa']

# Importar dados de treinamento
folder = "E:\\Doutorado\\TESE 1\\OFS & JLT EXTENDIDO\\JLT\\SENSOR 2"
df_train = pd.read_excel(os.path.join(folder, "trein_final_quasi_z.xlsx"), header=None)
num_samples_train = df_train.shape[0]
configs_train = df_train.iloc[:, 1:n_labels + 1].values

# Import training/validation spectra .txt
file_list_train = sorted(glob.glob(os.path.join(folder, "TREINAMENTO", "*.txt")),
                         key=lambda x: int(os.path.basename(x).split(".")[0]))
data_train = [pd.read_csv(file, delimiter='\t', skiprows=skip_rows,
                          nrows=n_wavelength_effective, usecols=range(1, 2)) for file in file_list_train]
data_train = np.reshape(np.array(data_train), (num_samples_train, n_wavelength_effective))

# Import test targets .xlsx
df_test = pd.read_excel(os.path.join(folder, "test_final_quasi_z.xlsx"), header=None)
num_samples_test = df_test.shape[0]
configs_test = df_test.iloc[:, 1:n_labels + 1].values

# Importar test spectra .txt
file_list_test = sorted(glob.glob(os.path.join(folder, "TESTE3", "*.txt")),
                        key=lambda x: int(os.path.basename(x).split(".")[0]))
data_test = [pd.read_csv(file, delimiter='\t', skiprows=skip_rows,
                         nrows=n_wavelength_effective, usecols=range(1, 2)) for file in file_list_test]
data_test = np.reshape(np.array(data_test), (num_samples_test, n_wavelength_effective))

# Scaler
scaler_input = MinMaxScaler()
data_train = scaler_input.fit_transform(data_train.reshape(-1, data_train.shape[-1])).reshape(data_train.shape)
data_test = scaler_input.transform(data_test.reshape(-1, data_test.shape[-1])).reshape(data_test.shape)

scaler_output = MinMaxScaler()
configs_train = scaler_output.fit_transform(configs_train)
configs_test = scaler_output.transform(configs_test)

# PCA
pca = PCA(n_components=8)
data_train = pca.fit_transform(data_train)
data_test = pca.transform(data_test)

print(f"Variância explicada pelo PCA: {sum(pca.explained_variance_ratio_):.4f}")

# Modelos XGBoost para X, Y, Z
xgb_models_x, xgb_models_y, xgb_models_z = [], [], []

start_time = time.time()
for i in range(5):
    xgb_x = XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.1, subsample=0.7)
    xgb_y = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.075, subsample=0.7)
    xgb_z = XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.1, subsample=0.7)

    xgb_x.fit(data_train, configs_train[:, i])
    xgb_y.fit(data_train, configs_train[:, i + 5])
    xgb_z.fit(data_train, configs_train[:, i + 10])

    xgb_models_x.append(xgb_x)
    xgb_models_y.append(xgb_y)
    xgb_models_z.append(xgb_z)

print(f"Tempo de treinamento: {time.time() - start_time:.2f} segundos")

# Test prediction
y_pred = np.zeros((num_samples_test, n_labels))
for i in range(5):
    y_pred[:, i] = xgb_models_x[i].predict(data_test)
    y_pred[:, i + 5] = xgb_models_y[i].predict(data_test)
    y_pred[:, i + 10] = xgb_models_z[i].predict(data_test)

# Scaler
y_pred = scaler_output.inverse_transform(y_pred)
configs_test = scaler_output.inverse_transform(configs_test)

# 3D reconstruction setup
cor_pred = '#000000'  
cor_real = '#ff7f0e'    
z_base = np.array([11, 8, 5, 2, 0])
mostrar_circulos = True  # Ativa ou desativa os círculos transparentes
pred_test = y_pred
  
#   Plot single angle, oblique (on/off)
for i in range(num_samples_test):
    x = pred_test[i, :5]
    y = pred_test[i, 5:10]
    z = z_base - pred_test[i, 10:15]

    x_real = configs_test[i, :5]
    y_real = configs_test[i, 5:10]
    z_real = z_base - configs_test[i, 10:15]

    if len(x) == len(y) == len(z):
        tck, u = splprep([x, y, z], s=0.01, k=2)
        x_new, y_new, z_new = splev(np.linspace(u.min(), u.max(), 1000), tck)

        tck_real, u_real = splprep([x_real, y_real, z_real], s=0.00, k=2)
        x_new_real, y_new_real, z_new_real = splev(np.linspace(u_real.min(), u_real.max(), 1000), tck_real)

        fig = plt.figure(figsize=(12, 6))  
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(x_new, y_new, z_new, color=cor_pred, label='Predicted shape', linewidth=1.5, alpha=0.99)
        ax.plot(x_new_real, y_new_real, z_new_real, color=cor_real, label='Actual shape', linewidth=1.5, alpha=0.99)

        if mostrar_circulos:
            num_circles = 22
            dist_circles = (u.max() - u.min()) / num_circles
            u_vals = np.arange(u.min(), u.max(), dist_circles)
            x_c, y_c, z_c = splev(u_vals, tck)
            x_r, y_r, z_r = splev(u_vals, tck_real)

            for j in range(len(x_c)):
                angle = np.linspace(0, 2 * np.pi, 100)
                circ_x = np.cos(angle)
                circ_y = np.sin(angle)
                circ_z = np.zeros_like(angle)
                ax.plot(x_c[j] + circ_x, y_c[j] + circ_y, z_c[j] + circ_z, color=cor_pred, alpha=0.3, linewidth=1.0)
                ax.plot(x_r[j] + circ_x, y_r[j] + circ_y, z_r[j] + circ_z, color=cor_real, alpha=0.3, linewidth=1.0)

        ax.set_xlim([7, -7])
        ax.set_ylim([-7, 7])
        ax.set_zlim([0, 12])
        ax.set_xlabel('x (cm)', fontsize=14)
        ax.set_ylabel('y (cm)', fontsize=14)
        ax.set_zlabel('z (cm)', fontsize=14, labelpad=-4.5)  
        ax.zaxis.set_tick_params(pad=0)       
        ax.tick_params(labelsize=14)
        ax.view_init(elev=45, azim=135)
        ax.legend(fontsize=14, loc='upper right', bbox_to_anchor=(1, 0.85))

        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        #plt.savefig(f"E:\\Doutorado\\TESE 1\\OFS & JLT EXTENDIDO\\JLT\\figuras\\figuras reconstrucoes\\svr 3 artigo/fig_{i:03d}.png", bbox_inches='tight', dpi=400)
        #plt.show()

#   Plot 3 angle (on/off)
for i in range(num_samples_test):
    x = pred_test[i, :5]
    y = pred_test[i, 5:10]
    z = z_base - pred_test[i, 10:15]

    x_real = configs_test[i, :5]
    y_real = configs_test[i, 5:10]
    z_real = z_base - configs_test[i, 10:15]

    if len(x) == len(y) == len(z):
        tck, u = splprep([x, y, z], s=0.01, k=2)
        x_new, y_new, z_new = splev(np.linspace(u.min(), u.max(), 1000), tck)

        tck_real, u_real = splprep([x_real, y_real, z_real], s=0.00, k=2)
        x_new_real, y_new_real, z_new_real = splev(np.linspace(u_real.min(), u_real.max(), 1000), tck_real)

        fig = plt.figure(figsize=(18, 6))  # largura maior para 3 subplots

        view_angles = [(0, 'Lateral View'), (45, 'Oblique View'), (90, 'Top View')]

        for idx, (elev, titulo) in enumerate(view_angles):
            ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
            
            ax.plot(x_new, y_new, z_new, color=cor_pred, label='Predicted shape', linewidth=1.0, alpha=0.99)
            ax.plot(x_new_real, y_new_real, z_new_real, color=cor_real, label='Actual shape', linewidth=1.0, alpha=0.99)

            if mostrar_circulos:
                num_circles = 22
                dist_circles = (u.max() - u.min()) / num_circles
                u_vals = np.arange(u.min(), u.max(), dist_circles)
                x_c, y_c, z_c = splev(u_vals, tck)
                x_r, y_r, z_r = splev(u_vals, tck_real)

                for j in range(len(x_c)):
                    angle = np.linspace(0, 2 * np.pi, 100)
                    circ_x = np.cos(angle)
                    circ_y = np.sin(angle)
                    circ_z = np.zeros_like(angle)
                    ax.plot(x_c[j] + circ_x, y_c[j] + circ_y, z_c[j] + circ_z, color=cor_pred, alpha=0.15, linewidth=1.5)
                    ax.plot(x_r[j] + circ_x, y_r[j] + circ_y, z_r[j] + circ_z, color=cor_real, alpha=0.15, linewidth=1.5)

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
        plt.savefig(f"E:\\Doutorado\\TESE 1\\OFS & JLT EXTENDIDO\\JLT\\figuras\\figuras reconstrucoes\\xgbr 3/fig_{i:03d}.png", bbox_inches='tight', dpi=300)
        #plt.show()

# Code compability
pred_test = y_pred