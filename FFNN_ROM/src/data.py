"""
Lecture "Physics-aware Machine Learning" (SoSe 23/24)
Task 1: Feed-Forward Neural Networks

==================

Authors: Jasper O. Schommartz
         
04/2024
"""

from os.path import join, dirname, realpath
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

fpath = lambda x, y: join(dirname(realpath('__file__')), 'data', x, y)

def read_data(dir_path, fnames_x, fnames_y, fnames_dy):
    ''' Auxiliary function for loading data from .txt files '''
    x_list = []
    y_list = []
    dy_list = []
    for fx, fy, fdy in zip(fnames_x, fnames_y, fnames_dy):
        x_list.append(pd.read_csv(join(dir_path, fx), sep=',', header=None).to_numpy().transpose()[1:, :])
        y_list.append(pd.read_csv(join(dir_path, fy), sep=',', header=None).to_numpy().transpose()[1:, :])
        dy_list.append(pd.read_csv(join(dir_path, fdy), sep=',', header=None).to_numpy().transpose()[1:, :])
    x = np.vstack(x_list)
    y = np.vstack(y_list)
    dy = np.vstack(dy_list)
    return x, y, dy

# def read_data(dir_name, fnames_x, fnames_y, fnames_dy):
#     ''' Auxiliary function for loading data from .txt files '''
#     x_list = []
#     y_list = []
#     dy_list = []
#     for fx, fy, fdy in zip(fnames_x, fnames_y, fnames_dy):
#         x_list.append(pd.read_csv(fpath(dir_name, fx), sep=',', header=None).to_numpy().transpose()[1:, :])
#         y_list.append(pd.read_csv(fpath(dir_name, fy), sep=',', header=None).to_numpy().transpose()[1:, :])
#         dy_list.append(pd.read_csv(fpath(dir_name, fdy), sep=',', header=None).to_numpy().transpose()[1:, :])
#     x = np.vstack(x_list)
#     y = np.vstack(y_list)
#     dy = np.vstack(dy_list)
#     return x, y, dy

def static(A, t):
    y = A * t / 5

    fig, ax = plt.subplots(figsize=(12, 2))
    ax.plot(t, y, 'b-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'Force (N)')
    ax.grid()
    fig.suptitle('Quasi-static excitation', fontsize=16,  y=.995)
    plt.tight_layout()
    plt.show()

def dirac(A, t0, t):
    y = np.where((t >= t0) * (t <= (t0 + 0.01)), A, 0 )

    fig, ax = plt.subplots(figsize=(12, 2))
    ax.plot(t, y, 'b-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'Force (N)')
    ax.grid()
    fig.suptitle('Dirac excitation', fontsize=16,  y=.995)
    plt.tight_layout()
    plt.show()

def step(A, t0, t):
    y = np.where(t >= t0, A, 0 )

    fig, ax = plt.subplots(figsize=(12, 2))
    ax.plot(t, y, 'b-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'Force (N)')
    ax.grid()
    fig.suptitle('Step excitation', fontsize=16,  y=.995)
    plt.tight_layout()
    plt.show()

def sine(A, f, t):
    y = A * np.sin(2 * np.pi * f * t)

    fig, ax = plt.subplots(figsize=(12, 2))
    ax.plot(t, y, 'b-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'Force (N)')
    ax.grid()
    fig.suptitle('Sine excitation', fontsize=16,  y=.995)
    plt.tight_layout()
    plt.show()

def multisin(A, f0, n, i, n_ph, t):
    y = np.zeros(t.shape[0])
    for k in range(n):
        phi_k = -(k + 1) * k * np.pi / n
        phi_i = 2 * np.pi * i / n_ph
        y += A * np.sin(2 * np.pi * f0 * (k + 1) * t + phi_k + phi_i)

    fig, ax = plt.subplots(figsize=(12, 2))
    ax.plot(t, y, 'b-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'Force (N)')
    ax.grid()
    fig.suptitle('Multisine excitation', fontsize=16,  y=.995)
    plt.tight_layout()
    plt.show()