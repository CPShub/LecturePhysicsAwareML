"""
Lecture "Physics-aware Machine Learning" (SoSe 23/24)
Task 1: Feed-Forward Neural Networks

==================

Authors: Dominik K. Klein, Jasper O. Schommartz
         
04/2024
"""

# %%   
"""
Import modules

"""

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf


def plot_loss(h):
    
    plt.figure()
    plt.semilogy(h.history['loss'], label='training loss')
    plt.grid(which='both')
    plt.xlabel('calibration epoch')
    plt.ylabel('log$_{10}$ MSE')
    plt.legend()
    plt.show()
    
    
    
def plot_data_model(xs, ys, xs_c, ys_c, model, model_name, case, ns):
    
    plt.figure()
    plt.scatter(xs_c[::ns], ys_c[::ns], c='green', label = 'calibration data')
    plt.plot(xs, ys, c='black', linestyle='--', label=f'{case} function')
    plt.plot(xs, model.predict(xs), label=model_name, color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    
    
def plot_data_model_grad(xs, ys, dys, xs_c, ys_c, dys_c, model, model_name, case, ns):
    
    ys_m, dys_m = model.predict(xs)
    
    plt.figure()
    plt.scatter(xs_c[::ns], ys_c[::ns], c='green', label = 'calibration data')
    plt.plot(xs, ys, c='black', linestyle='--', label=f'{case} function')
    plt.plot(xs, ys_m, label=model_name, color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.scatter(xs_c[::ns], dys_c[::ns], c='green', label = 'calibration data')
    plt.plot(xs, dys, c='black', linestyle='--', label=f'{case} function')
    plt.plot(xs, dys_m, label=model_name, color='red')
    plt.xlabel('x')
    plt.ylabel('dy_dx')
    plt.legend()
    plt.show()
    
    
def plot_data(xs, ys, xs_c, ys_c, case, ns):
    
    plt.figure(2)
    plt.scatter(xs_c[::ns], ys_c[::ns], c='green', label = 'calibration data')
    plt.plot(xs, ys, c='black', linestyle='--', label=f'{case} function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

