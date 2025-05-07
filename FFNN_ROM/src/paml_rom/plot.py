from matplotlib import pyplot as plt
import numpy as np


def plot_forces(Qf):
    """Plots forces over data points"""
    m = Qf.shape[0]
    n = Qf.shape[1]
    fig, axs = plt.subplots(n, 1, figsize=(12, 2 * n))
    for i in range(n):
        axs[i].plot(np.arange(m), Qf[:, i])
        axs[i].set_xlabel("Data point")
        axs[i].set_ylabel(rf"Qf{i + 1}")
    fig.suptitle("Time series of internal forces", fontsize=16, y=0.995)
    plt.tight_layout()


def plot_stiffnesses(QKQ):
    """Plot stiffnesses over data points"""
    m = QKQ.shape[0]
    n = QKQ.shape[1]
    fig, axs = plt.subplots(n, 1, figsize=(12, 2 * n))
    for i in range(n):
        axs[i].plot(np.arange(m), QKQ[:, i])
        axs[i].set_xlabel("Data point")
        axs[i].set_ylabel(f"QKQ{i + 1}")
    fig.suptitle("Time series of internal stiffnesses", fontsize=16, y=0.995)
    plt.tight_layout()


def plot_predictions(x, Qf, Qf_pred, QKQ, QKQ_pred, E=None):
    """Plots force and stiffness predictions over data"""
    m = x.shape[0]
    n = Qf.shape[1]

    # Plot correlation of forces
    fig, axs = plt.subplots(int(n / 2), 2, figsize=(8, 8))
    for i, ax in enumerate(axs.reshape(-1)):
        ax.plot(Qf[:, i], Qf_pred[:, i], "b.")
        ax.set_xlabel(f"Data: Qf{i + 1}")
        ax.set_ylabel(f"Prediction: Qf{i + 1}")
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        ax.plot([-1e8, 1e8], [-1e8, 1e8], "k--")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
    fig.suptitle("Correlation of internal forces", fontsize=16, y=0.995)
    plt.tight_layout()

    # Plot correlation of stiffnesses
    fig, axs = plt.subplots(int(n**2 / 2), 2, figsize=(8, 8 * 4))
    for i, ax in enumerate(axs.reshape(-1)):
        ax.plot(QKQ[:, i], QKQ_pred[:, i], "b.")
        ax.set_xlabel(f"Data: QKQ{i + 1}")
        ax.set_ylabel(f"Prediction: QKQ{i + 1}")
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        ax.plot([-1e8, 1e8], [-1e8, 1e8], "k--")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
    fig.suptitle("Correlation of internal stiffnesses", fontsize=16, y=0.995)
    plt.tight_layout()

    # Plot internal Energy over data points
    if E is not None:
        fig, ax = plt.subplots(figsize=(12, 2))
        ax.plot(np.arange(m), E, "b-", label="FFNN")
        ax.set_xlabel("Data point")
        ax.set_ylabel("E")
        ax.legend()
        fig.suptitle("Time series of internal energy", fontsize=16, y=0.995)
        plt.tight_layout()

    # Plot forces values over data points
    fig, axs = plt.subplots(n, 1, figsize=(12, 8))
    for i in range(n):
        axs[i].plot(np.arange(m), Qf[:, i], "r--", label="FEM")
        axs[i].plot(np.arange(m), Qf_pred[:, i], "b-", label="FFNN")
        axs[i].set_xlabel("Data point")
        axs[i].set_ylabel(f"Qf{i + 1}")
        axs[i].legend()
    fig.suptitle("Time series of function values", fontsize=16, y=0.995)
    plt.tight_layout()

    # Plot stiffnesses over data points
    fig, axs = plt.subplots(n**2, 1, figsize=(12, 2 * n**2))
    for i in range(n**2):
        axs[i].plot(np.arange(m), QKQ[:, i], "r--", label="FEM")
        axs[i].plot(np.arange(m), QKQ_pred[:, i], "b-", label="FFNN")
        axs[i].set_xlabel("Data point")
        axs[i].set_ylabel(f"QKQ{i + 1}")
        axs[i].legend()
    fig.suptitle("Time series of internal stiffnesses", fontsize=16, y=0.995)
    plt.tight_layout()
