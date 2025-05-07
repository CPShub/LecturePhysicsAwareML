from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def _read_and_transform_data_from_path(file_path: Path) -> np.ndarray:
    return pd.read_csv(
        file_path,
        sep=',',
        header=None
    ).to_numpy().transpose()[1:, :]


def read_data(
        dir_path: Path,
        file_names_x: list[str | Path],
        file_names_y: list[str | Path],
        file_names_dy: list[str | Path]
    ) -> tuple[np.ndarray, ...]:
    """Reads and append data from multiple text files.

        All files must be in the same directory specified by ``dir_path``.

    Args:
        dir_path: Path to the directory containing the data files.
        file_names_x: List of filenames for the input data.
        file_names_y: List of filenames for the output data.
        file_names_dy: List of filenames for the deriviative output data.

    Returns:
        Tuple of numpy arrays containing the concatenated data for x, y, and dy.
    """

    x_list = []
    y_list = []
    dy_list = []
    for fx, fy, fdy in zip(file_names_x, file_names_y, file_names_dy):
        x_list.append(_read_and_transform_data_from_path(dir_path / fx))
        y_list.append(_read_and_transform_data_from_path(dir_path / fy))
        dy_list.append(_read_and_transform_data_from_path(dir_path / fdy))
    x = np.vstack(x_list)
    y = np.vstack(y_list)
    dy = np.vstack(dy_list)
    return x, y, dy


def quasistatic(A: float, t: np.ndarray):
    """Plots a quasi-static, linearly incerasing excitation."""
    y = A * t / 5

    fig, ax = plt.subplots(figsize=(12, 2))
    ax.plot(t, y, 'b-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (N)')
    ax.grid()
    fig.suptitle('Quasi-static excitation', fontsize=16,  y=.995)
    plt.tight_layout()
    plt.show()

def dirac(A, t0, t):
    """Plots a Dirac delta function excitation."""
    y = np.where((t >= t0) * (t <= (t0 + 0.01)), A, 0 )

    fig, ax = plt.subplots(figsize=(12, 2))
    ax.plot(t, y, 'b-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (N)')
    ax.grid()
    fig.suptitle('Dirac excitation', fontsize=16,  y=.995)
    plt.tight_layout()
    plt.show()


def step(A, t0, t):
    """Plots a step function excitation."""
    y = np.where(t >= t0, A, 0 )

    fig, ax = plt.subplots(figsize=(12, 2))
    ax.plot(t, y, 'b-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (N)')
    ax.grid()
    fig.suptitle('Step excitation', fontsize=16,  y=.995)
    plt.tight_layout()
    plt.show()


def sine(A, f, t):
    """Plots a sine function excitation."""
    y = A * np.sin(2 * np.pi * f * t)

    fig, ax = plt.subplots(figsize=(12, 2))
    ax.plot(t, y, 'b-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (N)')
    ax.grid()
    fig.suptitle('Sine excitation', fontsize=16,  y=.995)
    plt.tight_layout()
    plt.show()


def multisin(A, f0, n, i, n_ph, t):
    """Plots a multisine function excitation."""
    y = np.zeros(t.shape[0])
    for k in range(n):
        phi_k = -(k + 1) * k * np.pi / n
        phi_i = 2 * np.pi * i / n_ph
        y += A * np.sin(2 * np.pi * f0 * (k + 1) * t + phi_k + phi_i)

    fig, ax = plt.subplots(figsize=(12, 2))
    ax.plot(t, y, 'b-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (N)')
    ax.grid()
    fig.suptitle('Multisine excitation', fontsize=16,  y=.995)
    plt.tight_layout()
    plt.show()