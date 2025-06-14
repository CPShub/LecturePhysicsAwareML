import numpy as np

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from IPython.display import HTML

from jaxtyping import Array
from typing import Literal

from pathlib import Path

from .._plotting import make_spring


def animate_spring_pendulum(
    ts: Array,
    qs: Array,
    fps: int = 30,
    speedup: int = 3,
    color=None,
    filename: Path | str | None = None,
) -> HTML:
    """Animate one or multiple solutions of the spring pendulum via
    matplotlib in a Jupyter notebook.

    Args:
        ts: 1D array of monotonically increasing time stamps
            with ``shape=(k,)``.
        qs: 2D or 3D array of ``shape=(..., k, n)`` containing the x
            and y coordinates of pendulum mass as the first two
            entries along the last axis. If 3D then the first axis is
            the batch axis.
        fps: Frames per second. Defaults to 30.
        speedup: Speedup of the animation. Defaults to 3.
        color: Color of the pendulum bob(s). Defaults to None, which uses blue for the first
            pendulum bob and gray for all others.
        filename: If a Path or string of a filename is provided, the animation will be saved
            as a gif file.

    Returns:
        HTML display object. When this object is returned by an
        expression or passed to the display function of a jupyter notebook,
        it will result in the data being displayed in the frontend.
    """
    qs = qs if qs.shape[-1] == 2 else qs[..., :2]

    qs = qs if qs.ndim == 3 else qs[np.newaxis, ...]

    fig, ax = plt.subplots()
    x_max, y_max = np.maximum(np.abs(qs).max(axis=(0, 1)), 0.5)

    if color is None:
        bob_colors = ["blue"] + ["gray"] * (qs.shape[0] - 1)
        spring_colors = ["black"] + ["gray"] * (qs.shape[0] - 1)
    else:
        bob_colors = [color] * qs.shape[0]
        spring_colors = ["gray"] * qs.shape[0]

    artists_per_q = []
    for batch_index, (q, bob_color, spring_color) in enumerate(
        zip(qs, bob_colors, spring_colors)
    ):
        ax.set(
            xlim=1.1 * np.array([-x_max, x_max]),
            ylim=1.1 * np.array([-y_max, 0.3]),
            xlabel="$q_x$",
            ylabel="$q_y$",
        )
        ax.set_aspect("equal")
        (spring,) = ax.plot(
            *make_spring(np.array([0.0, 0.0]), q[0], 50, 0.1),
            c=spring_color,
            zorder=3 if batch_index == 0 else 1,
            lw=1,
        )
        bob = ax.scatter(
            q[0, 0],
            q[0, 1],
            s=500,
            marker="o",
            c=bob_color,
            zorder=4 if batch_index == 0 else 2,
        )
        artists_per_q.append((spring, bob))
    ax.scatter(0, 0, s=50, marker="o", facecolors="white", edgecolors="black", zorder=3)

    def animate(t):
        i = np.argmin(np.abs(ts - t))
        for q, (spring, bob) in zip(qs, artists_per_q):
            bob.set_offsets(q[i])
            spring.set_data(*make_spring(np.array([0.0, 0.0]), q[i], 50, 0.1))

    delta_t = ts[-1] - ts[0]
    t_frames = np.linspace(ts[0], ts[-1], int(delta_t * fps / speedup))

    ani = FuncAnimation(fig, animate, frames=t_frames)  # type: ignore

    if filename is not None:
        filename = Path(filename)
        ani.save(filename, fps=fps, dpi=300)

    plt.close()
    return HTML(ani.to_jshtml(fps=fps))


def plot_trajectory(
    ts: Array,
    ys: Array,
    ax: Axes | None = None,
    states: Literal["positions", "velocities", "all"] = "positions",
    **kwargs,
) -> Axes:
    """Plot a trajectory of the spring pendulum.

    Args:
        ts: 1D array of monotonically increasing time stamps
            with ``shape=(k,)``.
        ys: 2D array of ``shape=(k, n)`` containing qx, qy, qx_t, qy_t
            along the last axis.
    """
    if ys.ndim != 2 or ys.shape[-1] < 2:
        raise ValueError("ys must be a 2D array with at least 2 columns.")

    if ax is None:
        ax = plt.gca()

    label = kwargs.pop("label", "")
    if states in ["positions", "all"]:
        ax.plot(ts, ys[:, 0], label=label + R" $q_x$", ls="-", **kwargs)
        ax.plot(ts, ys[:, 1], label=label + R" $q_y$", ls="-.", **kwargs)
    if states in ["velocities", "all"]:
        ax.plot(ts, ys[:, 2], label=label + R" $\dot q_x$", ls=":", **kwargs)
        ax.plot(ts, ys[:, 3], label=label + R" $\dot q_y$", ls="--", **kwargs)
    ax.set(
        xlabel="Time t",
        ylabel="State $y$",
        title="Spring Pendulum Trajectory",
    )
    ax.legend()
    return ax


def plot_energy(
    ts: Array,
    energy: Array,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """Plot energy over time.

    Args:
        ts: 1D array of monotonically increasing time stamps
            with ``shape=(k,)``.
        ys: 1D array of ``shape=(k)`` containing energy values.
    """

    if energy.ndim != 1:
        raise ValueError("energy must be a 1D array with the same shape as ts.")

    if ax is None:
        ax = plt.gca()

    kwargs.setdefault("label", "Energy")

    ax.plot(ts, energy, **kwargs)
    ax.set(
        xlabel="Time t",
        ylabel="Energy",
        title="Spring Pendulum Energy",
    )
    ax.legend()
    return ax
