from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from jaxtyping import Array
from typing import Literal


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
        ax.plot(ts, ys[:, 0], label=label + R" $q_1$", ls="-", **kwargs)
        ax.plot(ts, ys[:, 1], label=label + R" $q_2$", ls="-.", **kwargs)
    if states in ["velocities", "all"]:
        ax.plot(ts, ys[:, 2], label=label + R" $v_1$", ls=":", **kwargs)
        ax.plot(ts, ys[:, 3], label=label + R" $v_2$", ls="--", **kwargs)
    ax.set(
        xlabel="Time t",
        ylabel="State $y$",
        title="Trajectory",
    )
    ax.legend()
    return ax


def plot_energy(ts, energy, ax=None, **kwargs):
    if energy.ndim != 1:
        raise ValueError("energy must be a 1D array with the same shape as ts.")

    if ax is None:
        ax = plt.gca()

    kwargs.setdefault("label", "Energy")
    kwargs.setdefault("ls", "-")
    ax.plot(ts, energy, **kwargs)
    ax.set(
        xlabel="Time t",
        ylabel="Energy",
        title="Energy",
    )
    ax.legend()

    return ax


def set_minimum_axis_limits(ax, min_span=1e-3):
    # Get current axis limits
    ymin, ymax = ax.get_ylim()

    # Enforce minimum span on y-axis
    if ymax - ymin < min_span:
        center = (ymax + ymin) / 2
        ymin = center - min_span / 2
        ymax = center + min_span / 2
        ax.set_ylim(ymin, ymax)
