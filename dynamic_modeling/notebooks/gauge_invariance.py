"""
Here I briefly want to test if the Euler-Lagrange equation is invariant 
under specific (gauge-) transformations of the Lagrangian.

Gauge transformation:
L' = L + d/dt f(q, t)
"""

# %% Import
from dynamic_modeling import (
    ODESolver
)
from dynamic_modeling.two_mass_oscillator import (
    TwoMassOscillator,
    plot_trajectory,
    plot_energy,
)

import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx

from jaxtyping import Array, Scalar
from typing import Callable

import matplotlib.pyplot as plt

# Use jax in float 64 mode (not strictly necessary)
jax.config.update('jax_enable_x64', True)

# Define global keyword arguments for plotting
true_kwargs = dict(color="black", lw=2, marker='o', markevery=20)

# Set the seeds for the random number generator
key = jr.key(0)
model_key, loader_key = jr.split(key)


class EulerLagrangeEq(eqx.Module):
    lagrangian: Callable[[Array], Scalar]

    def __init__(self, lagrangian: Callable[[Array], Scalar]):
        self.lagrangian = lagrangian

    def __call__(self, t: Array, y: Array, u: Array | None = None) -> Array:
        q, q_t = jnp.split(y, 2, axis=-1)

        # Redefine the Lagrangian in terms of the generalized coordinates and velocities.
        # This is necessary to differentiate with respect to q and q_t individually.
        def _lagrangian(q, q_t):
            y = jnp.concat([q, q_t], axis=-1)
            return self.lagrangian(y)

        q_tt = jax.numpy.linalg.pinv(jax.hessian(_lagrangian, 1)(q, q_t)) @ (
                jax.grad(_lagrangian, 0)(q, q_t)
                - jax.jacfwd(jax.grad(_lagrangian, 1), 0)(q, q_t) @ q_t
            )
        return jnp.concat([q_t, q_tt])
    

# %% Define models

two_mass_oscillator = TwoMassOscillator(m1=1.0, m2=1.0, k1=1.0, k2=0.5, c1=0.0, c2=0.0) # Change this in later tasks
true_system = ODESolver(two_mass_oscillator)

def f(q):
    q1, q2 = q
    return 2*q1 + jnp.sin(3*q2)

def new_lagranian(y: Array) -> Scalar:
    q, q_t = jnp.split(y, 2, axis=-1)
    L_true = two_mass_oscillator.get_lagrangian(y)
    df_dq = jax.grad(f)(q)
    return 2*L_true - jnp.inner(df_dq, q_t)

eleq = EulerLagrangeEq(new_lagranian)
lagrange_system = ODESolver(eleq)


# %% Plot the model prediction(s)
y0_eval = jnp.array([0.7, -0.2, 0.0, 0.0])  # <<< Initial condition for evaluation
t_max_eval = 50                             # <<< Length of evaluation trajectory
states = "positions"                        # <<< Plot only: "positions", "velocities" or "all"
ts_eval = jnp.linspace(0, t_max_eval, 1000)


fig, axes = plt.subplots(3, 1, figsize=(12, 9))

ys_true = true_system(ts_eval, y0_eval)
plot_trajectory(ts_eval, ys_true, ax=axes[0], label="True State", states=states, **true_kwargs)
E_true = jax.vmap(two_mass_oscillator.get_energy)(ys_true)
plot_energy(ts_eval, E_true, ax=axes[1], label="True Energy", **true_kwargs)   # type: ignore
lagrangian_true = jax.vmap(two_mass_oscillator.get_lagrangian)(ys_true)
plot_energy(ts_eval, lagrangian_true, ax=axes[2], label="\"True\" Lagrangian", **true_kwargs)   # type: ignore

ys_lagr = lagrange_system(ts_eval, y0_eval)
plot_trajectory(ts_eval, ys_lagr, ax=axes[0], label="True State", states=states, c='red')
E_lagr = jax.vmap(two_mass_oscillator.get_energy)(ys_lagr)
plot_energy(ts_eval, E_lagr, ax=axes[1], label="True Energy", c='red')   # type: ignore
lagrangian_lagr = jax.vmap(lagrange_system.func.lagrangian)(ys_lagr)
plot_energy(ts_eval, lagrangian_lagr, ax=axes[2], label="Transformed Lagrangian", c='red')   # type: ignore

axes[2].set(title="Lagrangian")
plt.tight_layout()
plt.show()