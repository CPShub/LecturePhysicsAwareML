from typing import Tuple, Callable
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array

from .config import Config

# Define type aliases
BeamSolution = Tuple[Array, Array, Array, Array, Array]
BCFunction = Callable[[float, float, float, float, Array], BeamSolution]

# === -------------------------------------------------------------------- === #
# bc_case_0
# === -------------------------------------------------------------------- === #

@jax.jit
def bc_case_0(
    EI: float, 
    L: float, 
    F: float, 
    q0: float,
    x: Array
) -> BeamSolution:
    """Analytical solution for a cantilever beam with point load at the tip."""
    w = (0.5 * F * L * x**2 - 1. / 6. * F * x**3) / EI
    w_x = (F * L * x - 0.5 * F * x**2) / EI
    w_xx = F * (L - x) / EI
    w_xxx = - F * jnp.ones_like(x) / EI
    w_xxxx = jnp.zeros_like(x)

    return w, w_x, w_xx, w_xxx, w_xxxx


# === -------------------------------------------------------------------- === #
# bc_case_1
# === -------------------------------------------------------------------- === #

@jax.jit
def bc_case_1(
    EI: float,
    L: float,
    F: float,
    q0: float,
    x: Array
) -> BeamSolution:
    """Analytical solution for a beam with constant line load and moment-free
    ends"""
    xL = x / L
    w = (xL**4 - 2 * xL**3 + xL) * q0 * L**4 / 24 / EI
    w_x = (4 * xL**3 - 6 * xL**2 + 1) * q0 * L**3 / 24 / EI
    w_xx = (xL**2 - xL) * q0 * L**2 / 2 / EI
    w_xxx = (2 * xL - 1) * q0 * L / 2 / EI
    w_xxxx = q0 / EI * jnp.ones_like(x)

    return w, w_x, w_xx, w_xxx, w_xxxx


# === -------------------------------------------------------------------- === #
# get_bc_case
# === -------------------------------------------------------------------- === #

def get_bc_case(bc_case: int) -> BCFunction:
    if bc_case == 0:
        return bc_case_0
    elif bc_case == 1:
        return bc_case_1
    else:
        raise NotImplementedError(f"Boundary condition case {bc_case} not implemented.")

