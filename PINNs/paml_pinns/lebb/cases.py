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
    q: float,
    x: Array
) -> BeamSolution:
    """Analytical solution for a cantilever beam with point load at the tip."""
    w = (0.5 * F * L * x**2 - 1. / 6. * F * x**3) / EI
    w_x = (F * L * x - 0.5 * F * x**2) / EI
    M = - F * (L - x)
    Q = F * jnp.ones_like(x)
    w_xxxx = jnp.zeros_like(x)

    return w, w_x, M, Q, w_xxxx


# === -------------------------------------------------------------------- === #
# bc_case_1
# === -------------------------------------------------------------------- === #

@jax.jit
def bc_case_1(
    EI: float,
    L: float,
    F: float,
    q: float,
    x: Array
) -> BeamSolution:
    """Analytical solution for a beam with constant line load and moment-free
    ends"""
    xL = x / L
    w = (xL**4 - 2 * xL**3 + xL) * q * L**4 / 24 / EI
    w_x = (4 * xL**3 - 6 * xL**2 + 1) * q * L**3 / 24 / EI
    M = - (xL**2 - xL) * q * L**2 / 2
    Q = - (2 * xL - 1) * q * L / 2
    w_xxxx = q / EI * jnp.ones_like(x)

    return w, w_x, M, Q, w_xxxx


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


# === -------------------------------------------------------------------- === #
# get_data_decorator
# === -------------------------------------------------------------------- === #

def get_data_decorator(bc_func: Callable[[Config], dict[str, Array]]):
    """
    A decorator function that generated data for a given boundary condition
    function and take care of dedimesionalizing the input and outputs. This
    function's main purpose is to hide a lot of the internal functionality from
    the user, so he/she can focus on defining boundary conditions.
    """
    def wrapper(config: Config):
        bc_case = get_bc_case(config.bc_case)
        bc = bc_func(config)

        x = jnp.linspace(0.0, config.L, config.dataset_size).reshape(-1, 1)
        y = jax.vmap(bc_case, (None, None, None, None, 0))(config.EI, config.L, config.F, config.q, x)

        w, w_x, M, Q, w_xxxx = y

        if config.non_dim:
            x0 = config.L
            q0 = 1.0 if config.q == 0.0 else config.q # only valid for constant line loads
            w0 = q0 * x0**4 / config.EI

            x = x / x0
            w = w / w0
            w_x = w_x / w0 * x0
            M = M / q0 / x0**2
            Q = Q / q0 / x0
            w_xxxx = w_xxxx / w0 * x0**4

            bc_coords = ["w_bc_coords", "w_x_bc_coords", "M_bc_coords", "Q_bc_coords"]
            bc_values = ["w_bc_values", "w_x_bc_values", "M_bc_values", "Q_bc_values"]
            factors = [x0, x0, x0, x0]
            value_factors = [w0, w0 / x0, q0 * x0**2, q0 * x0]

            for coord, factor in zip(bc_coords, factors):
                if bc[coord] is not None:
                    bc[coord] = bc[coord] / factor

            for value, factor in zip(bc_values, value_factors):
                if bc[value] is not None:
                    bc[value] = bc[value] / factor
            
            EI = 1.0
            L = 1.0
            q = 0.0 if config.q == 0.0 else 1.0 # only valid for constant line loads

        else:
            EI = config.EI
            L = config.L
            q = config.q

        return x, (w, w_x, M, Q, w_xxxx), bc, EI, L, q

    return wrapper

