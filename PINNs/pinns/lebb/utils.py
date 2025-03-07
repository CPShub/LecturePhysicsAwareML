from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array

from .config import Config
from .cases import get_bc_case


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

