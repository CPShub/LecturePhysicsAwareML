from typing import Tuple, Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array

from .config import Config


# Define type aliases
BeamSolution = Tuple[Array, Array, Array, Array, Array]
BCFunction = Callable[[float, float, float, float, float, Array], BeamSolution]


@jax.jit
def bc_case_0(
    E: float, A: float, II: float, L: float, F: float, f0: float, q0: float, x: Array
) -> BeamSolution:
    """Analytical solution for a cantilever beam with point load at the tip."""
    EI = E * II
    xL = x / L
    u = -(F**2) * L**5 / (2 * EI**2) * (xL**5 / 20 - xL**4 / 4 + xL**3 / 3)
    w = F * L**3 / EI * (-(xL**3) / 6 + xL**2 / 2)
    w_x = F * L**2 / EI * (-(xL**2) / 2 + xL)
    N = jnp.zeros_like(x)
    M = F * L * (xL - 1)
    Q = F * jnp.ones_like(x)

    return u, w, w_x, N, M, Q


def get_bc_case(bc_case: int) -> BCFunction:
    if bc_case == 0:
        return bc_case_0
    else:
        raise NotImplementedError(f"Boundary condition case {bc_case} not implemented.")


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
        y = jax.vmap(bc_case, (None, None, None, None, None, None, None, 0))(
            config.E, config.A, config.II, config.L, config.F, config.f, config.q, x
        )

        u, w, w_x, N, M, Q = y

        # some auxiliary variables
        EA = config.E * config.A
        EI = config.E * config.II

        params = {
            "L": config.L,
            "EA": EA,
            "EI": EI,
            "f0": config.f,
            "q0": config.q,
        }

        return x, (u, w, w_x, N, M, Q), bc, params

    return wrapper
