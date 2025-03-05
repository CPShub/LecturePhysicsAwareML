from typing import Tuple, Callable
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array

# Define type aliases
BeamSolution = Tuple[Array, Array, Array, Array, Array]
BCFunction = Callable[[float, float, float, float, float, Array], BeamSolution]

# === -------------------------------------------------------------------- === #
# bc_case_0
# === -------------------------------------------------------------------- === #

@jax.jit
def bc_case_0(
    EI: float, 
    L: float, 
    F: float, 
    f0: float,
    q0: float,
    x: Array
) -> BeamSolution:
    """Analytical solution for a cantilever beam with point load at the tip."""
    u = jnp.zeros_like(x)
    u_x = jnp.zeros_like(x)
    u_xx = jnp.zeros_like(x)
    w = (0.5 * F * L * x**2 - 1. / 6. * F * x**3) / EI
    w_x = (F * L * x - 0.5 * F * x**2) / EI
    w_xx = F * (L - x) / EI
    w_xxx = - F * jnp.ones_like(x) / EI
    w_xxxx = jnp.zeros_like(x)

    return u, u_x, u_xx, w, w_x, w_xx, w_xxx, w_xxxx


# === -------------------------------------------------------------------- === #
# get_bc_case
# === -------------------------------------------------------------------- === #

def get_bc_case(bc_case: int) -> BCFunction:
    if bc_case == 0:
        return bc_case_0
    else:
        raise NotImplementedError(f"Boundary condition case {bc_case} not implemented.")

