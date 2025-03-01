import jax
import jax.numpy as jnp
from jaxtyping import Array


# === -------------------------------------------------------------------- === #
# bc_case_0
# === -------------------------------------------------------------------- === #

@jax.jit
def bc_case_0(EI: float, F: float, L: float, x: Array):
    """Analytical solution for a cantilever beam with point load at the tip."""
    w = (0.5 * F * L * x**2 - 1. / 6. * F * x**3) / EI
    w_x = (F * L * x - 0.5 * F * x**2) / EI
    M = F * (x - L)
    Q = F * jnp.ones_like(x)
    w_xxxx = jnp.zeros_like(x)

    return (
        x,
        w,
        w_x,
        M,
        Q,
        w_xxxx
    )


# === -------------------------------------------------------------------- === #
# bc_case_1
# === -------------------------------------------------------------------- === #

@jax.jit
def bc_case_1(EI: float, q: float, L: float, x: Array):
    """Analytical solution for a beam with constant line load and moment-free
    ends"""
    xL = x / L
    w = (xL**4 - 2 * xL**3 + xL) * q * L**4 / 24 / EI
    w_x = (4 * xL**3 - 6 * xL**2 + 1) * q * L**3 / 24 / EI
    M = (-xL**2 + xL) * q * L**2 / 2
    Q = (-2 * xL + 1) * q * L / 2
    w_xxxx = q / EI * jnp.ones_like(x)

    return (
        x,
        w,
        w_x,
        M,
        Q,
        w_xxxx
    )