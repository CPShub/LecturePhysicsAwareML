import jax.numpy as jnp
from jaxtyping import Array


def polar2cartesian(y: Array) -> Array:
    """
    Convert polar coordinates to cartesian coordinates.
    """
    r, θ, vr, vθ = jnp.split(y, 4, axis=-1)
    qx = r * jnp.sin(θ)
    qy = -r * jnp.cos(θ)
    vx = vr * jnp.sin(θ) + r * vθ * jnp.cos(θ)
    vy = -vr * jnp.cos(θ) + r * vθ * jnp.sin(θ)
    return jnp.concat([qx, qy, vx, vy], axis=-1)