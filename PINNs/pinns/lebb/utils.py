from collections.abc import Callable
from typing import Tuple

import numpy as np

import jax
import jax.numpy as jnp
from jaxtyping import Array

from .config import Config
from .cases import get_bc_case

# === -------------------------------------------------------------------- === #
# dataloader
# === -------------------------------------------------------------------- === #

def dataloader(arrays, batch_size):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = np.arange(dataset_size)
    while True:
        perm = np.random.permutation(indices)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


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
        y = jax.vmap(bc_case, (None, None, None, None, 0))(config.EI, config.L, config.F, config.q0, x)

        w, w_x, w_xx, w_xxx, w_xxxx = y

        w_bc_coords = bc["w_bc_coords"]
        w_bc_values = bc["w_bc_values"]
        w_x_bc_coords = bc["w_x_bc_coords"]
        w_x_bc_values = bc["w_x_bc_values"]
        w_xx_bc_coords = bc["w_xx_bc_coords"]
        w_xx_bc_values = bc["w_xx_bc_values"]
        w_xxx_bc_coords = bc["w_xxx_bc_coords"]
        w_xxx_bc_values = bc["w_xxx_bc_values"]

        if config.non_dim:
            x0 = config.L
            q0 = 1.0 if config.q0 == 0.0 else config.q0
            w0 = q0 * x0**4 / config.EI

            x = x / x0
            w = w / w0
            w_x = w_x / w0 * x0
            w_xx = w_xx / w0 * x0**2
            w_xxx = w_xxx / w0 * x0**3
            w_xxxx = w_xxxx / w0 * x0**4

            w_bc_coords = w_bc_coords / x0 if w_bc_coords is not None else None
            w_x_bc_coords = w_x_bc_coords / x0 if w_x_bc_coords is not None else None
            w_xx_bc_coords = w_xx_bc_coords / x0 if w_xx_bc_coords is not None else None
            w_xxx_bc_coords = w_xx_bc_coords / x0 if w_xxx_bc_coords is not None else None
            
            w_bc_values = w_bc_values / w0 if w_bc_coords is not None else None
            w_x_bc_values = w_x_bc_values / w0 * x0 if w_x_bc_coords is not None else None
            w_xx_bc_values = w_xx_bc_values / w0 * x0**2 if w_xx_bc_coords is not None else None
            w_xxx_bc_values = w_xxx_bc_values / w0 * x0**3 if w_xxx_bc_coords is not None else None
            
            q0 = 0.0 if config.q0 == 0.0 else 1.0
            L0 = 1.0

        else:
            q0 = config.q0 / config.EI
            L0 = config.L 

        bc = {
            "w_bc_coords": w_bc_coords,
            "w_bc_values": w_bc_values,
            "w_x_bc_coords": w_x_bc_coords,
            "w_x_bc_values": w_x_bc_values,
            "w_xx_bc_coords": w_xx_bc_coords,
            "w_xx_bc_values": w_xx_bc_values,
            "w_xxx_bc_coords": w_xxx_bc_coords,
            "w_xxx_bc_values": w_xxx_bc_values
        }

        return x, (w, w_x, w_xx, w_xxx, w_xxxx), bc, L0, q0

    return wrapper


# === -------------------------------------------------------------------- === #
# get_config_decorator
# === -------------------------------------------------------------------- === #

def get_config_decorator(fun: Callable[[int], Tuple[float, float, float, float]]):
    """
    A decorator to explose only those setting to the user that are relevant
    for completing the tasks.
    """
    def wrapper(
        bc_case: int,
        non_dim: bool,
        steps: int = 50_000
    ):
        EI, L, F, q0 = fun(bc_case)

        config = Config(
            EI=EI,
            L=L,
            F=F,
            q0=q0,
            bc_case=bc_case,
            dataset_size=1_000,
            steps=steps,
            learning_rate=1e-3,
            batch_size=32,
            weights={
                "w_bc": 1.0,
                "w_x_bc": 1.0,
                "w_xx_bc": 1.0,
                "w_xxx_bc": 1.0,
                "rw": 1.0
            },
            non_dim=non_dim
        )
        return config
    return wrapper