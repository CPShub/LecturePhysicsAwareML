import jax
import jax.numpy as jnp
from jaxtyping import Array

import equinox as eqx
import paramax

from matplotlib import pyplot as plt

@eqx.filter_jit
def compute_mse(model, x, w):
    model = paramax.unwrap(model)
    w_pred = jax.vmap(model)(x)
    return jnp.mean((w_pred - w)**2)


@eqx.filter_jit
def model_fn(model, x: Array) -> Array:
    model = paramax.unwrap(model)
    return jax.vmap(model.forward)(x)
    

def evaluate(model, x, w, w_x, M, Q, w_xxxx):
    err = compute_mse(model, x, w)

    print("\nEvaluation:\n-----------")
    print(f"  MSE: {err}")

    pred = model_fn(model, x)

    fig, axs = plt.subplots(3, 2, figsize=(10, 12))
    axs[0, 0].plot(x, pred[0], 'r', label='PINN')
    axs[0, 0].plot(x, w, 'b', linestyle='--', label='analytical')
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('w')
    axs[0, 0].grid()
    axs[0, 0].legend()

    axs[0, 1].plot(x, pred[1], 'r', label='PINN')
    axs[0, 1].plot(x, w_x, 'b', linestyle='--', label='analytical')
    axs[0, 1].set_xlabel('x')
    axs[0, 1].set_ylabel('dw')
    axs[0, 1].grid()
    axs[0, 1].legend()

    axs[1, 0].plot(x, pred[2], 'r', label='PINN')
    axs[1, 0].plot(x, M, 'b', linestyle='--', label='analytical')
    axs[1, 0].set_xlabel('x')
    axs[1, 0].set_ylabel('M')
    axs[1, 0].grid()
    axs[1, 0].legend()

    axs[1, 1].plot(x, pred[3], 'r', label='PINN')
    axs[1, 1].plot(x, Q, 'b', linestyle='--', label='analytical')
    axs[1, 1].set_xlabel('x')
    axs[1, 1].set_ylabel('Q')
    axs[1, 1].grid()
    axs[1, 1].legend()

    axs[2, 0].plot(x, pred[4], 'r', label='PINN')
    axs[2, 0].plot(x, w_xxxx, 'b', linestyle='--', label='analytical')
    axs[2, 0].set_xlabel('x')
    axs[2, 0].set_ylabel('ddddw')
    axs[2, 0].grid()
    axs[2, 0].legend()

    axs[2, 1].axis("off")

    plt.tight_layout()
    plt.show()