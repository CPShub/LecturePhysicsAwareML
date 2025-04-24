import jax
import jax.numpy as jnp
from jaxtyping import Array

import equinox as eqx
import paramax

from matplotlib import pyplot as plt

@eqx.filter_jit
def compute_mse(model, x, y):
    model = paramax.unwrap(model)
    y_pred = jax.vmap(model)(x)
    losses = []
    for d, p in zip(y, y_pred):
        losses.append(jnp.mean((d - p)**2))
    return tuple(losses)


@eqx.filter_jit
def model_fn(model, x: Array) -> Array:
    model = paramax.unwrap(model)
    return jax.vmap(model)(x)
    

def evaluate(model, x, y):
    err = compute_mse(model, x, y)

    u, w, w_x, N, M, Q = y

    print("\nEvaluation:\n-----------")
    print(f"  MSE u:     {err[0]}")
    print(f"  MSE w:     {err[1]}")
    print(f"  MSE w':    {err[2]}")
    print(f"  MSE N:     {err[3]}")
    print(f"  MSE M:     {err[4]}")
    print(f"  MSE Q:     {err[5]}")

    pred = model_fn(model, x)

    u_pred, w_pred, w_x_pred, N_pred, M_pred, Q_pred = pred

    fig, axs = plt.subplots(3, 2, figsize=(10, 12))

    axs[0, 0].plot(x, u_pred, 'r', label='PINN')
    axs[0, 0].plot(x, u, 'b', linestyle='--', label='data')
    axs[0, 0].set_xlabel(r'$x$')
    axs[0, 0].set_ylabel(r'$u$')
    axs[0, 0].grid()
    axs[0, 0].legend()

    axs[0, 1].plot(x, w_pred, 'r', label='PINN')
    axs[0, 1].plot(x, w, 'b', linestyle='--', label='data')
    axs[0, 1].set_xlabel(r'$x$')
    axs[0, 1].set_ylabel(r'$w$')
    axs[0, 1].grid()
    axs[0, 1].legend()

    axs[1, 0].plot(x, w_x_pred, 'r', label='PINN')
    axs[1, 0].plot(x, w_x, 'b', linestyle='--', label='data')
    axs[1, 0].set_xlabel(r'$x$')
    axs[1, 0].set_ylabel(r'$\partial_{x}w$')
    axs[1, 0].grid()
    axs[1, 0].legend()

    axs[1, 1].plot(x, N_pred, 'r', label='PINN')
    axs[1, 1].plot(x, N, 'b', linestyle='--', label='data')
    axs[1, 1].set_xlabel(r'$x$')
    axs[1, 1].set_ylabel(r'$N$')
    axs[1, 1].grid()
    axs[1, 1].legend()

    axs[2, 0].plot(x, M_pred, 'r', label='PINN')
    axs[2, 0].plot(x, M, 'b', linestyle='--', label='data')
    axs[2, 0].set_xlabel(r'$x$')
    axs[2, 0].set_ylabel(r'$M$')
    axs[2, 0].grid()
    axs[2, 0].legend()

    axs[2, 1].plot(x, Q_pred, 'r', label='PINN')
    axs[2, 1].plot(x, Q, 'b', linestyle='--', label='data')
    axs[2, 1].set_xlabel(r'$x$')
    axs[2, 1].set_ylabel(r'$Q$')
    axs[2, 1].grid()
    axs[2, 1].legend()

    plt.tight_layout()
    plt.show()