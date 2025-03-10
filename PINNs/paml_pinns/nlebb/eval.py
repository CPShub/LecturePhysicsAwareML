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

    u, u_x, u_xx, w, w_x, w_xx, w_xxx, w_xxxx = y

    print("\nEvaluation:\n-----------")
    print(f"  MSE u:     {err[0]}")
    print(f"  MSE u_x:   {err[1]}")
    print(f"  MSE u_xx:  {err[2]}")
    print(f"  MSE w:     {err[3]}")
    print(f"  MSE w':    {err[4]}")
    print(f"  MSE w'':   {err[5]}")
    print(f"  MSE w''':  {err[6]}")
    print(f"  MSE w'''': {err[7]}")

    pred = model_fn(model, x)

    u_pred, u_x_pred, u_xx_pred, w_pred, w_x_pred, w_xx_pred, w_xxx_pred, w_xxxx_pred = pred

    fig, axs = plt.subplots(4, 2, figsize=(10, 16))

    axs[0, 0].plot(x, u_pred, 'r', label='PINN')
    axs[0, 0].plot(x, u, 'b', linestyle='--', label='analytical')
    axs[0, 0].set_xlabel(r'$x$')
    axs[0, 0].set_ylabel(r'$u$')
    axs[0, 0].grid()
    axs[0, 0].legend()

    axs[0, 1].plot(x, u_x_pred, 'r', label='PINN')
    axs[0, 1].plot(x, u_x, 'b', linestyle='--', label='analytical')
    axs[0, 1].set_xlabel(r'$x$')
    axs[0, 1].set_ylabel(r'$\partial_{x}u$')
    axs[0, 1].grid()
    axs[0, 1].legend()

    axs[1, 0].plot(x, u_xx_pred, 'r', label='PINN')
    axs[1, 0].plot(x, u_xx, 'b', linestyle='--', label='analytical')
    axs[1, 0].set_xlabel(r'$x$')
    axs[1, 0].set_ylabel(r'$\partial_{xx}u$')
    axs[1, 0].grid()
    axs[1, 0].legend()

    axs[1, 1].plot(x, w_pred, 'r', label='PINN')
    axs[1, 1].plot(x, w, 'b', linestyle='--', label='analytical')
    axs[1, 1].set_xlabel(r'$x$')
    axs[1, 1].set_ylabel(r'$w$')
    axs[1, 1].grid()
    axs[1, 1].legend()

    axs[2, 0].plot(x, w_x_pred, 'r', label='PINN')
    axs[2, 0].plot(x, w_x, 'b', linestyle='--', label='analytical')
    axs[2, 0].set_xlabel(r'$x$')
    axs[2, 0].set_ylabel(r'$\partial_{x}w$')
    axs[2, 0].grid()
    axs[2, 0].legend()

    axs[2, 1].plot(x, w_xx_pred, 'r', label='PINN')
    axs[2, 1].plot(x, w_xx, 'b', linestyle='--', label='analytical')
    axs[2, 1].set_xlabel(r'$x$')
    axs[2, 1].set_ylabel(r'$\partial_{xx}w$')
    axs[2, 1].grid()
    axs[2, 1].legend()

    axs[3, 0].plot(x, w_xxx_pred, 'r', label='PINN')
    axs[3, 0].plot(x, w_xxx, 'b', linestyle='--', label='analytical')
    axs[3, 0].set_xlabel(r'$x$')
    axs[3, 0].set_ylabel(r'$\partial_{xxx}w$')
    axs[3, 0].grid()
    axs[3, 0].legend()

    axs[3, 1].plot(x, w_xxxx_pred, 'r', label='PINN')
    axs[3, 1].plot(x, w_xxxx, 'b', linestyle='--', label='analytical')
    axs[3, 1].set_xlabel(r'$x$')
    axs[3, 1].set_ylabel(r'$\partial_{xxxx}w$')
    axs[3, 1].grid()
    axs[3, 1].legend()

    plt.tight_layout()
    plt.show()