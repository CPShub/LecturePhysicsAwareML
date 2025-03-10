from typing import Tuple, Self

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

import paramax
import equinox as eqx

from ..nn import FFNN


# === -------------------------------------------------------------------- === #
# PINN
# === -------------------------------------------------------------------- === #

class PINN(eqx.Module):
    """ A PINN for the linear elastic Euler-Bernoulli beam."""
    nn: FFNN
    L0: float
    f0: float
    q0: float
    u_bc_coords: paramax.NonTrainable | None
    u_bc_values: paramax.NonTrainable | None
    u_x_bc_coords: paramax.NonTrainable | None
    u_x_bc_values: paramax.NonTrainable | None
    w_bc_coords: paramax.NonTrainable | None
    w_bc_values: paramax.NonTrainable | None
    w_x_bc_coords: paramax.NonTrainable | None
    w_x_bc_values: paramax.NonTrainable | None
    w_xx_bc_coords: paramax.NonTrainable | None
    w_xx_bc_values: paramax.NonTrainable | None
    w_xxx_bc_coords: paramax.NonTrainable | None
    w_xxx_bc_values: paramax.NonTrainable | None

    def __init__(
        self,
        L0: float,
        f0: float,
        q0: float,
        bc: dict[str, Array | None],
        *,
        key: PRNGKeyArray
    ):
        self.nn = FFNN(
            in_features=1,
            hidden_features=[8, 8],
            out_features=2,
            activations=[jax.nn.tanh, jax.nn.tanh],
            final_activation=lambda x: x,
            key=key
        )
        self.L0 = L0
        self.f0 = f0
        self.q0 = q0

        if bc["u_bc_coords"] is None:
            self.u_bc_coords = None
            self.u_bc_values = None
        else:
            self.u_bc_coords = paramax.NonTrainable(bc["u_bc_coords"].reshape(-1, 1))
            self.u_bc_values = paramax.NonTrainable(bc["u_bc_values"].reshape(-1, 1))

        if bc["u_x_bc_coords"] is None:
            self.u_x_bc_coords = None
            self.u_x_bc_values = None
        else:
            self.u_x_bc_coords = paramax.NonTrainable(bc["u_x_bc_coords"].reshape(-1, 1))
            self.u_x_bc_values = paramax.NonTrainable(bc["u_x_bc_values"].reshape(-1, 1))

        if bc["w_bc_coords"] is None:
            self.w_bc_coords = None
            self.w_bc_values = None
        else:
            self.w_bc_coords = paramax.NonTrainable(bc["w_bc_coords"].reshape(-1, 1))
            self.w_bc_values = paramax.NonTrainable(bc["w_bc_values"].reshape(-1, 1))

        if bc["w_x_bc_coords"] is None:
            self.w_x_bc_coords = None
            self.w_x_bc_values = None
        else:
            self.w_x_bc_coords = paramax.NonTrainable(bc["w_x_bc_coords"].reshape(-1, 1))
            self.w_x_bc_values = paramax.NonTrainable(bc["w_x_bc_values"].reshape(-1, 1))

        if bc["w_xx_bc_coords"] is None:
            self.w_xx_bc_coords = None
            self.w_xx_bc_values = None
        else:
            self.w_xx_bc_coords = paramax.NonTrainable(bc["w_xx_bc_coords"].reshape(-1, 1))
            self.w_xx_bc_values = paramax.NonTrainable(bc["w_xx_bc_values"].reshape(-1, 1))

        if bc["w_xxx_bc_coords"] is None:
            self.w_xxx_bc_coords = None
            self.w_xxx_bc_values = None
        else:
            self.w_xxx_bc_coords = paramax.NonTrainable(bc["w_xxx_bc_coords"].reshape(-1, 1))
            self.w_xxx_bc_values = paramax.NonTrainable(bc["w_xxx_bc_values"].reshape(-1, 1))

    def __call__(self, x: Array) -> Array:
        u = self.u(self, x)
        u_x = self.u_x(self, x)
        u_xx = self.u_xx(self, x)
        w = self.w(self, x)
        w_x = self.w_x(self, x)
        w_xx = self.w_xx(self, x)
        w_xxx = self.w_xxx(self, x)
        w_xxxx = self.w_xxxx(self, x)

        return u, u_x, u_xx, w, w_x, w_xx, w_xxx, w_xxxx

    def forward(self, x: Array) -> Tuple[Array, ...]:
        x = x / self.L0
        output = self.nn(x)
        u = output[0]
        w = output[1]
        return u, w

    def u(self, model: Self, x: Array) -> Array:
        u, _ = model.forward(x)
        return u
    
    def u_x(self, model: Self,  x: Array) -> Array:
        return jax.jacfwd(self.u, argnums=1)(model, x)[0]

    def u_xx(self, model: Self, x: Array) -> Array:
        return jax.jacfwd(self.u_x, argnums=1)(model, x)[0]

    def w(self, model: Self, x: Array) -> Array:
        _, w = model.forward(x)
        return w

    def w_x(self, model: Self,  x: Array) -> Array:
        return jax.jacfwd(self.w, argnums=1)(model, x)[0]

    def w_xx(self, model: Self, x: Array) -> Array:
        return jax.jacfwd(self.w_x, argnums=1)(model, x)[0]

    def w_xxx(self, model: Self, x: Array) -> Array:
        return jax.jacfwd(self.w_xx, argnums=1)(model, x)[0]

    def w_xxxx(self, model: Self, x: Array) -> Array:
        return jax.jacfwd(self.w_xxx, argnums=1)(model, x)[0]

    def res(self, model: Self, x: Array):
        u_x = self.u_x(model, x)
        u_xx = self.u_xx(model, x)
        w_x = self.w_x(model, x)
        w_xx = self.w_xx(model, x)
        w_xxxx = self.w_xxxx(model, x)
        
        ru = u_xx + 0.5 * w_x**2 + self.f0
        rw = - (u_xx * w_x + u_x * w_xx + 1.5 * w_x**2 * w_xx) +  w_xxxx - self.q0

        return ru, rw

    def losses(self, model, x):
        u_pred_fun = jax.vmap(self.u, (None, 0))
        u_x_pred_fun = jax.vmap(self.u_x, (None, 0))
        w_pred_fun = jax.vmap(self.w, (None, 0))
        w_x_pred_fun = jax.vmap(self.w_x, (None, 0))
        w_xx_pred_fun = jax.vmap(self.w_xx, (None, 0))
        w_xxx_pred_fun = jax.vmap(self.w_xxx, (None, 0))

        res_fun = jax.vmap(self.res, (None, 0))

        if self.u_bc_coords is None:
            u_bc_loss = jnp.array(0.)
        else:
            u_bc_pred = w_pred_fun(model, self.u_bc_coords)
            u_bc_loss = jnp.mean((u_bc_pred - self.u_bc_values)**2)

        if self.u_x_bc_coords is None:
            u_x_bc_loss = jnp.array(0.)
        else:
            u_x_bc_pred = w_pred_fun(model, self.u_x_bc_coords)
            u_x_bc_loss = jnp.mean((u_x_bc_pred - self.u_x_bc_values)**2)
        
        if self.w_bc_coords is None:
            w_bc_loss = jnp.array(0.)
        else:
            w_bc_pred = w_pred_fun(model, self.w_bc_coords)
            w_bc_loss = jnp.mean((w_bc_pred - self.w_bc_values)**2)

        if self.w_x_bc_coords is None:
            w_x_bc_loss = jnp.array(0.)
        else:
            w_x_bc_pred = w_x_pred_fun(model, self.w_x_bc_coords)
            w_x_bc_loss = jnp.mean((w_x_bc_pred - self.w_x_bc_values)**2)

        if self.w_xx_bc_coords is None:
            w_xx_bc_loss = jnp.array(0.)
        else:
            w_xx_bc_pred = w_xx_pred_fun(model, self.w_xx_bc_coords)
            w_xx_bc_loss = jnp.mean((w_xx_bc_pred - self.w_xx_bc_values)**2)

        if self.w_xxx_bc_coords is None:
            w_xxx_bc_loss = jnp.array(0.)
        else:
            w_xxx_bc_pred = w_xxx_pred_fun(model, self.w_xxx_bc_coords)
            w_xxx_bc_loss = jnp.mean((w_xxx_bc_pred - self.w_xxx_bc_values)**2)

        ru_pred, rw_pred = res_fun(model, x)
        ru_loss = jnp.mean(ru_pred**2)
        rw_loss = jnp.mean(rw_pred**2)

        loss_dict = {
            "u_bc": u_bc_loss,
            "u_x_bc": u_x_bc_loss,
            "w_bc": w_bc_loss,
            "w_x_bc": w_x_bc_loss,
            "w_xx_bc": w_xx_bc_loss,
            "w_xxx_bc": w_xxx_bc_loss,
            "ru": ru_loss,
            "rw": rw_loss
        }
        return loss_dict

    def loss(self, model, weights, x):
        losses = self.losses(model, x)
        weighted_losses = jax.tree.map(lambda x, y: x * y, losses, weights)
        loss = jax.tree.reduce(lambda x, y: x + y, weighted_losses)
        return loss