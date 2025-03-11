from functools import partial

import jax
from jaxtyping import Array

import equinox as eqx
import paramax
import optax 

from .utils import dataloader


def train(
    model: eqx.Module,
    x: Array,
    weights: dict[str, float],
    *,
    steps: int,
    batch_size: int = 32,
    learning_rate: float = 1e-3
):

    @partial(jax.jit, static_argnums=1)
    @jax.value_and_grad
    def compute_loss(params, static, weights, x):
        model = eqx.combine(params, static)
        model = paramax.unwrap(model)
        return model.loss(weights, x)

    @partial(jax.jit, static_argnums=1)
    def make_step(params, static, weights, x, opt_state):
        loss, grads = compute_loss(params, static, weights, x)
        updates, opt_state = optim.update(grads, opt_state)
        params = eqx.apply_updates(params, updates)
        return loss, params, opt_state

    params, static = eqx.partition(model, eqx.is_array)
    optim = optax.adam(learning_rate)
    opt_state = optim.init(params)

    iter_data = dataloader((x,), batch_size) # TODO: This is ugly. Fix the data loader

    loss = None
    for step, (x,) in zip(range(steps), iter_data):
        loss, params, opt_state = make_step(params, static, weights, x, opt_state)
        loss = loss.item()
        if step % 1000 == 0:
            print(f"Step: {step},\tLoss: {loss}")
    print(f"\nTraining finished, final loss={loss}")

    model = eqx.combine(params, static)

    return model