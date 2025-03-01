from functools import partial

import jax

import equinox as eqx
import paramax
import optax 

from .config import Config
from .utils import dataloader

def train(model, x, y, config: Config):

    @partial(jax.jit, static_argnums=1)
    @jax.value_and_grad
    def compute_loss(params, static, weights, x):
        model = eqx.combine(params, static)
        model = paramax.unwrap(model)
        return model.loss(model, weights, x)

    @partial(jax.jit, static_argnums=1)
    def make_step(params, static, weights, x, opt_state):
        loss, grads = compute_loss(params, static, weights, x)
        updates, opt_state = optim.update(grads, opt_state)
        params = eqx.apply_updates(params, updates)
        return loss, params, opt_state

    params, static = eqx.partition(model, eqx.is_array)
    optim = optax.adam(config.learning_rate)
    opt_state = optim.init(params)

    iter_data = dataloader((x, y), config.batch_size)

    loss = None
    for step, (x, y) in zip(range(config.steps), iter_data):
        loss, params, opt_state = make_step(params, static, config.weights, x, opt_state)
        loss = loss.item()
        if step % 1000 == 0:
            print(f"Step: {step},\tLoss: {loss}")
    print(f"\nTraining finished, final loss={loss}")

    model = eqx.combine(params, static)

    return model