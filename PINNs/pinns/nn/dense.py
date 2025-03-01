from typing import Literal, Optional, Union
from collections.abc import Callable

from jaxtyping import Array, PRNGKeyArray

import equinox as eqx

class Dense(eqx.Module):
    linear: eqx.nn.Linear
    activation: Callable

    def __init__(
        self,
        in_features: Union[int, Literal["scalar"]],
        out_features: Union[int, Literal["scalar"]], 
        activation: Callable,
        use_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray
    ):
        self.linear = eqx.nn.Linear(
            in_features,
            out_features,
            use_bias,
            dtype,
            key=key
        )
        self.activation = activation 

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        x = self.linear(x, key=key)
        return self.activation(x)