from collections.abc import Callable
from dataclasses import dataclass
from typing import Tuple


# === -------------------------------------------------------------------- === #
# Config
# === -------------------------------------------------------------------- === #

@dataclass
class Config:
    """Configuration for data generation and PINN training."""
    EI: float
    EA: float
    L: float
    F: float
    f: float
    q: float
    bc_case: int
    dataset_size: int
    non_dim: bool


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
    ):
        EI, EA, L, F, q, f = fun(bc_case)

        config = Config(
            EI=EI,
            EA=EA,
            L=L,
            F=F,
            f=f,
            q=q,
            bc_case=bc_case,
            dataset_size=1_000,
            non_dim=non_dim
        )
        return config
    return wrapper