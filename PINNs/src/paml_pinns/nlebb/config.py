from collections.abc import Callable
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    """Configuration for data generation and PINN training."""
    E: float
    A: float
    II: float
    L: float
    F: float
    f: float
    q: float
    bc_case: int
    dataset_size: int


def get_config_decorator(fun: Callable[[int], Tuple[float, ...]]):
    """
    A decorator to expose only those setting to the user that are relevant
    for completing the tasks.
    """
    def wrapper(
        bc_case: int,
    ):
        E, A, II, L, F, q, f = fun(bc_case)

        config = Config(
            E=E,
            A=A,
            II=II,
            L=L,
            F=F,
            f=f,
            q=q,
            bc_case=bc_case,
            dataset_size=1_000,
        )
        return config
    return wrapper