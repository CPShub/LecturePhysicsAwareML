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
    L: float
    F: float
    q: float
    bc_case: int
    dataset_size: int
    non_dim: bool

    def __init__(
        self,
        EI: float,
        L: float,
        F: float,
        q: float,
        bc_case: int,
        dataset_size: int,
        non_dim: bool
    ):
        self.EI = EI
        self.L = L
        self.F = F
        self.q = q
        self.bc_case = bc_case
        self.dataset_size = dataset_size
        self.non_dim = non_dim


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
        EI, L, F, q = fun(bc_case)

        config = Config(
            EI=EI,
            L=L,
            F=F,
            q=q,
            bc_case=bc_case,
            dataset_size=1_000,
            non_dim=non_dim
        )
        return config
    return wrapper