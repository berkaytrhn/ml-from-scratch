from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BaseModel(ABC):
    """ Base Abstract Class(or interface) for Modelling """
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def fit(self, x: np.ndarray, y:np.ndarray) -> None:
        pass
    @abstractmethod
    def transform(self, x:np.ndarray) -> np.ndarray:
        pass


