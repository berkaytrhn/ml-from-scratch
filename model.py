from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class Model(ABC):
    """ Base Abstract Class(or interface) for Modelling """
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def _initialize_params(self, n_features) -> None:
        pass
    @abstractmethod
    def _backward(self, x:np.ndarray, intermediates: Any=None):
        pass
    @abstractmethod
    def _optimize(self, x: np.ndarray, intermediates: Any=None) -> None:
        pass
    @abstractmethod
    def fit(self, x: np.ndarray, y:np.ndarray) -> None:
        pass
    @abstractmethod
    def transform(self, x:np.ndarray) -> np.ndarray:
        pass
