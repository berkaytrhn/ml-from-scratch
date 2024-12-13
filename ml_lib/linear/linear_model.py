from abc import abstractmethod
from ml_lib.model import BaseModel 
import numpy as np
from typing import Any

class LinearModel(BaseModel):
    @abstractmethod
    def _initialize_params(self, n_features) -> None:
        pass
    @abstractmethod
    def _backward(self, x:np.ndarray, intermediates: Any=None):
        pass
    @abstractmethod
    def _optimize(self, x: np.ndarray, intermediates: Any=None) -> None:
        pass