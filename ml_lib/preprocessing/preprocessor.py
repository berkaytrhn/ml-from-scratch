import numpy as np
from abc import ABC, abstractmethod
from typing import Union, List

class Preprocessor(ABC):
    
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def fit(self, data:Union[List, np.ndarray]) -> None:
        pass
    
    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        pass