from abc import ABC, abstractmethod
import numpy as np

class Activation(ABC):
    """Base Class For Activations"""
    @abstractmethod
    def __call__(self, z:np.ndarray):
        pass


    @abstractmethod
    def backward(self):
        pass



class Sigmoid(Activation):
    """
    Basic Sigmoid Activation Implementation
    """

    def __init__(self) -> None:
        self.output_cache=None

    def __call__(self, z:np.ndarray):
        self.output_cache = 1 / (1 + np.exp(-z))
        return self.output_cache

    def backward(self):
        return self.output_cache * (1 - self.output_cache)
