import numpy as np
from typing import Any
from abc import abstractmethod

class Regularization:
    
    
    @abstractmethod
    def __init__(self, _lambda:float=1e-3):
        self._lambda=_lambda
    
    @abstractmethod
    def forward(self, weights: np.ndarray):
        pass
        
    @abstractmethod
    def backward(self, weights: np.ndarray):
        pass    
    

class L1Regularization(Regularization):
    
    def __init__(self, _lambda):
        super().__init__(_lambda)
    
    def forward(self, weights: np.ndarray):
        return self._lambda * np.sum(np.abs(weights))
    
    def backward(self, weights: np.ndarray):
        return self._lambda * np.sign(weights)
    
    
class L2Regularization(Regularization):

    def __init__(self, _lambda):
        super().__init__(_lambda)
    
    def forward(self, weights: np.ndarray):
        return 0.5 * self._lambda * np.sum(weights ** 2)
    
    def backward(self, weights: np.ndarray):
        return self._lambda * weights