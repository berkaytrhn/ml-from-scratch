from typing import Any
import numpy as np 
from model import BaseModel

def euclidean_distance(v1: np.ndarray, v2: np.ndarray):
    return np.sqrt(np.sum(np.square(v1-v2)))

class KNNClassifier(BaseModel):
    """
    Basic KNN Implementation for classification tasks 
    """
    
    def __init__(self, k, distance) -> None:
        self.k=k
        self.distance=distance
        
    def fit(self, x: np.ndarray) -> None:
        self.x_train = x
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        # use broadcasting instead of nested loops
        pass
    def fit_transform(self, data: np.ndarray):
        self.fit(data)
        return self.transform(data)
    
    
