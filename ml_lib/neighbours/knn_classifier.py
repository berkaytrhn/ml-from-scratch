from typing import Any
import numpy as np 
from ml_lib.model import BaseModel



class KNNClassifier(BaseModel):
    """
    Basic KNN Implementation for classification tasks 
    """
    
    def __init__(self, k, distance) -> None:
        self.k=k
        self.distance=distance
    
    def fit_transform(self, data: np.ndarray):
        self.fit(data)
        return self.transform(data)
        
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x_train = x
        self.y_train = y
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        # TODO: Can be replaced with broadcasting and np.newaxis method
        nearest_neighbours = np.array([self._nearest_neighbour(sample) for sample in data])
        return nearest_neighbours.reshape(-1, 1)

    def _nearest_neighbour(self, sample: np.ndarray):
        distances = np.array([self.distance(data_point, sample) for data_point in self.x_train])
        
        # incides of closest k elements
        nearest_neighbours = np.argsort(distances)[:self.k]
        nearest_preds = self.y_train[nearest_neighbours]
        
        # majority voting using bincount
        """
        bincount receives 1d array so flattened and 
        since index of element corresponds to the element which
        we are counting, argmax returns the majority vote result
        """
        majority_vote = np.bincount(nearest_preds.flatten()).argmax()
        return majority_vote
    
