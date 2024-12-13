import numpy as np
from ml_lib.model import BaseModel
from tqdm import tqdm

class KNNRegressor(BaseModel):
    """
    Basic KNN Implementation for regression tasks 
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
        nearest_neighbours=list()
        for sample in tqdm(data):
            nearest_neighbours.append(self._nearest_neighbour(sample))
        return np.array(nearest_neighbours).reshape(-1, 1)

    def _nearest_neighbour(self, sample: np.ndarray):
        distances = np.array([self.distance(data_point, sample) for data_point in self.x_train])
        
        # incides of closest k elements
        nearest_neighbours = np.argsort(distances)[:self.k]
        nearest_preds = self.y_train[nearest_neighbours]
        
        # majority voting using bincount
        """
        unlike the classifier, we do not perform majority vote;
        we perform simple mean among k neighbour values
        """
        majority_voted_value = np.mean(nearest_preds)
        return majority_voted_value