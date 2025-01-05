from typing import Any
from ml_lib.model import BaseModel


class Kmeans(BaseModel):
    """
    Basic Kmeans Clustering Algorithm Implementation
    """
    
    def __init__(self, k=3, max_iteration=300, ):
        pass
    
    def _iterate(self,):
        # select random centroids
        
        # assign each data point to nearest centroid
        
        # set new centroids as mean of points in clusters
        
        # check how much change occured with new centroids with tolerance param
        pass
    
    def fit(self, x, y):
        pass
    
    def transform(self, x):
        pass
    