from typing import Any
from ml_lib.model import BaseModel
import numpy as np
from tqdm import tqdm
from typing import Callable


class KMeans(BaseModel):
    """
    Basic Kmeans Clustering Algorithm Implementation
    """
    
    def __init__(
        self, 
        distance: Callable,
        k=3, 
        max_iteration=300,
        tolerance = 1e-4
    ) -> None:
        self.k=k
        self.max_iteration=max_iteration
        self.distance=distance
        self.tolerance=tolerance
    
    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """
        Returns assigned centroids with shape (n_samples,) 
            -> can be configured to be as (n_samples, 1) if needed
        """
        assigned_centroids=np.zeros((X.shape[0]))
        for i,data_point in enumerate(X):
            distances = np.array([self.distance(data_point, centroid) for centroid in self.centroids])
            min_index=np.argmin(distances)
            
            assigned_centroids[i] = min_index
        return assigned_centroids
    
    def _iterate(
        self, 
        X:np.ndarray
    ) -> None:
        with tqdm(total=self.max_iteration) as pbar:
            for _ in range(self.max_iteration):
                
                # TODO: perform efficiently with np.newaxis
                # TODO: consider rewrite with np.vectorize
                
                
                # assign each data point to nearest centroid
                assigned_centroids = self._assign_clusters(X)
                
                # set new centroids as mean of points in clusters
                new_centroids = np.array([np.mean(X[assigned_centroids==i], axis=0) for i in range(self.k)])


                # check how much change occured with new centroids with tolerance param
                if np.all(np.linalg.norm(new_centroids - self.centroids, axis=1) < self.tolerance):
                    # axis=0, collapse through rows
                    break
            
                
                # update if acceptable 
                self.centroids=new_centroids
                pbar.update(1)
            pass
    
    def fit(
        self, 
        X: np.ndarray
    ) -> None:
        n_samples = X.shape[0]
        # select random centroids
        # Replace false, no duplicates
        centroid_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[centroid_indices]
        
        # iterate for finding optimal centroids
        self._iterate(X)
    
    def transform(self, X:np.ndarray) -> np.ndarray:
        return self._assign_clusters(X)
    