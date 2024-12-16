import numpy as np
from typing import Union, List
from .preprocessor import Preprocessor


class OneHotEncoder(Preprocessor):
    
    def __init__(self):
        """
        Initialize clas fields to use later
        """
        self.unique_values=None
    
    
    def fit(self, data: Union[List, np.ndarray]) -> None:
        """
        Assumes provided 1D numpy array or a vanilla python list to be encoded
        """
        if isinstance(data, np.ndarray):
            data = np.array(data)
            
        self.unique_values = np.unique(data)
        
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Perform One Hot Encoded Generation
        """
        num_uniq = len(self.unique_values)
        num_samples = len(data)
        one_hots = np.zeros((num_samples, num_uniq))
        # buradan devam
        
        for uniq in self.unique_values:
            mask = np.equal(data, uniq)
            one_hots
            one_hots[:, uniq] = mask.reshape(-1)
        
        return one_hots
        
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)