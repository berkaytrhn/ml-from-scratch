import numpy as np
from .preprocessor import Preprocessor


class StandardScaler(Preprocessor):
    """
    -> Basic standarization utility class
    -> Support for two dim array for now
    """
    
    def __init__(self) -> None:
        self.means=None
        self.stddevs=None
    
    def fit(self, data: np.ndarray):
        """ Calculating mean and stddev """
        # TODO: generalize support for multidim
        self.means = np.mean(data, axis=0)
        self.stddevs = np.std(data, axis=0)
    
    
    def transform(self, data:np.ndarray):
        """ Perform Scaling """
        assert (data.shape[1] == self.means.shape[0]), "Incompatible shapes"
        return (data - self.means)/self.stddevs
    
    def fit_transform(self, data:np.ndarray):
        self.fit(data)
        return self.transform(data)