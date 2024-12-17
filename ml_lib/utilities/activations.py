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


class Softmax(Activation):
    """
    Basic Softmax Activation Implementation
    """
    def __init__(self):
        self.cache=None
    
    def __call__(self, z):
        exps = np.exp(z - np.max(z, axis=1, keepdims=True)) 
        
        self.cache = exps / np.sum(exps, axis=1, keepdims=True)
        
        # print("Softmax output shape:", self.cache.shape)
        
        return self.cache
        
    def backward(self, d_out: np.ndarray):
        """
        Since softmax depends on number_of_class values,
        we should calculate partial derivative wrt all individually in a loop
        
        d_out: Derivative of loss wrt softmax output -> (N,C) = (number_of_samples, number_of_classes)
        """
        N, C = self.cache.shape
        
        # for a case like receiving next derivatives and calculate  
        d_logits = np.zeros_like(d_out)
        
        # for a case like calculating immediate derivatives and later connect them
        # jacobians = np.zeros((N, C, C))
        
        # Loop in all datapoints
        for i in range(N):  
            
            # Get the softmax output for the i-th data point
            # Shape from (1,C) to (C, 1)  with reshape (-1, 1)
            softmax_output = self.cache[i].reshape(-1, 1)  
            
            # Computing the Jacobian matrix for softmax: "diag(softmax) - softmax * softmax^T"
            # Shape (C, C)
            jacobian = np.diagflat(softmax_output) - np.dot(softmax_output, softmax_output.T)  
            
            # Computing the gradient for the i-th data point for complete usage
            d_logits[i] = np.dot(jacobian, d_out[i])
            
            # Computing jacobian matrices for all samples for immediate calculation implementation
            # Shape : (N, C, C)
            # jacobians[i] = jacobian
        
        return d_logits
    