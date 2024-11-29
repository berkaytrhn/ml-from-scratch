import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple

class Loss(ABC):

    @abstractmethod
    def __call__(
        self, 
        y_true:np.ndarray, 
        y_pred:np.ndarray
    ) -> np.float64:
        """Loss Calculation Abstract Method"""

    @abstractmethod
    def backward(
        self, 
        y_true:np.ndarray, 
        y_pred:np.ndarray
    ) -> Tuple[np.float64, np.float64]:
        """Loss Backward Abstract Method"""



class MSELoss(Loss):
    """
    Basic Implementation of MSE Loss
    """

    def __call__(
        self,
        y_true:np.ndarray,
        y_pred:np.ndarray
    ) -> np.float64:
        return np.mean(np.square(y_true-y_pred))

    def backward(
        self, 
        y_true:np.ndarray, 
        y_pred:np.ndarray
    ) -> Tuple[np.float64, np.float64]:
        """
        Derivative of Loss with respect to predictions(y', intermediate value) 
        Since calculating:
            -> dw = -(2 / n_samples) * np.dot(X.T, (y_true - y_pred))
            and 
            -> dw = np.dot(X.T, (-(2 / n_samples) * (y_true - y_pred)))
            identical operations;
            We do not need X values in here, we can simply calculate intermediate 
            derivative value and return it to be used as exactly done in below operations.
        return:
            (dw_intermediate, db)
        """
        n_samples = len(y_true)
        factor = -(2 / n_samples)
        residual = (y_true - y_pred)
        return (factor * residual, factor * np.sum(residual))
        