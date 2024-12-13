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
        For one layered implementation(linear regression for ex.)
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
        # TODO: consider if it is problem to return db from here
        return (factor * residual, factor * np.sum(residual))


class BCELoss(Loss):
    """
    Basic Implementation of Binary Cross Entropy Losss
    """
    def __clipping_preds(self, data, epsilon=1e-15):
        return np.clip(data, epsilon, 1 - epsilon)
    
    def __call__(
        self,
        y_true:np.ndarray,
        y_pred:np.ndarray
    ) -> np.float64:
        """BCE Loss Calculation """
        
        y_pred = self.__clipping_preds(y_pred)
        # print("neg_log_0 : ", neg_log_0.shape)
        # print("neg_log_1 : ", neg_log_1.shape)
        # print((neg_log_0+neg_log_1).shape)
        # print(-np.mean(neg_log_0+neg_log_1))
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    
        
    def backward(
        self, 
        y_true:np.ndarray, 
        y_pred:np.ndarray
    ) -> np.ndarray:
        """Loss Backward """

        y_pred = self.__clipping_preds(y_pred)

        # Calculate derivative of "loss" wrt "y_pred"
        return (y_pred - y_true) / (y_pred * (1 - y_pred))
    
