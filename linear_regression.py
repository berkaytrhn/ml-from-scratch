from typing import Tuple
import numpy as np
from losses import Loss
from model import LinearModel

from tqdm import tqdm

class LinearRegression(LinearModel):
    """
    Basic Linear Regression Implementation
    """

    def __init__(
        self,
        loss: Loss,
        learning_rate: float = None,
        epochs: int = None,
    ) -> None:
        """
        Initializing Linear Regression Params.
        """
        super().__init__()
        
        self.vectorized = (learning_rate is None) and (epochs is None)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss = loss
        self.weights = None
        self.bias = None

    def _initialize_params(self, n_features) -> None:
        """Xavier Initialization"""
        shape = (n_features, 1)
        self.weights = np.random.randn(*shape) * np.sqrt(2.0 / (shape[0] + shape[1]))
        self.bias = 0  # zero init

    def _backward(self, x:np.ndarray, intermediates: Tuple) -> Tuple[np.ndarray, np.float64]:
        w_intermediate, db = intermediates
        dw = np.dot(x.T, w_intermediate)
        return (dw, db)

    def _optimize(self, x: np.ndarray, intermediates: Tuple[np.ndarray, np.float64]=None):
        """
        Performing Stochastic Gradient Descent
        x*w+b = y
        Loss = mean((y-y')**2)
        derivative(dL/dy) -> -(2/n_samples)*(y-y')
        derivative(dy/yw) -> x
        derivative(dL/dw) -> -(2/n_samples)*((y-y')*x) --> individual
        vectorized_derv(dL/dW) -> -(2/n_samples)*np.dot(X.T, (Y-Y'))
            ->  -(2/n_samples)*np.dot((n_features, n_samples), (n_samples, 1))
            -> -(2/n_samples)*(n_features, 1)
                --> same shape with weight vector
        """
        dw, db = self._backward(x, intermediates)
        
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        x.shape -> (n_samples, n_fetaures)
        y.shape -> (n_samples, 1)
        self.weights.shape -> (n_features, 1)
        self.bias -> single number

        (self.fit().transform()).shape(n_samples, 1) --> comparable with y
        """

        assert (
            x.shape[0] == y.shape[0]
        ), "Number of train(x) samples and number of ground truth values(y) must be equal!"

        n_samples, n_features = x.shape
        self._initialize_params(n_features)

        if self.vectorized:
            #TODO: Implement closed form solution
            return
        with tqdm(total=self.epochs, desc="Training Progress...") as pbar:
            for epoch in range(self.epochs):
                # Predictions
                y_pred = self.transform(x)

                # calculate loss
                loss = self.loss(y, y_pred)
                
                # loss backward
                intermediates = self.loss.backward(y, y_pred)
                
                # perform sgd with intermediate params
                self._optimize(x, intermediates)

                pbar.set_description(f"Epoch {epoch + 1}, Loss {loss:.4f}")
                
                # Update the progress bar
                pbar.update(1)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return np.dot(x, self.weights) + self.bias
