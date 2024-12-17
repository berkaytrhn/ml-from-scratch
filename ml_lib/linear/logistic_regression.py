from tqdm import tqdm


from ml_lib.linear import LinearModel
from ml_lib.metrics import Loss
from ml_lib.utilities import Activation, Sigmoid, Softmax
from ml_lib.utilities import Regularization
import numpy as np


class LogisticRegression(LinearModel):
    """
    Basic Logistic Regression Class Based Implementation
    
    TODO: A bit dependent to its' building blocks such as Sigmoid, Softmax or SGD, should be generalized well
    TODO: 
    """

    def __init__(
        self,
        loss: Loss,
        activation: Activation,
        regularization: Regularization,
        learning_rate:float,
        epochs:int=None
    ) -> None:
        """
        Initializing Logistic Regression Params.
        """
        super().__init__()
        self.loss = loss
        self.activation=activation
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.regularization=regularization
        # Params
        self.weights = None
        self.bias = None


    def _initialize_params(self, n_features, n_classes) -> None:
        """Xavier Initialization"""
        shape = (n_features, n_classes) # prev was 1, generalized for multiclass clf
        self.weights = np.random.randn(*shape) * np.sqrt(2.0 / (shape[0] + shape[1]))
        self.bias = 0  # zero init
    
    def _backward(self, x: np.ndarray, intermediates: np.ndarray = None):
        """
        * Stochastic Gradient Descent Implementation
        - In every component that are used in forward propagation calculates its'
        own derivative with respect to previous meaningful component that is used 
        in computation like:
        -> BCELoss.backward() returns "(y_pred - y_true) / (y_pred * (1 - y_pred))"  which is dL/dy'
        -> Sigmoid.backward() returns "sigmoid_result * (1 - sigmoid_result)"        which is dy'/dz
        similarly, since self.transform() method is performing a part of overall forward propagation;
        it has also a backward method which performs computations like:
        -> , for dw
        -> , fow db 
        """
        # dL/dy' shape -> (n_samples, 1)
        dy_pred = intermediates[0]
        # dy'/dz shape -> (n_samples, 1)
        dz = intermediates[1]
        # regularization intermediates
        dreg = intermediates[2]


        # Since w.shape -> (n_features, 1), np.dot(x.T, <derivatives>) needed for result
        # dz/dw  = x and shape -> (n_samples, n_features)
       
        
        # no negative sign unlike linear regression since loss includes negativeness
        n_samples = x.shape[0]
        factor = (1/n_samples) 
        
        prev_derivatives = dy_pred * dz
        dw = factor * np.dot(x.T, prev_derivatives)
        # dz/db = 1
        db = factor * np.sum(prev_derivatives * 1)
        
        # update weights gradients with derivative of regularization term provided(might be zero if not)
        dw += dreg
        
        # dw = self.__clip_gradients(dw)
        # db = self.__clip_gradients(db)
        
        # print(db)
        
        return (dw, db)

    def _optimize(self, x: np.ndarray, intermediates: np.ndarray = None) -> None:
        
        dw, db = self._backward(x, intermediates)
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db



    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        x.shape -> (n_samples, n_fetaures)
        y.shape -> (n_samples, 1)
        self.weights.shape -> (n_features, 1)
        self.bias -> single number

        (self.fit().transform()).shape(n_samples, 1) --> comparable with y
        """
        assert(
            x.shape[0] == y.shape[0]
        ), "Incompatible shapes between x and y"
        
        n_samples, n_features = x.shape
        
        # init params
        """
        assumes y.shape as (batch_size, n_classes) one hot encoded 
        TODO: For cross entropy loss and softmax together, assumption is y.shape is one hot encoded, 
            should be generalized to be able to handle (batch_size, 1) shapes for multiclass clf
        """
        self._initialize_params(n_features, y.shape[1]) 
        
        with tqdm(total=self.epochs, desc="Training Progress...") as pbar:
            for epoch in range(self.epochs):
                # forward pass
                y_pred = self.transform(x)

                assert np.all(y_pred >= 0) and np.all(y_pred <= 1), "y_pred values are out of bounds!"
                
                
                regularize = self.regularization is not None
                regularization_term = self.regularization.forward(self.weights) if regularize else 0
                # calculating the loss
                loss = self.loss(y, y_pred) + regularization_term
                
                reg_grads = self.regularization.backward(self.weights) if regularize else 0
                # loss backward
                loss_intermediates: np.ndarray = self.loss.backward(y, y_pred)
                
                # TODO: Fix bug for following backprop of softmax activation
                # caches the activation value on __call__
                activation_intermediate = None
                if isinstance(self.activation, Softmax):  
                    # returns (N, C, C) jacobian matrix
                    activation_intermediate = self.activation.backward(loss_intermediates)
                    # performing optimnization
                    loss_intermediates=1
                
                else:
                    activation_intermediate = self.activation.backward()
                
                # performing optimnization
                self._optimize(x, (loss_intermediates, activation_intermediate, reg_grads))
                
                
                if isinstance(self.activation, Softmax):
                    y_true_labels = np.argmax(y, axis=1)
                    # print(y[:4], y_true_labels[:4])
                    y_pred_labels = np.argmax(y_pred, axis=1)
                    # print(y_pred[:5], y_pred_labels[:5])
                else:
                    y_true_labels = y
                    y_pred_labels = y_pred>=0.5
                train_acc = np.mean(y_true_labels==y_pred_labels)*100
                pbar.set_description(f"Epoch {epoch + 1}, Train Acc: {train_acc:.4f}%, Loss {loss:.4f}")
                
                # Update the progress bar
                pbar.update(1)
        
        
        
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        z = np.dot(x, self.weights) + self.bias
        
        # print(x.shape, self.weights.shape)
        # print("z: ",z.shape)
        # print(z[:5,:])
        a = self.activation(z)
        # print("a: ",a.shape)
        # print(a[:5,:])
        
        return a
        
        
        