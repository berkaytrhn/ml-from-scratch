
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_breast_cancer
import numpy as np

from linear_regression import LinearRegression
from logistic_regression import LogisticRegression
from losses import MSELoss, BCELoss
from utils import StandardScaler
from activations import Sigmoid
from knn_regressor import KNNRegressor
from knn_classifier import KNNClassifier
from distances import euclidean_distance



def test_linear_regression():
    # Testing model on california housing dataset from sklearn
    california_housing = fetch_california_housing()

    X = california_housing["data"]
    y = california_housing["target"]

    # standarization for overcome overflow
    scaler = StandardScaler()
    scaler.fit(X)
    
    y = y.reshape(y.shape[0], 1)
    print(X.shape, y.shape)
    
    
    X = scaler.transform(X)
    
    lr = LinearRegression(MSELoss(), 0.01, 10000)

    lr.fit(X, y)

def test_logistic_regression():
    # Testing logictic regression model on ...
    
    breast_cancer = load_breast_cancer()

    X = breast_cancer["data"]
    y = breast_cancer["target"]

    y = y.reshape(y.shape[0], 1)
    print("Shapes: ", X.shape, y.shape)
    
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    loss = BCELoss()
    activation = Sigmoid()
    log_reg = LogisticRegression(
        loss=loss,
        activation=activation,
        learning_rate=0.01,
        epochs=10000
    )
    
    
    log_reg.fit(X, y)

def test_knn_classifier():
    breast_cancer = load_breast_cancer()

    X = breast_cancer["data"]
    y = breast_cancer["target"]

    y = y.reshape(y.shape[0], 1)
    print("Shapes: ", X.shape, y.shape)
    
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    clf = KNNClassifier(k=3, distance=euclidean_distance)
    
    clf.fit(X, y)
    preds = clf.transform(X)
    
    acc = np.mean(preds == y)
    print(acc)
    
    
def test_knn_regressor():
    california_housing = fetch_california_housing()

    X = california_housing["data"]
    y = california_housing["target"]

    # standarization for overcome overflow
    scaler = StandardScaler()
    scaler.fit(X)
    
    y = y.reshape(y.shape[0], 1)
    print(X.shape, y.shape)
    
    
    X = scaler.transform(X)
    
    clf = KNNRegressor(k=3, distance=euclidean_distance)
    
    # for performance, takes too long otherwise
    number_of_samples = 1500
    clf.fit(X[:number_of_samples], y[:number_of_samples])
    preds = clf.transform(X[:number_of_samples])
    
    
    loss = np.mean(np.square(y[:number_of_samples]-preds))
    print(f"loss: {loss}")


def main():
    """Main"""
    # test_linear_regression()
    # test_logistic_regression()
    # test_knn_classifier()
    test_knn_regressor()

if __name__ == "__main__":
    main()
    