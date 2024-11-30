
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_breast_cancer

from linear_regression import LinearRegression
from logistic_regression import LogisticRegression
from losses import MSELoss, BCELoss
from utils import StandardScaler
from activations import Sigmoid




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


def main():
    """Main"""
    # test_linear_regression()
    test_logistic_regression()

if __name__ == "__main__":
    main()
    