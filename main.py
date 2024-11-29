
from sklearn.datasets import fetch_california_housing

from linear_regression import LinearRegression
from losses import MSELoss
import numpy as np


def main():
    """Main"""

    # Testing model on california housing dataset from sklearn
    california_housing = fetch_california_housing()

    X = california_housing["data"]
    y = california_housing["target"]

    # standarization for overcome overflow
    _mean = np.mean(X)
    _stddev= np.std(X)
    X = (X-_mean)/_stddev
    y = (y-_mean)/_stddev

    y = y.reshape(y.shape[0], 1)
    print(X.shape, y.shape)
    lr = LinearRegression(MSELoss(), 0.01, 1000)

    lr.fit(X, y)

if __name__ == "__main__":
    main()
    