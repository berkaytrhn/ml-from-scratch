
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ml_lib.linear import LinearRegression
from ml_lib.linear import LogisticRegression
from ml_lib.metrics import MSELoss, BCELoss, CrossEntropyLoss
from ml_lib.preprocessing import StandardScaler, OneHotEncoder
from ml_lib.utilities import Sigmoid, Softmax
from ml_lib.neighbours import KNNRegressor, KNNClassifier
from ml_lib.utilities import euclidean_distance, manhattan_distance
from ml_lib.utilities import L1Regularization, L2Regularization


from ml_lib.cluster import KMeans as KM
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans



def test_linear_regression():
    # Testing model on california housing dataset from sklearn
    california_housing = fetch_california_housing()

    X = california_housing["data"]
    y = california_housing["target"]
    y = y.reshape(y.shape[0], 1)
    print(X.shape, y.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # standarization for overcome overflow
    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    
    
    l2 = L2Regularization(_lambda=1e-4)
    
    loss = MSELoss()
    lr = LinearRegression(loss, l2, 0.0001, 10000)

    lr.fit(X_train, y_train)
    
    preds = lr.transform(X_test)
    test_loss = loss(y_test, preds)
    print(f"Test Loss: '{test_loss}'")
    

def test_logistic_regression():
    # Testing logictic regression model on ...
    
    iris_dataset = load_iris()
    X = iris_dataset["data"]
    y = iris_dataset["target"]
    
    # breast_cancer = load_breast_cancer()
    # X = breast_cancer["data"]
    # y = breast_cancer["target"]
    
    
    y = y.reshape(y.shape[0], 1)
    print(X.shape, y.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    
    encoder = OneHotEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.fit_transform(y_test)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    loss =  CrossEntropyLoss() # BCELoss()
    activation = Softmax() # Sigmoid() 
    l2 = L2Regularization(_lambda=1e-6)
    
    
    log_reg = LogisticRegression(
        loss=loss,
        activation=activation,
        regularization=l2,
        learning_rate=0.1,
        epochs=10000
    )
    
    
    log_reg.fit(X_train, y_train)
    
    preds = log_reg.transform(X_test)
    
    # TODO: implement metrics.py -> acc, prec, recall and f1 for both binary and multiclass clf
    if isinstance(activation, Softmax):
        y_true_labels = np.argmax(y_test, axis=1)
        y_pred_labels = np.argmax(preds, axis=1)
    else:
        y_true_labels = y_test
        y_pred_labels = (preds>=0.5)
    
    
    test_acc = accuracy_score(y_true_labels, y_pred_labels)*100
    precision = precision_score(y_true_labels, y_pred_labels, average="weighted")
    recall = recall_score(y_true_labels, y_pred_labels, average="weighted")
    f1 = f1_score(y_true_labels, y_pred_labels, average="weighted")

    print(f"Test Acc: '{test_acc:.4f}%', Precision: '{precision:.4f}', Recall: '{recall:.4f}', F1 Score: '{f1:.4f}'")

def test_knn_classifier():
    breast_cancer = load_breast_cancer()

    X = breast_cancer["data"]
    y = breast_cancer["target"]

    y = y.reshape(y.shape[0], 1)
    print("Shapes: ", X.shape, y.shape)
    
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    clf = KNNClassifier(k=3, distance=manhattan_distance)
    
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


def test_kmeans_cluster():
    # Implementation
    k=8
    km = KM(euclidean_distance, k=k)
    X, y = make_blobs(n_samples=5000, centers=8, n_features=2)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    km.fit(X)
    preds = km.transform(X)
    
    
    # Sklearn
    km_sk = KMeans(n_clusters=k, init='random', max_iter=300)
    preds_sk = km_sk.fit_predict(X)
    print(preds_sk)
    
    
    draw_clusters(X, y, preds,preds_sk, k)
    
    
    
    
    
def draw_clusters(X, real, preds, preds_sklearn, k):
    # assumes two dim data
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:,0], X[:, 1], c=real, alpha=0.7)
    plt.title(f"Real, k: {len(np.unique(real))}")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:,0], X[:, 1], c=preds, alpha=0.7)
    plt.title(f"Preds, k: {k}")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:,0], X[:, 1], c=preds_sklearn, alpha=0.7)
    plt.title(f"Preds(Sklearn), k: {k}")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    """Main"""
    np.random.seed(42)
    
    # test_linear_regression()
    # test_logistic_regression()
    # test_knn_classifier()
    # test_knn_regressor()
    test_kmeans_cluster()
    


if __name__ == "__main__":
    main()
    