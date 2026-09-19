import numpy as np


class BaseRegression:

    def __init__(self, learning_rate: float = 0.001, n_iter: int = 1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

        # weight and bias
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights, self.bias = np.zeros(n_features), 0

        # minimizing the loss, and finding thr correct weights and biases using Gradient Descent
        for _ in range(self.n_iter):
            y_predicted = self._approximation(X, self.weights, self.bias)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return self._predict(X, self.weights, self.bias)

    def _predict(self, X, w, b):
        raise NotImplementedError

    def _approximation(self, X, w, b):
        raise NotImplementedError

class LinearRegression(BaseRegression):

    def _approximation(self, X, w, b):
        return np.dot(X,w) + b

    def _predict(self, X, w, b):
        return np.dot(X,w) + b

class LogisticRegression(BaseRegression):

    def _approximation(self, X, w, b):
        linear_model = np.dot(X,w) + b
        return self._sigmoid(linear_model)

    def _predict(self, X, w, b):
        linear_model = np.dot(X, w) + b
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        return 1 / (np.exp(-x) + 1)



if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn import  datasets

    def r2_score(y_true, y_pred):
        corr_matrix = np.corrcoef(y_true, y_pred)
        corr = corr_matrix[0,1]
        return  corr ** 2

    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    X, y = datasets.make_regression(
        n_samples=100, n_features=1, noise=20, random_state=4
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    LinRegressor = LinearRegression(learning_rate=0.0001, n_iter=1000)
    LinRegressor.fit(X_train, y_train)
    predictions = LinRegressor.predict(X_test)

    accu = r2_score(y_test, predictions)
    print("Linear Regression accuracy : " + str(accu))


    data = datasets.load_breast_cancer()
    X,y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    LogRegression = LogisticRegression(learning_rate=0.0001, n_iter=1000)
    LogRegression.fit(X_train, y_train)
    predictions = LogRegression.predict(X_test)

    print("Logistic Regression classification accuracy : " + str(accuracy(y_test, predictions)))