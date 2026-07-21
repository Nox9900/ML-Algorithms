
import numpy as np


class LogisticRegressor:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_sample, n_features = X.shape

        #parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        #gradient descent
        for _ in range(self.n_iters):
            #approximate y with linear combination of weight and x, plus bias
            linear_model = np.dot(X, self.weights) + self.bias
            #apply the sigmoid fucntion
            y_predicted = self._sigmoid(linear_model)

            #gradient descent computation
            dw = (1/n_sample) * np.dot(X.T, (y_predicted - y))
            db = (1/n_sample) * np.sum(y_predicted - y)

            #update the parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr - db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)
    

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    

# testing

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)
    
    dt = datasets.load_breast_cancer()
    X, y = dt.data, dt.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    regressor = LogisticRegressor(learning_rate=0.0001, n_iters=1000)
    regressor.fit(X_train, y_train)
    preds = regressor.predict(X_test)

    print("LogisticRegression accuracy : ", accuracy(y_test, preds))


