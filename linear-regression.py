

import numpy as np

def r2_Score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0,1]
    return corr ** 2


class LinearRegression:
    def __init__(self, learning_rate=0.0001, n_iters=100) :
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            # gradient descent computation
            dw = ( 1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = ( 1/ n_samples) * np.sum(y_predicted - y)
            #update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X) :
        y_approximated = np.dot(X, self.weights) + self.bias 
        return y_approximated
    

#testing
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def mean_squared_error(y_true, y_pred):
        return np.mean(y_true - y_pred) ** 2
    
    
    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    regressor = LinearRegression(learning_rate=0.01, n_iters=1000)
    regressor.fit(X_train, y_train)
    preds = regressor.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    print(f"MSE : {mse:.4f}")

    accurancy = r2_Score(y_test, preds)
    print(f"Accurancy : {accurancy}")

    y_pred_line = regressor.predict(X)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(8,6))
    plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    plt.scatter(X_test, y_test, color=cmap(0.5), s=10)

    plt.plot(X, y_pred_line, color='black', linewidth=2, label="Prediction")
    plt.savefig("./linear-regression.png", dpi=120)
    # plt.show()
    plt.close()



