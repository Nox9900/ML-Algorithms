from collections import Counter
import numpy as np

def Euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN :
    def __init__(self, k: int=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        y_pred = [self._predict(X) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        #compute the distance x and all examples in the training set
        distance = [Euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indice = np.argsort(distance)[: self.k]  #sort by distance and return indeices of the first k neighbors
        k_neighbor_label = [self.y_train[i] for i in k_indice] # extract the labels pf the k nearest neighbor training samples 
        most_commom = Counter(k_neighbor_label).most_common(1)

        return most_commom[0][0]
    
if __name__ == "__main__" :
    from matplotlib.colors import ListedColormap
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
        
    iris = datasets.load_iris()
    X,y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    k = 3
    clf = KNN(k=k)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(f"KNN classification accuracy : {accuracy(y_test, preds):.4f} ")