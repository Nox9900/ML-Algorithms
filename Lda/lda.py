import numpy as np

class LDA:
    def __init__(self, n_components) :
        self.n_components = n_components
        self.linear_discriminants = None

    def fit(self, X, y) :
        n_features = X.shape[1]
        class_labels = np.unique(y)

        #within class scatter matrix
        # SW = sum((X_c -  mean_X_c) ^ 2)

        #between class scatter
        # SB = sum(n_c * (mean_X_c - mean_overall) ^ 2)

        mean_overall = np.mean(X, axis=0)
        SW = np.zeros((n_features, n_features))
        SB = np.zeros((n_features, n_features))

        for c in class_labels :
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            SW += (X_c - mean_c).T.dot((X_c - mean_c))
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            SB += n_c * mean_diff.dot(mean_diff.T)

        # determine SW^-1 & SB
        A = np.linalg.inv(SW).dot(SB)
        # get the eigenvalues and eigenvectors of W^-1 * SB
        eigenvalues, eigenvectors = np.linalg.eig(A)

        # eigenvector v = [:. 1] column vector, transpose for easier calculations
        # sort eigenvalues high to low

        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # store first m eigenvectors
        self.linear_discriminants = eigenvectors[0 :  self.n_components]

    def transform(self, X):
        return np.dot(X, self.linear_discriminants.T)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn import datasets

    data = datasets.load_iris()
    X,y = data.data, data.target

    #projection of the data onto the 2 primary linear discriminant
    lda = LDA(2)
    lda.fit(X, y)
    X_projected = lda.transform(X)

    print("Shape of X :", X.shape)
    print("Shape of transformed : ", X_projected.shape)

    X1, X2 = X_projected[:, 0], X_projected[:, 1]
    plt.scatter(X1, X2, c=y, edgecolors="none", alpha=0.8, cmap=plt.get_cmap("viridis", 3))
    plt.xlabel("Linear Discriminant 1")
    plt.ylabel("Linear Discriminant 2")
    plt.colorbar()
    plt.savefig("./lda.png", dpi=300)
    plt.show()