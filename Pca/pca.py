import numpy as np

class PCA :
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # mean centering
        self.mean = np.mean(X, axis=0)
        X = X -  self.mean

        # covariance, function needs samples as columns
        cov = np.cov(X.T)

        #eigenvalues, eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # eigenvectors v= [:, i] column vector, transpose for easier calculation
        # sort eigenvectors

        eigenvectors = eigenvectors.T
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[idx]

        # store frist n eigenvectors
        self.n_components = eigenvectors[0 : self.n_components]

    def transform(self, X):
        X= X -  self.mean
        return np.dot(X, self.n_components.T)


# testing
if  __name__ == "__main__":
    from sklearn import datasets
    import matplotlib.pyplot as plt

    data = datasets.load_iris()
    X, y =  data.data, data.target

    pca = PCA(2)
    pca.fit(X)
    X_projected = pca.transform(X)

    print(f"Shape of X : {X.shape}")
    print(f"Shape of transformed X : {X_projected.shape}")

    X1 =X_projected[:, 0]
    X2 =X_projected[:, 1]

    plt.scatter(X1, X2, c=y, edgecolors="none", alpha=0.8, cmap="viridis")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()

    # plt.savefig("pca.png", dpi=120)
    plt.show()