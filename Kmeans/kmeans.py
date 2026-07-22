import matplotlib.pyplot as plt
import numpy as np

def Euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

np.random.seed(42)

class KMeans:
    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

    def predict(self, X) : 
        self.X = X
        self.n_sample, self.n_features = X.shape
        
        # init
        random_sample_idxs = np.random.choice(self.n_sample, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        #optimize clusters
        for _ in range(self.max_iters):
            self.clusters = self._create_clusters(self.centroids) # assign samples to closest centroids (create clusters)
            
            if self.plot_steps :
                self.plot()

            #calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # check if clusters have changed
            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        return self._get_cluster_labels(self.clusters)
    
    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_sample) # each sample will get the lable of the vluster it was assigned to

        for clusters_idx, cluster in enumerate(clusters):
            for sample_index in cluster :
                labels[sample_index] = clusters_idx

        return labels
    
    def _create_clusters(self, centroids) :
        # assign the sample to the closest centroids to create clusters

        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroids_idx = self._closest_centroid(sample, centroids)
            clusters[centroids_idx].append(idx)

        return clusters
    
    def _closest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distance = [Euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distance)
        return closest_index
    
    def _get_centroids(self, clusters):
        # assing mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for clusters_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[clusters_idx] = cluster_mean

        return centroids
    
    def _is_converged(self, centroids_old, centroids) :
        # distance between each old and new centroids, for all centroids
        distance = [
            Euclidean_distance(centroids_old[i], centroids)
            for i in range(self.K)
        ]
        return sum(distance) == 0
    
    def plot(self, title="KMeans Algorithm"):
        fig , axes = plt.subplots(figsize=(12, 8))
        
        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            axes.scatter(*point)

        for point in self.centroids:
            axes.scatter(*point, marker='x', color="black", linewidth=2)

        plt.suptitle(title)
        plt.show()
        # plt.savefig("./kmean_plot.png", dpi=120)
        plt.close()

# testing

if __name__ == "__main__" :
    from sklearn.datasets import make_blobs

    X, y = make_blobs(
        centers=3, n_samples=500, n_features=2,
        shuffle=True, random_state=40
    )
    print("X dimension : ",X.shape)
    clusters = len(np.unique(y))
    print("Number of clusters : " ,clusters)

    k = KMeans(K=clusters, max_iters=150, plot_steps=True)
    y_pred = k.predict(X)
    k.plot()