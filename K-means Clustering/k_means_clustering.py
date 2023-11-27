import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from scipy.spatial.distance import cdist

iris = load_iris()
X = iris.data

# 1-Select Elbow point
def find_optimal_k(X, max_k=10):
    inertia_values = []
    
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia_values.append(kmeans.inertia_)
    
    return inertia_values

# 2-Select iteration number
n_iterations = 20

def k_means_custom(X, K):
    # 3- Select k different random centroids
    centroids = X[np.random.choice(X.shape[0], K, replace=False)]
    
    while True:
        # 4- Calculation of distances
        distances = cdist(X, centroids)
        # 5- Group based on min distance
        clust_index = np.argmin(distances, axis=1)
        new_centroids = np.array([X[clust_index == k].mean(axis=0) for k in range(K)])
        
        # 6- any centroid to move
        if np.all(new_centroids == centroids): # 7- terminate for that iteration
            break
        #8-change centroids to the mean
        centroids = new_centroids
    
    return centroids, clust_index

# Plot the elbow curve
def plot_elbow_curve(inertia_values):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(inertia_values) + 1), inertia_values, marker='o')
    plt.title('Elbow Method For Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Sum of Squared Distances (Inertia)')
    plt.show()


inertia_values = find_optimal_k(X)
plot_elbow_curve(inertia_values)

# Find the index of the elbow point using the second derivative
second_derivative = np.gradient(np.gradient(inertia_values))
elbow_point_index = np.argmax(second_derivative[:-1] > second_derivative[1:]) + 2
optimal_k = elbow_point_index

print("Optimal K (Elbow Point):", optimal_k)


best_inertia = float('inf')
best_centroids = None
best_clust_index = None

# 9- Iterate to find best clustering
for _ in range(n_iterations):
    centroids, clust_index = k_means_custom(X, K=optimal_k)
    
    current_inertia = np.sum((X - centroids[clust_index]) ** 2)
    
    if current_inertia < best_inertia:
        best_inertia = current_inertia
        best_centroids = centroids
        best_clust_index = clust_index

# Plot the final clusters of the best fit
plt.scatter(X[:, 0], X[:, 1], c=best_clust_index, cmap='viridis', marker='o', label='Data Points')
plt.scatter(best_centroids[:, 0], best_centroids[:, 1], marker='X', s=200, label='Centroids', color='red')
plt.legend()
plt.show()