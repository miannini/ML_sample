############### 1st Part ############################
# Importing the libraries
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values


############### 2nd Part ############################
#fitting the Model (choose one by one and test - restart kernel each time)
#K-Means clustering
# Find optimal cluster number
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('the Elbow Method')
plt.xlabel('number of clusters')
plt.ylabel('WCSS')
plt.show()
#train the K-means clutering model
kmeans = KMeans(n_clusters=5,init='k-means++', max_iter=300, n_init=10, random_state=0)
y_cluster = kmeans.fit_predict(X)

#Hierarchical clustering
# Find optimal cluster number
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()
#train the Hierarchical clutering model
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage='ward')
y_cluster = hc.fit_predict(X)


############### 3rd Part ############################
#Visualizations - only in 2D spaces (using first 2 dimensions or columns here)
# Visualising the clusters
plt.scatter(X[y_cluster==0, 0], X[y_cluster==0, 1], s=100, c='red', label='cluster_1')
plt.scatter(X[y_cluster==1, 0], X[y_cluster==1, 1], s=100, c='blue', label='cluster_2')
plt.scatter(X[y_cluster==2, 0], X[y_cluster==2, 1], s=100, c='green', label='cluster_3')
plt.scatter(X[y_cluster==3, 0], X[y_cluster==3, 1], s=100, c='cyan', label='cluster_4')
plt.scatter(X[y_cluster==4, 0], X[y_cluster==4, 1], s=100, c='magenta', label='cluster_4')
#next line only for kmeans
#plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='yellow', label='Centroids')
#
plt.title('Cluster of clients')
plt.xlabel('Annual income (k$)')
plt.ylabel('Spending score (1-100)')
plt.legend()
plt.show()

####
# Visualising the clusters renamed
plt.scatter(X[y_cluster==0, 0], X[y_cluster==0, 1], s=100, c='red', label='Careful')
plt.scatter(X[y_cluster==1, 0], X[y_cluster==1, 1], s=100, c='blue', label='Standard')
plt.scatter(X[y_cluster==2, 0], X[y_cluster==2, 1], s=100, c='green', label='Target')
plt.scatter(X[y_cluster==3, 0], X[y_cluster==3, 1], s=100, c='cyan', label='careless')
plt.scatter(X[y_cluster==4, 0], X[y_cluster==4, 1], s=100, c='magenta', label='Sensible')
#next line only for kmeans
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='yellow', label='Centroids')
#
plt.title('Cluster of clients')
plt.xlabel('Annual income (k$)')
plt.ylabel('Spending score (1-100)')
plt.legend()
plt.show()