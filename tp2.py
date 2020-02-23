# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 19:08:00 2019

@author: Mariya Hristova and Yana Frandjelska
"""
### PACKAGES 

import numpy as np
from tp2_aux import images_as_matrix
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE 
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from pandas.plotting import parallel_coordinates
from sklearn.neighbors import  NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans 
from sklearn import metrics
from tp2_aux import report_clusters
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import scipy.cluster.hierarchy as shc

### READING THA DATA

data= images_as_matrix("C:/Users/User/Documents/Portugal/Classes/Machine Learning/Assignement 2")
labels = np.loadtxt("C:/Users/User/Documents/Portugal/Classes/Machine Learning/Assignement 2/labels.txt", delimiter = ",")
labels_true = np.array(labels[:,1]).reshape(-1)   # the cycle of the cell

### FEATURE EXTRACTION

## PCA feature extraction 

pca = PCA(n_components = 6)

pca.fit(data) 
features = range(pca.n_components)
transformed_pca = pca.transform(data)

transformed_pca = pd.DataFrame(transformed_pca)
transformed_pca.columns = ('PCA 1', 'PCA 2', 'PCA 3', 'PCA 4', 'PCA 5', 'PCA 6')

## TSNE feature extraction

TSNE = TSNE( method='exact', n_components = 6)
transformed_TSNE = TSNE.fit_transform(data)

transformed_TSNE = pd.DataFrame(transformed_TSNE)
transformed_TSNE.columns = ('TSNE 1', 'TSNE 2', 'TSNE 3', 'TSNE 4', 'TSNE 5', 'TSNE 6')

## ISOMAP feature extraction

isomap = Isomap(n_neighbors = 6, n_components= 6)
transformed_isomap = isomap.fit_transform(data)

transformed_isomap = pd.DataFrame(transformed_isomap)
transformed_isomap.columns = ('ISOMAP 1', 'ISOMAP 2', 'ISOMAP 3',
                              'ISOMAP 4', 'ISOMAP 5', 'ISOMAP 6')

## Combining all features into one data frame

features = pd.concat([ pd.DataFrame(transformed_pca),pd.DataFrame(transformed_TSNE), 
                      pd.DataFrame(transformed_isomap)], axis=1) 

### STANDARDIZING THE FEATURES

st = StandardScaler()   
st.fit(features)
features = st.transform(features)

### CHOOSING THE BEST FEATURES FOR CLUSTERING

## Correlation matrix analysis

# Combining the features and the true label into one data frame and renaming the features

features = pd.DataFrame(features)
df = pd.concat([ pd.DataFrame(labels_true), features], axis=1)
df.columns = ('Label','PCA 1', 'PCA 2', 'PCA 3', 'PCA 4', 'PCA 5', 'PCA 6', 
              'TSNE 1', 'TSNE 2', 'TSNE 3', 'TSNE 4', 'TSNE 5', 'TSNE 6', 
              'ISOMAP 1', 'ISOMAP 2', 'ISOMAP 3', 'ISOMAP 4', 'ISOMAP 5', 'ISOMAP 6')

# Correlation matrix

cor_matrix = pd.DataFrame(round(df.corr(),2))
cor_matrix.abs().sort_values(by=['Label'], ascending=False)

# Correlation matrix of the 6 features with the highest corelation with the label:
# ISOMAP 2, PCA 3, ISOMAP 6, PCA 1, PCA 2, ISOMAP 4

top_cor = cor_matrix.loc[['ISOMAP 2', 'PCA 3','TSNE 4', 'ISOMAP 6', 'PCA 1', 'PCA 2'],
                         ['ISOMAP 2', 'PCA 3','TSNE 4', 'ISOMAP 6', 'PCA 1', 'PCA 2']]

# As 'ISOMAP 2' is higly correlated with 'PCA 2' and 'PCA 3', respectivly 0.71 and -0.56 we exclude it from the selected features

## F- test feature selection 

# We calculate the f-value for the given set of features

f, prob = f_classif(df.loc[:,['PCA 1', 'PCA 2', 'PCA 3', 'PCA 4', 'PCA 5', 'PCA 6', 
              'TSNE 1', 'TSNE 2', 'TSNE 3', 'TSNE 4', 'TSNE 5', 'TSNE 6', 
              'ISOMAP 1', 'ISOMAP 2', 'ISOMAP 3', 'ISOMAP 4', 'ISOMAP 5', 'ISOMAP 6']], df.loc[:,['Label']])

labels_names = ['PCA 1', 'PCA 2', 'PCA 3', 'PCA 4', 'PCA 5', 'PCA 6', 
              'TSNE 1', 'TSNE 2', 'TSNE 3', 'TSNE 4', 'TSNE 5', 'TSNE 6', 
              'ISOMAP 1', 'ISOMAP 2', 'ISOMAP 3', 'ISOMAP 4', 'ISOMAP 5', 'ISOMAP 6']

labels_names = pd.DataFrame(labels_names)

f_df = pd.concat([ pd.DataFrame(f), labels_names], axis=1)
f_df = pd.DataFrame(f_df)
f_df.columns = ['F-value','Labels']

# The features arranged by F-values. We select the 6 features with the highest f-value:
# ISOMAP 2, PCA 3, PCA 2, PCA 1, ISOMAP 1, ISOMAP 4 

top_fv = f_df.sort_values(by=['F-value'], ascending=False)

### VISUALIZATION OF POSSIBLE FEATURES  

# Feature sets to choose from - cor accoridng to correlation analysis, f accoridng to f value, yam and mya our suggestions

df_top_feat_yam = df[['PCA 1', 'PCA 2', 'PCA 3','ISOMAP 1','ISOMAP 4', 'ISOMAP 6']]
df_top_feat_yam = np.array(df_top_feat_yam)

df_top_feat_mya = df[['ISOMAP 2', 'ISOMAP 3', 'ISOMAP 4', 'ISOMAP 6', 'PCA 1']]
df_top_feat_mya = np.array(df_top_feat_mya)

df_top_feat_cor = df[['PCA 3', 'PCA 2', 'ISOMAP 4', 'ISOMAP 6', 'PCA 1']]
df_top_feat_cor = np.array(df_top_feat_cor)

df_top_feat_f = df[['PCA 1', 'PCA 2', 'PCA 3', 'ISOMAP 1', 'ISOMAP 2',  'ISOMAP 4']]
df_top_feat_f = np.array(df_top_feat_f)

## Scatter Plots 

def plotBestFeatures(X_new, y):
    class_1 = y == 1
    class_2 = y == 2
    class_3 = y == 3
    mark_size = 30
    #print(X[:, 1])
    plt.scatter(X_new[class_1, 0], X_new[class_1, 1], c='b', s=mark_size, label=('Class 1: Cell starts to divide'))
    plt.scatter(X_new[class_2, 0], X_new[class_2, 1], c='r', s=mark_size, label=('Class 2: First part of the division'))
    plt.scatter(X_new[class_3, 0], X_new[class_3, 1], c='g', s=mark_size, label=('Class 3: Final stage of cell division'))
    plt.legend()

plotBestFeatures(df_top_feat_yam, labels_true)
plotBestFeatures(df_top_feat_mya, labels_true)
plotBestFeatures(df_top_feat_cor, labels_true)
plotBestFeatures(df_top_feat_f, labels_true)

## Parallel lines plot

parallel_coordinates(df[['Label','PCA 1', 'PCA 2', 'PCA 3','ISOMAP 1','ISOMAP 4', 'ISOMAP 6']], 'Label', color=('None','r','g','b'))
plt.savefig('parallel_yam.png', dpi=600,bbox_inches='tight')
plt.close()

parallel_coordinates(df[['Label','ISOMAP 2', 'ISOMAP 3', 'ISOMAP 4', 'ISOMAP 6', 'PCA 1']], 'Label', color=('None','r','g','b'))
plt.savefig('parallel_mya.png', dpi=600,bbox_inches='tight')
plt.close()

# Based on the plots we choose feature selectiom yam

### CLUSTERING

## DBSCAN 

df_top_feat = pd.DataFrame(df[['PCA 1', 'PCA 2', 'PCA 3','ISOMAP 1', 'ISOMAP 4', 'ISOMAP 6']])

# Optimal epsilon using K-nearest neighbours to calculate the distances

neigh = NearestNeighbors(n_neighbors=5)
nbrs = neigh.fit(df_top_feat)
distances, indices = nbrs.kneighbors(df_top_feat)

distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
plt.savefig('optimal_eps.png', dpi=600,bbox_inches='tight')
plt.show()
plt.close()

# Epsilon loop to determine the optimal value of epsilon in order to maximize or minimize the clustering measure parameters:

eps= np.arange (0.5, 1.9, 0.1)

for i in eps: 
    dbsc = DBSCAN(eps = i, min_samples = 5, metric = 'euclidean').fit(df_top_feat)
    labels_dbsc = dbsc.labels_
    print("\n")
    print("Epsilon : %f" % i)
    print("Silhouette Coefficient for DBSCAN: %0.3f"
      % metrics.silhouette_score(df_top_feat, labels_dbsc))
    n_clusters_ = len(set(labels_dbsc)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels_dbsc).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Adjusted Rand Index for DBSCAN: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels_dbsc))

# Based on the performance of the measures we choose epsilon to be 0.84

dbsc = DBSCAN(eps = 0.84, min_samples = 5, metric = 'euclidean').fit(df_top_feat)
labels_dbsc = dbsc.labels_
print(labels_dbsc)
n_clusters_ = len(set(labels_dbsc)) - (1 if -1 else 0)
print(n_clusters_)

## K-means

df_top_feat.as_matrix().astype("float32", copy = False)

Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(df_top_feat)
    Sum_of_squared_distances.append(km.inertia_)

# Plot of k-means
    
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.savefig('optimal_k.png', dpi=600,bbox_inches='tight')
plt.show()
plt.close()

# After we choose the optimal k, at k = 6, we implement the algorithm
k_means = KMeans(n_clusters=6)
k_means.fit(df_top_feat) 
labels_kmeans = k_means.predict(df_top_feat)

## Agglomerative Clustering

# Dentrogram to help us choose the number of cluster 

plt.figure(figsize=(10, 7))
plt.title("Customer Dendograms")
dend = shc.dendrogram(shc.linkage(df_top_feat, method='ward'))

# We choose to have 7 clusters

connectivity = kneighbors_graph(df_top_feat, n_neighbors=10, include_self=False)
clustering = AgglomerativeClustering(n_clusters = 7, connectivity=connectivity,
                                     linkage= 'ward').fit(df_top_feat)
clustering
AgglomerativeClustering()
labels_clust = clustering.labels_

### COMPARING ALGORITHMS

n_clusters_ = len(set(labels_dbsc)) - (1 if -1 in labels else 0)
n_noise_ = list(labels_dbsc).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

print("\n")

print("Adjusted Rand Index for DBSCAN: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels_dbsc))

print("Silhouette Coefficient for DBSCAN: %0.3f"
      % metrics.silhouette_score(df_top_feat, labels_dbsc))

print("\n")

print("Adjusted Rand Index for k-Means: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels_kmeans))

print("Silhouette Coefficient for k-Means: %0.3fs"
      % metrics.silhouette_score(df_top_feat, labels_kmeans))

print("\n")

print("Adjusted Rand Index for Agglomerative Clustering: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels_clust))

print("Silhouette Coefficient for Agglomerative Clustering: %0.3fs"
      % metrics.silhouette_score(df_top_feat, labels_clust))

### CLUSTER REPORTS

ids = np.array(range(0,563))
report_clusters(ids, labels_dbsc,  "example_labels_dbsc.html")
report_clusters(ids, labels_kmeans,  "example_labels_kmeans.html")
report_clusters(ids, labels_clust,  "example_labels_clust.html")
