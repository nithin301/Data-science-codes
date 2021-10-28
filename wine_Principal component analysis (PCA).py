# Importing libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from sklearn.preprocessing import scale 

#importing daataset
wn = pd.read_csv("wine.csv")
wn.describe() 
wn.head()

# Considering only numerical data 
wn_data = wn.iloc[:,1:]
# Standerdizing the numerical data 
wn_normal = scale(wn_data)

#PCA model building
pca = PCA()
pca_comp=pca.components_[0:]
pca_values = pca.fit_transform(wn_normal)# Data obtained after performing PCA
pca_values.shape


# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
# Cumulative variance 
cum_var = np.cumsum(np.round(var,decimals = 4)*100)
# as per cum_var first 3 variables consume 67% information,
#hence selecting first 3 variables as Principle Componentss

# Data obtained after performing PCA
new_wn = pd.DataFrame(pca_values[:,0:3])

# Variance plot for PCA components obtained 
plt.plot(cum_var,color="red")

# plot between PC1 and PC2 
x = np.array(pca_values[:,0])
y = np.array(pca_values[:,1])
plt.plot(x,y,"ro")
plt.plot(np.arange(178),x,"ro") # no where pca1 and pca2 are correlated


################### Clustering  ##########################


          ### USING K-MEANS ###
          
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 

## Using original data ##

k = list(range(2,15))
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(wn_data)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(wn_data.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,wn_data.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)


## Using the data obtained after performing PCA ##

k = list(range(2,15))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(new_wn)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(new_wn.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,new_wn.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)

## With the help of the scree plot it is observed that the same number of clusters are obtained
## using both original and PCA data

#perfoming Kmeans Clustering
kmeans = KMeans(n_clusters = 3).fit(new_wn)
kmeans.labels_
md=pd.Series(kmeans.labels_)  # converting numpy array into pandas series object 

new_wn['clust']=md # creating a  new column and assigning it to new column 
wn1=new_wn #naming dataframe obtained by Performing Kmeans as wn1
wn1.head()

wn1 = wn1.iloc[:,[3,0,1,2]] #arranging the sequence of columns
wn1_m=wn1.iloc[:,1:].groupby(wn1.clust).mean()


            ### USING HIERARCHICAL CLUSTERING ###            
            
from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch # for creating dendrogram 

## Using original data ##

z = linkage(wn_data, method="complete",metric="euclidean")

plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()


## Using the data obtained after performing PCA ##

z = linkage(new_wn, method="complete",metric="euclidean")

plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

## With the help of the DENDOGRAM it is observed that the same number of clusters are obtained
## using both original and PCA data

# Performing Hierarchical clustering #
from sklearn.cluster import	AgglomerativeClustering 
h_complete = AgglomerativeClustering(n_clusters=4,linkage='complete',affinity = "euclidean").fit(new_wn) 
h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)

new_wn['clust']=cluster_labels # creating a  new column and assigning it as new column 
wn2=new_wn #naming dataframe obtained by Performing Hierarchical clustering as wn1
wn2.head()

wn2 = wn2.iloc[:,[3,0,1,2]] #arranging the sequence of columns
wn2_m=wn2.iloc[:,1:].groupby(wn2.clust).mean()
