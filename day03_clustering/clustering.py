#%%[markdown]
# <h1>Clustering</h1>

#%%
import pandas as pd
import numpy as np

#%%
# read in the CSV
df = pd.read_csv('day03_clustering/data/national-library-board-infopedia-articles.csv')

#%%
df.head(5)

#%%
df.summary

#%%
# Vectorize the document
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
vec_summary = vectorizer.fit_transform(df.summary)

#%% 
# K-Means
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(vec_summary)

#%%
# predict
clusters = kmeans.predict(vec_summary)

#%%
print('rec: %d' %len(df.summary))
print('len clusters: %d' %len(clusters))
print('sum of items in clusters: %d' %sum(clusters))

#%%
from matplotlib import pyplot as plt

df['summary'][clusters==0]

#%%
cluster_count = [ len(df['summary'][clusters==i]) for i in range(0, 10)]
plt.bar(range(0, 10), cluster_count, label='number of elements in cluster')
plt.legend()
plt.grid()
plt.xticks(range(0, 10))

#%%
cluster_size = 16
inertia = []
for i in range(2, cluster_size):
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(vec_summary)
    inertia.append(km.inertia_)
    sil_score = silhouette_score(vec_summary, km.labels_, metric='euclidean')
    print('Cluster: %d, silhouette score: %.9f' %(i, sil_score))

#%%
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(range(2, cluster_size), inertia, label='elbow plot', marker='o', markerfacecolor='r')
ax.set_xticks(range(2, cluster_size))
ax.grid()
plt.legend()

#%%
clusters25 = km.predict(vec_summary)
