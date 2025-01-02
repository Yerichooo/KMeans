# KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

file_path = 'sales.xlsx'
df_sales = pd.read_excel(file_path, sheet_name='in')

columns_to_use = ['W0', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8']
data_for_clustering = df_sales[columns_to_use]

data_for_clustering = data_for_clustering.dropna()

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_for_clustering)

inertia = []
range_clusters = range(1, 11)
for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range_clusters, inertia, marker='o', linestyle='--')
plt.title('Jumlah Cluster Optimal')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(data_scaled)

df_sales['Cluster'] = clusters

pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.7)
plt.colorbar(scatter, label='Cluster')
plt.title('Visualisasi Hasil Clustering dengan K-Means')
plt.xlabel('Komponen 1')
plt.ylabel('Komponen 2')
plt.show()
