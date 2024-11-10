import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import silhouette_score
from Archivos import *

class KmeansClustering:
    def __init__(self, k = 4):
        self.k = k
        self.centroids = None
    
    @staticmethod
    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum((data_point - centroids) ** 2, axis = 1))
    
    def fit(self, X, iteraciones_max = 200):
        
        self.centroids = np.random.uniform(np.amin(X, axis = 0), np.amax(X, axis = 0), size = (self.k, X.shape[1]))
        
        for _ in range(iteraciones_max):
            y = []
            for data_point in X: 
                distances = KmeansClustering.euclidean_distance(data_point, self.centroids)
                cluster_num = np.argmin(distances)
                y.append(cluster_num)
            
            y = np.array(y)
            
            cluster_indices = []
            for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i))
                
            cluster_centers = []
            
            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indices], axis = 0)[0])
                    
            if np.max(self.centroids - cluster_centers) < 1e-6:
                break   
            else:
                self.centroids = np.array(cluster_centers)
        return y

    def plot_clusters(self, X, y):
        """Visualiza los clusters y centroides"""
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, 
                   label='Centroides')
        plt.title('Clusters K-means con Centroides')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.colorbar(scatter, label='Cluster')
        plt.show()

    def plot_clusters_3d(self, X, y):
        """Visualiza los clusters y centroides en 3D"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], 
                           c=y, cmap='viridis')
        ax.scatter(self.centroids[:, 0], 
                  self.centroids[:, 1], 
                  self.centroids[:, 2],
                  c='red', marker='x', s=200, linewidth=3, 
                  label='Centroides')
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('Clusters K-means 3D con Centroides')
        plt.legend()
        plt.colorbar(scatter, label='Cluster')
        plt.show()
    
    def evaluate_clustering(self, X, y):
        """Evalúa la calidad del clustering usando el coeficiente de silueta"""
        score = silhouette_score(X, y)
        return score

if __name__ == "__main__":
    # Cargar datos completos del CSV
    for _ in range(1):  
        df_resultados = pd.read_csv('resultados.csv')
        
        caracteristicas = ['Hu2','Hu3','Mean_B','Mean_G','Mean_R']
        # Extraer características para el clustering (todas excepto 'Nombre')
        X = df_resultados[caracteristicas].values
        
        # Crear y ajustar el modelo
        kmeans = KmeansClustering(k=4)
        clusters = kmeans.fit(X)
        
        # Añadir la columna de clusters al dataframe original
        df_resultados['Cluster'] = clusters
        
        # Guardar resultados del clustering
        df_resultados.to_csv('resultados_clustering.csv', index=False)
        
        # Guardar centroides en un archivo separado
        centroides_df = pd.DataFrame(
            kmeans.centroids,
            columns=caracteristicas
        )
        centroides_df.index.name = 'Cluster'
        centroides_df.to_csv('centroides.csv')
        
        # Análisis de clusters por tipo de vegetal
        print("\nDistribución de vegetales por cluster:")
        for cluster in sorted(df_resultados['Cluster'].unique()):
            print(f"\nCluster {cluster}:")
            cluster_counts = df_resultados[df_resultados['Cluster'] == cluster]['Nombre'].value_counts()
            for nombre, count in cluster_counts.items():
                print(f"{count} {nombre}")
            print("-" * 20)