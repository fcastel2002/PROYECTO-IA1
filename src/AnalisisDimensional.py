import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

class AnalisisDimensional:
    def __init__(self, csv_path='base_datos_std.csv'):
        try:
            self.datos = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"Error: El archivo '{csv_path}' no existe.")
            self.datos = None
        self.pca = None

    def graficar_2d(self, X_pca, y_encoded, etiquetas):
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, cmap='viridis')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.title('Análisis PCA 2D')
        
        # Crear la leyenda personalizada
        from matplotlib.lines import Line2D
        colores = scatter.cmap(scatter.norm(range(len(etiquetas))))
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=etiquetas[i],
                                  markerfacecolor=colores[i], markersize=10)
                           for i in range(len(etiquetas))]
        plt.legend(handles=legend_elements, title="Clases")
        
    def graficar_3d(self, X_pca, y_encoded, etiquetas):
        plt.subplot(122, projection='3d')
        ax = plt.gca()
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                           c=y_encoded, cmap='viridis')
        ax.set_xlabel('Componente Principal 1')
        ax.set_ylabel('Componente Principal 2')
        ax.set_zlabel('Componente Principal 3')
        ax.set_title('Análisis PCA 3D')
        
        # Crear la leyenda personalizada
        from matplotlib.lines import Line2D
        colores = scatter.cmap(scatter.norm(range(len(etiquetas))))
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=etiquetas[i],
                                  markerfacecolor=colores[i], markersize=10)
                           for i in range(len(etiquetas))]
        ax.legend(handles=legend_elements, title="Clases")

    def aplicar_pca(self, n_componentes=3):
        if self.datos is None:
            print("Error: No se pudo cargar el archivo de datos.")
            return None
        
        # Separar características de las etiquetas
        X = self.datos.iloc[:, :-1]  # Todas las columnas excepto la última (etiqueta)
        y = self.datos.iloc[:, -1]   # Última columna como etiqueta

        # Codificar las etiquetas
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        etiquetas = label_encoder.classes_

        # Aplicar PCA
        self.pca = PCA(n_components=n_componentes)
        X_pca = self.pca.fit_transform(X)

        # Mostrar la varianza explicada de cada componente
        varianza_explicada = self.pca.explained_variance_ratio_
        print("Varianza explicada por cada componente principal:")
        for i, varianza in enumerate(varianza_explicada):
            print(f"Componente {i+1}: {varianza * 100:.2f}%")

        # Mostrar el aporte de cada característica a cada componente principal
        componentes = self.pca.components_
        features = X.columns
        print("\nContribución de cada característica a cada componente principal:")
        for i, componente in enumerate(componentes):
            print(f"Componente {i+1}:")
            for feature, valor in zip(features, componente):
                print(f"  {feature}: {valor:.4f}")
        
        # Crear gráficos 2D y 3D
        plt.figure(figsize=(15, 6))
        self.graficar_2d(X_pca, y_encoded, etiquetas)
        self.graficar_3d(X_pca, y_encoded, etiquetas)
        
        plt.tight_layout()
        plt.show()