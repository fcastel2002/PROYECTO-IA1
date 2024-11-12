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

        # Crear el gráfico 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(
            X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
            c=y_encoded, cmap='viridis', label=etiquetas
        )

        # Crear una leyenda
        legend_handles = scatter.legend_elements()[0]
        ax.legend(legend_handles, etiquetas, title="Etiquetas")

        ax.set_xlabel('Componente Principal 1')
        ax.set_ylabel('Componente Principal 2')
        ax.set_zlabel('Componente Principal 3')
        plt.show()