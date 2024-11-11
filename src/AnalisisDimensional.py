from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.mplot3d import Axes3D

class AnalisisDimensional:
    def __init__(self, csv_path='base_datos.csv'):
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

        self.pca = PCA(n_components=n_componentes)
        componentes_principales = self.pca.fit_transform(X)
        
        print("PCA aplicado.")
        print("Varianza explicada por cada componente:", [f"{var:.5f}" for var in self.pca.explained_variance_ratio_])
        print("Aporte de cada feature a cada componente principal:")
        print(pd.DataFrame(self.pca.components_, columns=X.columns))
        
        return componentes_principales

    def graficar_componentes(self, componentes):
        if componentes is None:
            return

        etiquetas = self.datos.iloc[:, -1]
        label_encoder = LabelEncoder()
        etiquetas_numericas = label_encoder.fit_transform(etiquetas)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(componentes[:, 0], componentes[:, 1], componentes[:, 2], c=etiquetas_numericas, cmap='viridis')
        
        ax.set_xlabel('Componente Principal 1')
        ax.set_ylabel('Componente Principal 2')
        ax.set_zlabel('Componente Principal 3')
        ax.set_title('Análisis PCA de las Características')
        
        plt.colorbar(scatter, ticks=range(len(label_encoder.classes_)), label='Etiquetas')
        plt.show()
