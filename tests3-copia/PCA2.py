import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
import os


class AnalisisPCA:

    def __init__(self, database):
        self.database = database
        self.pca = None
        self.feature_names = []

    def filtrar_caracteristicas_por_correlacion(self, df, umbral=0.2, metodo="mutual_info"):
        """
        Filtra características basado en correlación lineal y no lineal.

        Args:
            df: DataFrame con las características.
            umbral: valor mínimo de correlación promedio (default 0.01).
            metodo: método de correlación no lineal, "mutual_info" o "spearman" (default "mutual_info").

        Returns:
            DataFrame filtrado.
        """
        try:
            # Calcular matriz de correlación lineal
            corr_matrix = df.corr().abs()
            avg_corr = corr_matrix.mean()

            if metodo == "mutual_info":
                # Calcular mutual information (no lineal)
                mi_scores = []
                for col in df.columns:
                    # Mutual info entre cada característica y las demás
                    otros = df.drop(columns=[col])
                    mi = mutual_info_regression(otros, df[col])
                    mi_scores.append(mi.mean())
                mi_scores = pd.Series(mi_scores, index=df.columns)

            elif metodo == "spearman":
                # Calcular correlación de Spearman
                corr_spearman = df.corr(method="spearman").abs()
                avg_corr = corr_spearman.mean()

            else:
                raise ValueError(f"Método desconocido: {metodo}")

            # Combinar correlaciones
            avg_corr_combined = avg_corr + mi_scores if metodo == "mutual_info" else avg_corr
            avg_corr_combined /= avg_corr_combined.max()  # Normalizar

            # Filtrar características
            features_to_keep = avg_corr_combined[avg_corr_combined >= umbral].index
            print(
                f"\nCaracterísticas eliminadas por baja correlación (umbral={umbral}):")
            print(set(df.columns) - set(features_to_keep))

            return df[features_to_keep]

        except Exception as e:
            print("Error al filtrar características:", e)
            return df

    def calcular_importancia(self, pca, feature_names):
        """
        Calcula la importancia de las características basado en PCA.

        Args:
            pca: Objeto PCA ajustado
            feature_names: Nombres de las características

        Returns:
            DataFrame con la importancia de las características
        """
        # Calcular importancia usando varianza explicada y loadings
        loadings = np.abs(pca.components_)
        importancia = np.sum(
            loadings * pca.explained_variance_ratio_[:, np.newaxis], axis=0)

        # Normalizar importancia para que sume 1
        importancia = importancia / np.sum(importancia)

        # Crear DataFrame con resultados
        importancia_df = pd.DataFrame({
            'Característica': feature_names,
            'Importancia': importancia
        }).sort_values(by='Importancia', ascending=False)

        return importancia_df

    def calcular_y_filtrar_por_importancia(self, features, umbral_importancia=0.0167):
        try:
            # Estandarizar datos
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # Calcular PCA
            pca_temp = PCA()
            pca_temp.fit(features_scaled)

            # Calcular importancia
            importancia_df = self.calcular_importancia(
                pca_temp, features.columns)

            # Filtrar características
            features_importantes = importancia_df[importancia_df['Importancia']
                                                  >= umbral_importancia]['Característica']

            print(f"\nCaracterísticas eliminadas por baja importancia (umbral={
                  umbral_importancia}):")
            print(set(features.columns) - set(features_importantes))
            print("\nImportancia de características:")
            print(importancia_df)

            # Visualizar importancia
            plt.figure(figsize=(10, 6))
            plt.barh(importancia_df['Característica'],
                     importancia_df['Importancia'], color='skyblue')
            plt.xlabel('Importancia relativa')
            plt.title('Importancia de características')
            plt.axvline(x=umbral_importancia, color='r', linestyle='--',
                        label=f'Umbral ({umbral_importancia:.3f})')
            plt.legend()
            plt.gca().invert_yaxis()
            plt.grid(axis='x')
            plt.tight_layout()
            plt.show()

            return features[features_importantes]

        except Exception as e:
            print("Error al calcular y filtrar por importancia:", e)
            return features


    def analisis_pca(self):
        try:
            # Cargar y preparar datos
            df = pd.read_csv(self.database.csv_file_std)
            # Añadir columna de orden para identificar índices originales
            df['Orden'] = df.index
            etiquetas = df['Etiqueta']
            # Copia del DataFrame original para preservar todas las características
            original_df = df.copy()

            # Filtrar características por correlación
            features = self.filtrar_caracteristicas_por_correlacion(
                original_df.drop(['Etiqueta', 'Orden'], axis=1))
            feature_names = features.columns
            features = StandardScaler().fit_transform(features)

            # Configuración de iteraciones
            iteraciones = 0
            max_iteraciones = 80
            puntos_problematicos_totales = []

            while iteraciones < max_iteraciones:
                print(f"\nIteración {iteraciones + 1}:")

                # PCA 3D
                pca_3d = PCA(n_components=3)
                self.pca = pca_3d
                principalComponents_3d = pca_3d.fit_transform(features)
                explained_variance = [
                    f'{x:.2f}' for x in pca_3d.explained_variance_ratio_]
                print("Explained variance ratio for 3D PCA:", explained_variance)

                # Crear DataFrame de PCA
                principalDf_3d = pd.DataFrame(
                    data=principalComponents_3d, columns=['PC1', 'PC2', 'PC3'])
                finalDf_3d = pd.concat(
                    [principalDf_3d, etiquetas, df['Orden']], axis=1)

                # Detección de puntos problemáticos (distancia euclidiana)
                puntos_problematicos = []

                for etiqueta in finalDf_3d['Etiqueta'].unique():
                    puntos_etiqueta = finalDf_3d[finalDf_3d['Etiqueta'] == etiqueta]
                    centroide = puntos_etiqueta[[
                        'PC1', 'PC2', 'PC3']].mean().values
                    distancias = np.linalg.norm(
                        puntos_etiqueta[['PC1', 'PC2', 'PC3']] - centroide, axis=1)
                    umbral = np.mean(distancias) + 3 * \
                        np.std(distancias)  # Umbral dinámico

                    # Detectar puntos fuera del umbral
                    puntos_outliers = puntos_etiqueta.index[distancias > umbral].tolist(
                    )
                    puntos_problematicos.extend(puntos_outliers)

                print(f"Puntos problemáticos detectados (índices): {
                    puntos_problematicos}")

                if not puntos_problematicos:
                    print(
                        "No se encontraron más puntos problemáticos. Finalizando iteraciones.")
                    break

                # Almacenar puntos problemáticos globalmente
                puntos_problematicos_totales.extend(puntos_problematicos)

                # Eliminar puntos problemáticos del DataFrame original
                df = df.drop(index=puntos_problematicos).reset_index(drop=True)
                etiquetas = df['Etiqueta']
                features = df.drop(['Etiqueta', 'Orden'], axis=1)

                # Re-standardize features after dropping problematic points
                features = StandardScaler().fit_transform(features)

                iteraciones += 1

            # Graficar PCA 3D sin puntos problemáticos
            targets = ['berenjena', 'camote', 'papa', 'zanahoria']
            colors = ['r', 'g', 'b', 'y']

            fig_3d = plt.figure(figsize=(8, 8))
            ax_3d = fig_3d.add_subplot(111, projection='3d')
            ax_3d.set_xlabel('Principal Component 1', fontsize=15)
            ax_3d.set_ylabel('Principal Component 2', fontsize=15)
            ax_3d.set_zlabel('Principal Component 3', fontsize=15)
            ax_3d.set_title(
                '3 component PCA (sin puntos problemáticos)', fontsize=20)

            for target, color in zip(targets, colors):
                indicesToKeep = finalDf_3d['Etiqueta'] == target
                ax_3d.scatter(finalDf_3d.loc[indicesToKeep, 'PC1'],
                            finalDf_3d.loc[indicesToKeep, 'PC2'],
                            finalDf_3d.loc[indicesToKeep, 'PC3'], c=color, s=50)

            ax_3d.legend(targets)
            ax_3d.grid()
            plt.show()

            print(f"\nPuntos problemáticos eliminados en total: {
                puntos_problematicos_totales}")
            return df, puntos_problematicos_totales

        except Exception as e:
            print("Error al realizar análisis PCA: ", e)
            return None


    def get_filtered_dataframe(self):
        try:
            # Perform PCA analysis to filter out problematic points
            filtered_df, puntos_problematicos_totales = self.analisis_pca()

            # Extract features and labels from the filtered DataFrame
            features = filtered_df.drop(['Etiqueta', 'Orden'], axis=1)
            etiquetas = filtered_df['Etiqueta']

            return pd.concat([features, etiquetas], axis=1)
        except Exception as e:
            print("Error al obtener el DataFrame filtrado: ", e)
            return None

    def calcular_importancia_int(self):
        try:
            if not hasattr(self, 'pca') or not hasattr(self, 'feature_names'):
                print("Debe ejecutar `analisis_pca` antes de calcular la importancia.")
                return

            # Calcular la importancia total para cada característica
            importancia_df = self.calcular_importancia(
                self.pca, self.feature_names)

            print("\nImportancia de las características:")
            print(importancia_df)

            # Gráfico de importancia
            plt.figure(figsize=(10, 6))
            plt.barh(importancia_df['Característica'],
                     importancia_df['Importancia'], color='skyblue')
            plt.xlabel('Importancia acumulada')
            plt.title('Importancia de características basada en PCA')
            plt.gca().invert_yaxis()
            plt.grid(axis='x')
            plt.show()

        except Exception as e:
            print("Error al calcular la importancia: ", e)
            return None

    def guardar_dataframe_filtrado(self, output_path):
        """
        Guarda el DataFrame filtrado en un archivo CSV.

        Args:
            output_path: Ruta del archivo CSV de salida.
        """
        try:
            filtered_df = self.get_filtered_dataframe()
            if os.path.exists(output_path):
                os.remove(output_path)
            if filtered_df is not None:
                filtered_df.to_csv(output_path, index=False)
                print(f"DataFrame filtrado guardado en {output_path}")
            else:
                print("No se pudo obtener el DataFrame filtrado.")
        except Exception as e:
            print("Error al guardar el DataFrame filtrado: ", e)
