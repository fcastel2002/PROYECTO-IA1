from sklearn.neighbors import KNeighborsClassifier

class KNN_Predict:
    def __init__(self, k=3):
        self.modelo = KNeighborsClassifier(n_neighbors=k)

    def entrenar(self, datos_entrenamiento, etiquetas_entrenamiento):
        self.modelo.fit(datos_entrenamiento, etiquetas_entrenamiento)
        print("Modelo entrenado.")

    def predecir(self, datos_prueba):
        predicciones = self.modelo.predict(datos_prueba)
        print("Predicciones realizadas.")
        return predicciones
