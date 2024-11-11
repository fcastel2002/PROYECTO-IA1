import pandas as pd

def normalizar_centroides():
    # Leer el archivo de centroides
    df = pd.read_csv('centroides.csv')
    
    # Seleccionar las columnas a normalizar
    columnas_a_normalizar = df.columns[1:]
    
    # Aplicar estandarización (z-score normalization) por fila
    for index, row in df.iterrows():
        df.loc[index, columnas_a_normalizar] = (row[columnas_a_normalizar] - row[columnas_a_normalizar].mean()) / row[columnas_a_normalizar].std()
    
    # Guardar el resultado en un nuevo archivo
    df.to_csv('centroides_normalizados.csv', index=False)

if __name__ == '__main__':
    normalizar_centroides()
