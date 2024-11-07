import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('momentos_hu.csv')
print(data.head())
print(data.columns)
data.columns = data.columns.str.strip()
hu_momentos_columnas = [f'Hu{i+1}' for i in range(7)]
missing_columns = [col for col in hu_momentos_columnas if col not in data.columns]
if missing_columns:
    raise KeyError(f"Missing columns in the DataFrame: {missing_columns}")

X = data[hu_momentos_columnas].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3'])

print("Variancia explicada por cada componente:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {var:.2f}")

# Codificar los nombres de las verduras
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data['Nombre'])
    

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], c=labels, cmap='viridis', s=50)

# Add legend to show names corresponding to each color
unique_labels = np.unique(labels)
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(label / max(labels)), markersize=10) for label in unique_labels]
legend1 = ax.legend(handles, label_encoder.inverse_transform(unique_labels), title='Verduras')
ax.add_artist(legend1)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

plt.show()