import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from glob import glob

# Função para carregar dados a partir dos arquivos
def load_data(filename):
    # Carregar o arquivo como um DataFrame, assumindo que possui duas colunas
    data = pd.read_csv(filename, sep='\t', header=None)
    data.columns = ['wavenumber', 'transmittance']
    return data

# Carregar os dados dos arquivos fornecidos
file_names = {
    'LA1_Recife': 'LA1_Recife.txt',
    'LA2_Aldeia': 'LA2_Aldeia.txt',
    'LO': 'LO.txt',
    'LG': 'LG.txt'
}

datasets = {}
for label, filename in file_names.items():
    datasets[label] = load_data(filename)

# Unificar os dados para aplicar o PCA
all_transmittance = []
labels = []

for label, data in datasets.items():
    all_transmittance.append(data['transmittance'])
    labels.extend([label] * len(data))

# Converter para DataFrame e preencher NaNs com a média para alinhar os dados
df = pd.DataFrame(all_transmittance).transpose().fillna(method='ffill')

# Aplicar o PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df)

# Criar um DataFrame para os componentes principais
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['label'] = labels[:len(pca_df)]

# Plotar os componentes principais
plt.figure(figsize=(10, 7))
colors = {'LA1_Recife': 'blue', 'LA2_Aldeia': 'green', 'LO': 'red', 'LG': 'purple'}

for label in pca_df['label'].unique():
    plt.scatter(
        pca_df.loc[pca_df['label'] == label, 'PC1'],
        pca_df.loc[pca_df['label'] == label, 'PC2'],
        label=label,
        color=colors[label]
    )

plt.title('PCA of FTIR Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()
