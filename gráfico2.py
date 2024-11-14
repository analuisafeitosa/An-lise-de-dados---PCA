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

# Inicializar listas para armazenar as amostras e os labels
all_samples = []
labels = []

# Processar cada arquivo separadamente
for label, filename in file_names.items():
    data = load_data(filename)
    data['label'] = label  # Adicionar o label de origem para cada linha
    all_samples.append(data)

# Concatenar todas as amostras em um único DataFrame
combined_df = pd.concat(all_samples, ignore_index=True)

# Criar uma tabela pivotada onde as linhas são os wavenumbers e as colunas são as amostras
pivot_df = combined_df.pivot_table(index='wavenumber', columns='label', values='transmittance')

# Preencher valores NaN (caso haja diferenças nos wavenumbers entre os arquivos)
pivot_df.fillna(method='ffill', inplace=True)
pivot_df.fillna(method='bfill', inplace=True)

# Transpor o DataFrame para que cada linha seja uma amostra para o PCA
df_transposed = pivot_df.transpose()

# Aplicar o PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_transposed)

# Criar um DataFrame para os componentes principais
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['label'] = df_transposed.index

# Plotar os componentes principais
plt.figure(figsize=(10, 7))
colors = {'LG': 'purple', 'LA1_Recife': 'blue', 'LA2_Aldeia': 'green', 'LO': 'red'}

# Plotar todas as amostras como pontos individuais no gráfico
for label in pca_df['label'].unique():
    plt.scatter(
        pca_df.loc[pca_df['label'] == label, 'PC1'],
        pca_df.loc[pca_df['label'] == label, 'PC2'],
        label=label,
        color=colors[label],
        s=30  # Ajuste o tamanho dos pontos
    )

plt.title('PCA of FTIR Data (All Samples)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()
