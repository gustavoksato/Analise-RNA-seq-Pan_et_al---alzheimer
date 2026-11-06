import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

# 1. Leer la matriz SWeeP reconociendo el encabezado (58326 células)
caminho_sweep = '/media/gustavo-k-sato/D61CF48E1CF46B3F1/Analise-RNA-seq-Pan_et_al---alzheimer/Documents/BIOINFO/analise.pan_etal/Matriz_SWeeP_Alzheimer.txt'
df_sweep = pd.read_csv(caminho_sweep, sep='\t', header=0)

# Filtrar las últimas 600 dimensiones en caso de que R haya exportado una columna de índices residual
if df_sweep.shape[1] > 600:
    df_sweep = df_sweep.iloc[:, -600:]

# 2. Leer los metadatos biológicos
metadados = pd.read_csv('anotacoes_tecidos.txt', sep='\t', index_col=0)

# 3. Construir el objeto AnnData con la matriz matemática
adata_pca = sc.AnnData(X=df_sweep.values)

# 4. Restaurar los nombres de las células transfiriendo el índice exacto de los metadatos
adata_pca.obs_names = metadados.index

# 5. Inyectar las anotaciones y consolidar las 7 clases
adata_pca.obs['manual_annotation'] = metadados['manual_annotation'].replace(
    ['VLMCs', 'Endothelial'], 'Endothelial / Outros'
)

# 6. Forzar el tipo 'category' para que Scanpy aplique una paleta de colores discretos
adata_pca.obs['manual_annotation'] = adata_pca.obs['manual_annotation'].astype('category')

# 7. Ejecutar y guardar el PCA
sc.tl.pca(adata_pca, n_comps=50)
sc.pl.pca(adata_pca, color='manual_annotation', 
          title="PCA da Matriz SWeeP 600D (7 Classes Consolidadas)",
          save="_sweep_600d_7classes_corrigido.png")