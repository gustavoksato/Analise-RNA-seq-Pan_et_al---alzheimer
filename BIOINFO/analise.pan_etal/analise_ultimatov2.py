import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

print("1. Carregando dados brutos...")
adata_completo = sc.read_h5ad("BIOINFO/analise.pan_etal/GSE267301_Full_Dataset.h5ad")
adata = ad.AnnData(
    X = adata_completo.layers['nuclear_original'].copy(),
    obs = adata_completo.obs.copy(),
    var = adata_completo.var.copy()
)
del adata_completo

print(adata.var.columns)
print(adata.obs.columns)

print(adata.var.index1)


print("2. Aplicando Filtros Biológicos (QC)...")
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

# Aplicação sequencial e limpa dos três filtros (Mitocôndria, Genes e UMIs)
adata = adata[
    (adata.obs.pct_counts_mt <= 15) & 
    (adata.obs.n_genes_by_counts >= 200) & 
    (adata.obs.n_genes_by_counts <= 8000) & 
    (adata.obs.total_counts >= 500), :
].copy()

print(f"-> Células mantidas após QC biológico: {adata.shape[0]}")

print("3. Mapeando Anotações Manuais...")
anotacoes_df = pd.read_csv("BIOINFO/analise.pan_etal/anotacoes_tecidos.txt", sep='\t', header=None, names=['Barcode', 'cell_type'])
dict_anotacoes = dict(zip(anotacoes_df['Barcode'], anotacoes_df['cell_type']))

adata.obs['cell_type_text'] = adata.obs.index.map(dict_anotacoes)

dicionario_map = {
    'Astrocytes': 1, 'Oligodendrocytes': 2, 'Exc. Neurons': 3, 
    'Inh. Neurons': 4, 'OPCs': 5, 'Microglia': 6, 
    'Endothelial': 7, 'VLMCs': 8
}
adata.obs['Cell_Type_Num'] = adata.obs['cell_type_text'].map(dicionario_map)

print("4. Removendo Células sem Rótulo (NaN)...")
adata = adata[~adata.obs['Cell_Type_Num'].isna()].copy()
adata.obs['Cell_Type_Num'] = adata.obs['Cell_Type_Num'].astype(int)

print(f"-> Shape da Matriz Ouro (Validada): {adata.shape}")

print("5. Binarizando Matriz de Expressão...")
# Usa int8 para compressão máxima de memória
adata.X = (adata.X > 0).astype(np.int8)

print("6. Exportando Arquivos Finais para PyTorch e R...")
# 6.1 Matriz Matemática
sio.mmwrite("Matriz_Pan_Binarizada_Limpa.mtx", adata.X)
# 6.2 Target (y) para o PyTorch
adata.obs[['Cell_Type_Num']].to_csv("Rotulos_Pan_Limpos.csv", index=False)
# 6.3 Metadados para montagem
adata.obs.index.to_series().to_csv("barcodes_pan_limpo.tsv", sep='\t', index=False, header=False)
adata.var.index.to_series().to_csv("features_pan_limpo.tsv", sep='\t', index=False, header=False)

print("\nPipeline de Preparação Concluído com Sucesso! 🎯")