import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.io as sio

# ==================== 1. PREPARAÇÃO DO MATISSE ====================
print("Carregando e binarizando o Matisse...")
adata = sc.read_h5ad("BIOINFO/analise.pan_etal/matrizFiltradaeNormalizadaM.h5ad")

adata_binario = adata.copy()
# Binariza a matriz do Matisse mantendo a esparsidade
adata_binario.X = (adata_binario.X > 0).astype(int)


# ==================== 2. CARREGANDO O GABARITO (PAN) ====================
print("Lendo os genes de referência do Pan...")
# Lê o arquivo TSV que você exportou (como não tem header, usamos header=None)
genes_pan_df = pd.read_csv("features_bin.tsv", sep='\t', header=None)

# Converte a coluna do DataFrame em uma lista simples de Python
genes_pan = genes_pan_df[0].tolist()


# ==================== 3. ALINHAMENTO ====================
print(f"Iniciando alinhamento para {len(genes_pan)} genes. Inserindo 0.5 nos ausentes...")

genes_matisse = set(adata_binario.var_names)
# Dicionário de busca rápida para otimizar o tempo do loop
idx_matisse = {gene: i for i, gene in enumerate(adata_binario.var_names)}

colunas_alinhadas = []
n_celulas_matisse = adata_binario.n_obs

for gene in genes_pan: 
    if gene in genes_matisse:
        # Copia a coluna original do Matisse binarizado
        col = adata_binario.X[:, idx_matisse[gene]]
        colunas_alinhadas.append(col)
    else:
        # Cria a coluna sintética preenchida com 0.5
        col_missing = sp.csr_matrix(np.full((n_celulas_matisse, 1), 0.5))
        colunas_alinhadas.append(col_missing)

print("Empilhando matriz alinhada (Isso pode consumir RAM)...")
nova_matriz_matisse = sp.hstack(colunas_alinhadas).tocsr()


# ==================== 4. NOVO OBJETO ====================
# Recria o DataFrame de variáveis (genes) baseado na lista do Pan
var_pan_alinhado = pd.DataFrame(index=genes_pan)

adata_matisse_alinhado = ad.AnnData(
    X=nova_matriz_matisse,
    obs=adata_binario.obs.copy(), # Células originais do Matisse
    var=var_pan_alinhado          # Genes originais do Pan
)

print("\n--- Resultado Final ---")
print(f"Shape do Matisse Alinhado: {adata_matisse_alinhado.shape}")


# ==================== 5. EXPORTAÇÃO (Opcional) ====================
# print("Exportando Matisse Alinhado para R/SWeeP...")
# sio.mmwrite("Matriz_Matisse_Alinhada.mtx", adata_matisse_alinhado.X)
# adata_matisse_alinhado.obs.index.to_series().to_csv("barcodes_matisse.tsv", sep='\t', index=False, header=False)
# adata_matisse_alinhado.var.index.to_series().to_csv("features_matisse.tsv", sep='\t', index=False, header=False)



import pandas as pd

# 1. Carrega os genes do Pan (Gabarito)
genes_pan_df = pd.read_csv("features_bin.tsv", sep='\t', header=None)
set_pan = set(genes_pan_df[0])

# 2. Extrai os genes do Matisse (Assumindo que adata_binario já está na memória)
set_matisse = set(adata_binario.var_names)

# 3. Cruzamento Matemático
genes_comuns = set_pan.intersection(set_matisse)
genes_so_no_pan = set_pan - set_matisse
genes_so_no_matisse = set_matisse - set_pan

# 4. Relatório de Alinhamento
print("\n================ DIAGNÓSTICO DE GENES ================")
print(f"Total de genes originais no PAN: {len(set_pan)}")
print(f"Total de genes originais no Matisse: {len(set_matisse)}")
print("------------------------------------------------------")
print(f"[MATCH] Genes em comum: {len(genes_comuns)}")
print(f"        -> Estes manterão seus valores originais (0 e 1).")
print(f"[MISSING] Genes exclusivos do PAN: {len(genes_so_no_pan)}")
print(f"        -> Estes receberão o valor imputado (0.5) no Matisse.")
print(f"[DROP] Genes exclusivos do Matisse: {len(genes_so_no_matisse)}")
print(f"        -> Estes serão descartados para espelhar o PAN.")
print("======================================================\n")

# Opcional: Espiar alguns nomes para garantir que não há erro de formatação
# (ex: PAN usando 'TP53' e Matisse usando 'ENSG00000141510')
print("Exemplos de genes em comum:", list(genes_comuns)[:5])
print("Exemplos de genes exclusivos do PAN:", list(genes_so_no_pan)[:5])




import pandas as pd

# 1. Carrega os genes do Pan
genes_pan_df = pd.read_csv("features_bin.tsv", sep='\t', header=None)

# 2. APLICA A LÓGICA DO FUJITA: Corta o decimal (versionamento) de todos os genes
genes_pan_limpos = set(genes_pan_df[0].str.split('.').str[0])

# 3. Puxa os genes do Matisse
set_matisse = set(adata_binario.var_names)

# 4. Novo Cruzamento Matemático
novos_genes_comuns = genes_pan_limpos.intersection(set_matisse)
novos_so_no_pan = genes_pan_limpos - set_matisse
novos_so_no_matisse = set_matisse - genes_pan_limpos

print("\n================ NOVO DIAGNÓSTICO (SEM DECIMAIS) ================")
print(f"Total de genes no PAN (limpos): {len(genes_pan_limpos)}")
print(f"Total de genes no Matisse: {len(set_matisse)}")
print("-----------------------------------------------------------------")
print(f"[MATCH] Genes em comum subiu de 19466 para: {len(novos_genes_comuns)}")
print(f"[MISSING] Exclusivos do PAN caiu para: {len(novos_so_no_pan)}")
print(f"[DROP] Exclusivos do Matisse caiu para: {len(novos_so_no_matisse)}")
print("=================================================================\n")



import pandas as pd

# 1. Extraindo os 20 primeiros genes do PAN
genes_pan_df = pd.read_csv("features_bin.tsv", sep='\t', header=None)
pan_head = genes_pan_df[0].head(20).tolist()

# 2. Extraindo os 20 primeiros genes do Matisse
# (Assumindo que adata_binario ou adata ainda estão na memória)
matisse_head = adata_binario.var_names[:20].tolist()

# 3. Print formatado para facilitar a comparação visual
print("=== INSPECÃO VISUAL: 20 PRIMEIROS GENES ===")

print("\n[PAN] (features_bin.tsv):")
for i, gene in enumerate(pan_head):
    print(f"{i+1:02d} - {gene}")

print("\n[MATISSE] (matrizFiltradaeNormalizadaM.h5ad):")
for i, gene in enumerate(matisse_head):
    print(f"{i+1:02d} - {gene}")

print("6. Exportando para o R (SWeeP)...")
sio.mmwrite("Matriz_Matisse_Alinhada.mtx", adata_final.X)
adata_final.obs.index.to_series().to_csv("barcodes_matisse.tsv", sep='\t', index=False, header=False)
adata_final.var.index.to_series().to_csv("features_matisse.tsv", sep='\t', index=False, header=False)
print("===========================================")