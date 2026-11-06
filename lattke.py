import scanpy as sc
import pandas as pd
import os

# 1. Caminho exato da matriz extraída
pasta_amostra = "/media/gustavo-k-sato/D61CF48E1CF46B3F1/Analise-RNA-seq-Pan_et_al---alzheimer/Documents/dataset_down_syndrome/GSM9162706_S379_counts_matrix"
arquivo_saida = os.path.join(pasta_amostra, "barcodes_aprovados_QC.csv")

print("[1/3] Carregando a matriz bruta de 10.000 células no Scanpy...")
# Lendo usando gene_symbols apenas para facilitar a busca pelo prefixo 'MT-'
adata = sc.read_10x_mtx(pasta_amostra, var_names='gene_symbols')
total_inicial = adata.n_obs

print("\n[2/3] Calculando métricas de QC...")
# Identifica genes mitocondriais
adata.var['mt'] = adata.var_names.str.upper().str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

# 3. Aplicando os mesmos filtros do Pan et al.
mask_mito = adata.obs['pct_counts_mt'] < 15.0
mask_genes = (adata.obs['n_genes_by_counts'] >= 200) & (adata.obs['n_genes_by_counts'] < 8000)
mask_umis = adata.obs['total_counts'] >= 500

mask_final = mask_mito & mask_genes & mask_umis

adata_limpo = adata[mask_final].copy()
total_final = adata_limpo.n_obs

print("\n[3/3] Resultado do Funil de QC (Regras Pan et al.):")
print(f" -> Células iniciais: {total_inicial}")
print(f" -> Após Filtro Mitocondrial (<15%): {mask_mito.sum()}")
print(f" -> Após Filtro de Genes (200-8000): {(mask_mito & mask_genes).sum()}")
print(f" -> Após Filtro de UMIs (>=500): {total_final}")
print(f" -> Taxa de Retenção: {(total_final/total_inicial)*100:.1f}%")

# Exporta os barcodes aprovados
barcodes_validos = adata_limpo.obs_names.to_series()
barcodes_validos.to_csv(arquivo_saida, index=False, header=["Barcode"])

print(f"\n[SUCESSO] Barcodes aprovados salvos em: {arquivo_saida}")