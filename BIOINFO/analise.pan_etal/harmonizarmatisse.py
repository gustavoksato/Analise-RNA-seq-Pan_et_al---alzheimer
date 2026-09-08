import pandas as pd
import scanpy as sc
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import gc

print("1. Carregando e HIGIENIZANDO o Dicionário (Pedra de Roseta)...")
df_roseta = pd.read_csv('features(1).tsv', sep='\t', header=None)
df_roseta[0] = df_roseta[0].astype(str).str.strip().str.upper()
df_roseta[1] = df_roseta[1].astype(str).str.strip().str.upper()
dict_ensg_to_symbol = dict(zip(df_roseta[0], df_roseta[1]))

print("2. Carregando Matisse e o Modelo Ouro (PAN)...")
adata_matisse = sc.read_h5ad("matrizFiltradaeNormalizadaM.h5ad", backed='r')
matisse_genes = [str(g).strip().upper() for g in adata_matisse.var_names]
num_celulas = adata_matisse.n_obs

# ATENÇÃO: Coloque o nome correto do seu arquivo h5ad do PAN aqui
adata_pan = sc.read_h5ad("GSE267301_Full_Dataset.h5ad", backed='r')
index1_list = adata_pan.var['index1'].tolist()
gene_name_list = adata_pan.var.index.tolist()
num_genes_pan = len(gene_name_list)

# Liberando a memória do PAN que não vamos mais usar
del adata_pan
gc.collect()

print("\n3. Traduzindo IDs (via index1) e Blindando Colisões...")
features_pan_traduzidas = []
seen_names = set()
colisoes_evitadas = 0

for ensg_raw, original_name in zip(index1_list, gene_name_list):
    ensg_clean = str(ensg_raw).split('.')[0].strip().upper()
    
    # Tenta traduzir
    if ensg_clean in dict_ensg_to_symbol:
        candidato = dict_ensg_to_symbol[ensg_clean]
    else:
        candidato = str(original_name).strip().upper()
        
    # TRAVA ANTICOLISÃO: Se o nome já existir, revertemos para o ENSG puro
    # Isso garante que a matriz SWeeP não perca nenhuma das 61.541 dimensões
    if candidato in seen_names:
        candidato = str(ensg_raw).strip().upper()
        colisoes_evitadas += 1

    seen_names.add(candidato)
    features_pan_traduzidas.append(candidato)

if colisoes_evitadas > 0:
    print(f" -> [ALERTA CORRIGIDO] {colisoes_evitadas} genes sofreram colisão e foram revertidos para manter a dimensão.")

print("\n4. Cruzando informações dos Datasets...")
pan_idx_map = {gene: idx for idx, gene in enumerate(features_pan_traduzidas)}

genes_manter = set(matisse_genes).intersection(features_pan_traduzidas)
genes_ausentes = set(features_pan_traduzidas) - set(matisse_genes)

print(f" -> Células no Matisse: {num_celulas}")
print(f" -> Genes no Matisse (Total): {len(matisse_genes)}")
print(f" -> Genes em comum (mantidos): {len(genes_manter)}")
print(f" -> Genes exclusivos do PAN (receberão 0.5): {len(genes_ausentes)}")

print("\n5. Realinhando Matriz Base...")
X_matisse = adata_matisse.to_memory().X.tocoo()

# Mapeia as colunas do Matisse para as novas colunas traduzidas do PAN
old_to_new_col = {i: pan_idx_map[g] for i, g in enumerate(matisse_genes) if g in genes_manter}

# Isola apenas os dados que deram match
mask = np.isin(X_matisse.col, list(old_to_new_col.keys()))
linhas_base = X_matisse.row[mask]
colunas_old = X_matisse.col[mask]
dados_base = X_matisse.data[mask]

# Traduz as coordenadas para o padrão do PAN
mapper = np.zeros(max(colunas_old.max() + 1, len(matisse_genes)), dtype=np.int32)
for old, new in old_to_new_col.items():
    mapper[old] = new
colunas_base_new = mapper[colunas_old]

# Limpeza de RAM crítica
del X_matisse, adata_matisse, mapper, mask, colunas_old, old_to_new_col
gc.collect()

print(f"\n6. Injetando 0.5 em {len(genes_ausentes)} colunas (ATENÇÃO: Pico de Memória RAM)...")
indices_ausentes = [pan_idx_map[g] for g in genes_ausentes]

ruido_colunas = np.repeat(indices_ausentes, num_celulas)
ruido_linhas = np.tile(np.arange(num_celulas), len(indices_ausentes))
ruido_dados = np.full(len(ruido_colunas), 0.5, dtype=np.float32)

print("\n7. Construindo Matriz Final e Exportando (.mtx)...")
linhas_finais = np.concatenate([linhas_base, ruido_linhas])
colunas_finais = np.concatenate([colunas_base_new, ruido_colunas])
dados_finais = np.concatenate([dados_base, ruido_dados])

X_final = sp.coo_matrix((dados_finais, (linhas_finais, colunas_finais)), shape=(num_celulas, num_genes_pan))

nome_saida = "Matriz_Matisse_Harmonizada.mtx"
sio.mmwrite(nome_saida, X_final)

print(f"\n[SUCESSO] Operação concluída! Matriz salva como {nome_saida}")