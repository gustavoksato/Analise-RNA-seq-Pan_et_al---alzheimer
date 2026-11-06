import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import scanpy as sc
import random
import gc

print("1. Carregando Matriz Ouro Base e Features...")
X_ouro_coo = sio.mmread("Matriz_Pan_Binarizada_Limpa.mtx").astype(np.float32)
num_linhas = X_ouro_coo.shape[0]
formato_original = X_ouro_coo.shape

features_pan = pd.read_csv("features_bin.tsv", sep='\t', header=None)[0].tolist()

print("\n2. Isolando genes exclusivos do PAN (ausentes no Matisse)...")
adata_matisse = sc.read_h5ad("matrizFiltradaeNormalizadaM.h5ad", backed='r')
features_matisse = set(adata_matisse.var_names)

genes_exclusivos = [g for g in features_pan if g not in features_matisse]
print(f" -> Total de genes exclusivos encontrados: {len(genes_exclusivos)}")

random.seed(42) 
# Expandido para 10.000 genes
genes_alvo_ruido = random.sample(genes_exclusivos, 10000) 
indices_colunas_ruido = [features_pan.index(g) for g in genes_alvo_ruido]

linhas_originais = X_ouro_coo.row
colunas_originais = X_ouro_coo.col
dados_originais = X_ouro_coo.data

del X_ouro_coo
gc.collect()

print("\n3. Gerando Ruído com Distribuição de Bernoulli (5k a 10k)...")
# Loop iniciando em 5k e indo até 10k
for i in range(10, 11):
    nivel_ruido = i * 1000
    print(f" -> Construindo cenário {i}k (Corrompendo {nivel_ruido} genes cumulativos)...")
    
    colunas_para_alterar = np.array(indices_colunas_ruido[:nivel_ruido])
    
    mask_manter = ~np.isin(colunas_originais, colunas_para_alterar)
    linhas_temp = linhas_originais[mask_manter]
    colunas_temp = colunas_originais[mask_manter]
    dados_temp = dados_originais[mask_manter]
    
    lista_linhas_ruido = []
    lista_colunas_ruido = []
    
    for gene_idx in colunas_para_alterar:
        sorteio = np.random.rand(num_linhas) > 0.5 
        celulas_que_deram_cara = np.where(sorteio)[0]
        
        lista_linhas_ruido.append(celulas_que_deram_cara.astype(np.int32))
        lista_colunas_ruido.append(np.full(len(celulas_que_deram_cara), gene_idx, dtype=np.int32))
        
    ruido_linhas_sparse = np.concatenate(lista_linhas_ruido)
    ruido_colunas_sparse = np.concatenate(lista_colunas_ruido)
    ruido_dados_sparse = np.ones(len(ruido_linhas_sparse), dtype=np.float32)
    
    linhas_finais = np.concatenate([linhas_temp, ruido_linhas_sparse])
    colunas_finais = np.concatenate([colunas_temp, ruido_colunas_sparse])
    dados_finais = np.concatenate([dados_temp, ruido_dados_sparse])
    
    X_novo = sp.coo_matrix((dados_finais, (linhas_finais, colunas_finais)), shape=formato_original)
    nome_arquivo = f"Matriz_Pan_Ruido_{i}k.mtx"
    sio.mmwrite(nome_arquivo, X_novo)
    
    print(f"    [OK] Salvo: {nome_arquivo}")
    
    del mask_manter, linhas_temp, colunas_temp, dados_temp
    del lista_linhas_ruido, lista_colunas_ruido
    del ruido_linhas_sparse, ruido_colunas_sparse, ruido_dados_sparse
    del linhas_finais, colunas_finais, dados_finais, X_novo
    gc.collect()

print("\n[SUCESSO] Matrizes de 5k a 10k geradas sem travar a máquina!")