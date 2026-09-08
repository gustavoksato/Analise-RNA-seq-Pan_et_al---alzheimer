import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import scanpy as sc
import random
import gc

print("1. Carregando Matriz Ouro e Features do PAN...")
X_ouro_coo = sio.mmread("Matriz_Pan_Binarizada_Limpa.mtx").astype(np.float32)
num_linhas = X_ouro_coo.shape[0]
formato_original = X_ouro_coo.shape

features_pan = pd.read_csv("features_pan_limpo.tsv", sep='\t', header=None)[0].tolist()

print("2. Puxando Genes do Matisse para achar os Exclusivos...")
adata_matisse = sc.read_h5ad("matrizFiltradaeNormalizadaM.h5ad", backed='r')
features_matisse = set(adata_matisse.var_names)

genes_exclusivos = [g for g in features_pan if g not in features_matisse]
random.seed(42)
genes_alvo_ruido = random.sample(genes_exclusivos, 10000)

print(f"-> Total de genes exclusivos do PAN: {len(genes_exclusivos)}")
print(f"-> Sorteados para corrupção: 10.000 genes.")

indices_colunas_ruido = [features_pan.index(g) for g in genes_alvo_ruido]

print("\n3. Configurando a matriz base (NumPy Arrays)...")
linhas_atual = X_ouro_coo.row
colunas_atual = X_ouro_coo.col
dados_atual = X_ouro_coo.data

del X_ouro_coo
gc.collect()

print("\n4. Gerando as 10 Matrizes Ruidosas (Modo Acumulativo)...")
for i in range(1, 11):
    nivel_ruido = i * 1000
    print(f"-> Adicionando ruído na leva {i} (acumulando para {nivel_ruido} genes)...")
    
    # Isola APENAS os 1000 genes desta rodada específica
    novos_genes_alvo = np.array(indices_colunas_ruido[(i-1)*1000 : i*1000])
    
    # Remove os dados originais dessas 1000 colunas da matriz atual
    mask_manter = ~np.isin(colunas_atual, novos_genes_alvo)
    linhas_atual = linhas_atual[mask_manter]
    colunas_atual = colunas_atual[mask_manter]
    dados_atual = dados_atual[mask_manter]
    
    # Gera o bloco de ruído de 0.5 estritamente para essas 1000 colunas
    ruido_colunas = np.repeat(novos_genes_alvo, num_linhas)
    ruido_linhas = np.tile(np.arange(num_linhas), len(novos_genes_alvo))
    ruido_dados = np.full(len(ruido_colunas), 0.5, dtype=np.float32)
    
    # Anexa o novo bloco de ruído, atualizando o estado da matriz de forma cumulativa
    linhas_atual = np.concatenate([linhas_atual, ruido_linhas])
    colunas_atual = np.concatenate([colunas_atual, ruido_colunas])
    dados_atual = np.concatenate([dados_atual, ruido_dados])
    
    # Reconstrói a matriz e salva
    X_novo = sp.coo_matrix((dados_atual, (linhas_atual, colunas_atual)), shape=formato_original)
    
    nome_arquivo = f"Matriz_Pan_Ruido_{i}k.mtx"
    sio.mmwrite(nome_arquivo, X_novo)
    print(f"   Salvo: {nome_arquivo}")
    
    # Limpa apenas as variáveis temporárias deste loop
    del mask_manter, novos_genes_alvo, ruido_colunas, ruido_linhas, ruido_dados, X_novo
    gc.collect()

print("\n[SUCESSO] 10 matrizes de ruído cumulativas exportadas!")