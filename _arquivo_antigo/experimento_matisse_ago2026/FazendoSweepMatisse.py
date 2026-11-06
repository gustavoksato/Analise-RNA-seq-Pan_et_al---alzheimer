import scanpy as sc
import pandas as pd
import numpy as np
import gc

print("1. Carregando Gabaritos e Base Ortogonal do SWeeP...")
features_pan = pd.read_csv("features_bin.tsv", sep='\t', header=None)[0].tolist()
# Carrega a base matemática que o R exportou
base_sweep = pd.read_csv("base_sweep_pan.csv").values.astype(np.float32) 

print("2. Carregando Matisse...")
adata_matisse = sc.read_h5ad("matrizFiltradaeNormalizadaM.h5ad")
matisse_genes = adata_matisse.var_names.tolist()

print("\n3. Mapeando Colunas...")
pan_idx_map = {gene: idx for idx, gene in enumerate(features_pan)}
genes_manter = set(matisse_genes).intersection(features_pan)
genes_ausentes = set(features_pan) - set(matisse_genes)

# Mapeamento rápido e vetorial
indices_ausentes = [pan_idx_map[g] for g in genes_ausentes]
old_cols = [i for i, g in enumerate(matisse_genes) if g in genes_manter]
new_cols = [pan_idx_map[matisse_genes[i]] for i in old_cols]

num_celulas = adata_matisse.n_obs
num_genes_pan = len(features_pan)
chunk_size = 2000 # Tamanho do lote (seguro para qualquer notebook)

arquivo_saida = "SWeeP_600_Matisse_Python.txt"

print(f"\n[INFO] Iniciando Streaming Matemático (SWeeP) em {num_celulas} células...")
with open(arquivo_saida, "w") as f:
    # Escreve o cabeçalho (V1 a V600)
    cabecalho = "\t".join([f"V{i+1}" for i in range(600)]) + "\n"
    f.write(cabecalho)
    
    for start in range(0, num_celulas, chunk_size):
        end = min(start + chunk_size, num_celulas)
        
        # Pega um pedaço dos dados originais (em formato denso temporário)
        chunk_data = adata_matisse.X[start:end].toarray()
        
        # Cria uma matriz "em branco" do tamanho exato do PAN para esse lote
        X_alinhado = np.zeros((end - start, num_genes_pan), dtype=np.float32)
        
        # A MÁGICA: Preenche os genes ausentes com 0.5 instantaneamente
        X_alinhado[:, indices_ausentes] = 0.5
        
        # A MÁGICA 2: Encaixa os dados reais nas colunas certas do PAN
        X_alinhado[:, new_cols] = chunk_data[:, old_cols]
        
        # A MÁGICA 3: Multiplicação SWeeP! (Matriz de células x Base Ortogonal)
        sweep_chunk = np.dot(X_alinhado, base_sweep)
        
        # Salva o lote já com as 600 dimensões reduzidas no arquivo de texto
        pd.DataFrame(sweep_chunk).to_csv(f, sep='\t', header=False, index=False)
        
        print(f" -> Processado e Projetado: {end}/{num_celulas} células...")
        
        # Libera a RAM imediatamente
        del chunk_data, X_alinhado, sweep_chunk
        gc.collect()

print(f"\n[SUCESSO] Operação finalizada! Matriz SWeeP salva em '{arquivo_saida}'.")