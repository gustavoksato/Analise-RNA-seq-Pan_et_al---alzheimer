import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from anndata.io import read_elem
import gc

caminho_h5ad = "/home/gustavo-k-sato/Downloads/GSE263468_processed_data.h5ad"
chunk_size = 10000

print("--- EXPLORAÇÃO GLOBAL DE QC NA CAMADA 'COUNTS' ---")
try:
    with h5py.File(caminho_h5ad, 'r') as f:
        obs = read_elem(f['obs'])
        var = read_elem(f['var'])
        total_cells = obs.shape[0]
        total_genes = var.shape[0]
        
        # Identificando os genes mitocondriais
        mito_genes = var.index.str.upper().str.startswith('MT-')
        
        # Listas para guardar as métricas de todas as células
        all_n_genes = []
        all_total_counts = []
        all_pct_mt = []

        print(f"Processando {total_cells} células no disco em lotes de {chunk_size}...")
        
        # Acessando os ponteiros da matriz CSR bruta
        counts_layer = f['layers']['counts']
        dset_data = counts_layer['data']
        dset_indices = counts_layer['indices']
        dset_indptr = counts_layer['indptr']

        for i in range(0, total_cells, chunk_size):
            end = min(i + chunk_size, total_cells)
            
            ptr_start = dset_indptr[i]
            ptr_end = dset_indptr[end]
            
            # Reconstrói apenas o bloco atual na RAM
            chunk_X = sparse.csr_matrix(
                (dset_data[ptr_start:ptr_end], 
                 dset_indices[ptr_start:ptr_end], 
                 dset_indptr[i:end+1] - ptr_start), 
                shape=(end - i, total_genes)
            )
            
            # Cálculo de QC
            total_counts = np.sum(chunk_X, axis=1).A1
            mt_counts = np.sum(chunk_X[:, mito_genes], axis=1).A1
            n_genes = np.sum(chunk_X > 0, axis=1).A1
            
            with np.errstate(divide='ignore', invalid='ignore'):
                pct_mt = np.nan_to_num((mt_counts / total_counts) * 100)
                
            all_n_genes.extend(n_genes)
            all_total_counts.extend(total_counts)
            all_pct_mt.extend(pct_mt)
            
            if end % 50000 == 0 or end == total_cells:
                print(f"  -> Analisadas {end}/{total_cells} células...")
                
            del chunk_X
            gc.collect()

    print("\n--- RESUMO ESTATÍSTICO GLOBAL (CAMADA COUNTS) ---")
    df_qc = pd.DataFrame({
        'n_genes': all_n_genes,
        'total_counts': all_total_counts,
        'pct_mt': all_pct_mt
    })

    # Imprime estatísticas descritivas (média, min, max, quartis)
    print(df_qc.describe().round(2))

    print("\n--- SIMULAÇÃO DOS SEUS CORTES BIOLÓGICOS (SWeeP) ---")
    # Simula o corte exato que você parametrizou no caderno de laboratório
    filtro = (
        (df_qc['pct_mt'] <= 15) & 
        (df_qc['n_genes'] >= 200) & 
        (df_qc['n_genes'] <= 8000) & 
        (df_qc['total_counts'] >= 500)
    )

    celulas_mantidas = filtro.sum()
    print(f"Total original: {total_cells} células")
    print(f"Células que sobrevivem ao seu QC: {celulas_mantidas} ({(celulas_mantidas/total_cells)*100:.2f}%)")
    print(f"Células descartadas (doublets, lixo ou alta mitocôndria): {total_cells - celulas_mantidas}")

except Exception as e:
    print(f"Erro na exploração: {e}")