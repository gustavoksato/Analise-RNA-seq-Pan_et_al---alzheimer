import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from anndata.io import read_elem
import scipy.io as sio
import gc

caminho_h5ad = "/home/gustavo-k-sato/Downloads/GSE263468_processed_data.h5ad"
chunk_size = 10000

print("1. Abrindo metadados e verificando 'gene_id'...")
with h5py.File(caminho_h5ad, 'r') as f:
    obs = read_elem(f['obs'])
    var = read_elem(f['var'])
    
    print(f"Colunas disponíveis em 'var' (Genes): {var.columns.tolist()}")
    
    # Checagem dinâmica do ID Gênico
    if 'gene_id' in var.columns:
        print("-> SUCESSO: 'gene_id' encontrado! Ele será usado para a exportação das features.")
        features_export = var['gene_id']
    elif 'ensembl_id' in var.columns:
        print("-> SUCESSO: 'ensembl_id' encontrado! Ele será usado para a exportação das features.")
        features_export = var['ensembl_id']
    else:
        print("-> AVISO: 'gene_id' não detectado nas colunas. O index principal será exportado.")
        features_export = var.index.to_series()

    # Dicionário de tradução ontológica
    # IMPORTANTE: Inspecione as strings exatas da coluna 'Major_celltypes' do dataset
    # e ajuste as chaves deste dicionário caso os autores tenham escrito diferente (ex: 'Excitatory' vs 'Exc. Neurons').
    dicionario_map = {
        'Astrocytes': 1, 'Oligodendrocytes': 2, 'Exc. Neurons': 3, 
        'Inh. Neurons': 4, 'OPCs': 5, 'Microglia': 6, 
        'Endothelial': 7, 'VLMCs': 8
    }
    
    # Mapeamento e criação da coluna numérica de labels
    obs['Cell_Type_Num'] = obs['Major_celltypes'].map(dicionario_map)

    # Identificando genes mitocondriais via index para o QC
    mito_genes = var.index.str.upper().str.startswith('MT-')
    
    total_cells = obs.shape[0]
    total_genes = var.shape[0]
    
    valid_sparse_chunks = []
    valid_obs_list = []

    print("\n2. Iniciando Binarização e Filtragem em Streaming (Camada: counts)...")
    counts_layer = f['layers']['counts']
    dset_data = counts_layer['data']
    dset_indices = counts_layer['indices']
    dset_indptr = counts_layer['indptr']

    for i in range(0, total_cells, chunk_size):
        end = min(i + chunk_size, total_cells)
        ptr_start = dset_indptr[i]
        ptr_end = dset_indptr[end]
        
        # Reconstrói bloco esparso na RAM
        chunk_X = sparse.csr_matrix(
            (dset_data[ptr_start:ptr_end], 
             dset_indices[ptr_start:ptr_end], 
             dset_indptr[i:end+1] - ptr_start), 
            shape=(end - i, total_genes)
        )
        
        # Puxa o pedaço correspondente dos metadados
        chunk_obs = obs.iloc[i:end].copy()
        
        # QC do Lote
        total_counts = np.sum(chunk_X, axis=1).A1
        mt_counts = np.sum(chunk_X[:, mito_genes], axis=1).A1
        n_genes = np.sum(chunk_X > 0, axis=1).A1
        
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_mt = np.nan_to_num((mt_counts / total_counts) * 100)
            
        chunk_obs['pct_counts_mt'] = pct_mt
        chunk_obs['n_genes_by_counts'] = n_genes
        chunk_obs['total_counts'] = total_counts
        
        # Filtro Biológico e Exclusão de células sem correspondência de rótulo (NaN)
        mask_qc = (
            (chunk_obs['pct_counts_mt'] <= 15) & 
            (chunk_obs['n_genes_by_counts'] >= 200) & 
            (chunk_obs['n_genes_by_counts'] <= 8000) & 
            (chunk_obs['total_counts'] >= 500) &
            (~chunk_obs['Cell_Type_Num'].isna())
        )
        
        chunk_obs_filtrado = chunk_obs[mask_qc].copy()
        chunk_X_filtrado = chunk_X[mask_qc.values, :]
        
        # Binarização topológica In-Place
        if chunk_X_filtrado.shape[0] > 0:
            chunk_X_filtrado.data = (chunk_X_filtrado.data > 0).astype(np.int8)
            
            valid_sparse_chunks.append(chunk_X_filtrado)
            valid_obs_list.append(chunk_obs_filtrado)

        # Higienização de RAM
        del chunk_X, chunk_obs, chunk_X_filtrado, chunk_obs_filtrado
        gc.collect()
        
        if end % 50000 == 0 or end == total_cells:
            print(f"  -> Processadas {end}/{total_cells} células...")

print("\n3. Reconstrução e Exportação Final...")
X_final = sparse.vstack(valid_sparse_chunks)
obs_final = pd.concat(valid_obs_list)
obs_final['Cell_Type_Num'] = obs_final['Cell_Type_Num'].astype(int)

print(f"-> Células sobreviventes de alta qualidade e rotuladas: {X_final.shape[0]}")
print(f"-> Shape da Matriz Ouro Final: {X_final.shape}")

print("-> Exportando Matriz .mtx...")
sio.mmwrite("Matriz_Artigo_B inarizada_Limpa.mtx", X_final)

print("-> Exportando Rótulos e Metadados...")
obs_final[['Cell_Type_Num']].to_csv("Rotulos_Artigo_Limpos.csv", index=False)
obs_final.index.to_series().to_csv("barcodes_artigo_limpo.tsv", sep='\t', index=False, header=False)
features_export.to_csv("features_artigo_limpo.tsv", sep='\t', index=False, header=False)

print("Extração, binarização e exportação concluídas com sucesso!")