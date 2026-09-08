import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
#import sweep as sp
import os
adata_completo = sc.read_h5ad("BIOINFO/analise.pan_etal/GSE267301_Full_Dataset.h5ad")
adata_completo.layers
#PEGAR A CAMADA NUCLEAR_ORIGINAL - é merged da spliced e unspliced
adata_completo.layers 
#nuclear, nuclear_original, spliced, spliced_original, unspliced, unspliced_original
adata = ad.AnnData(
    X = adata_completo.layers['nuclear_original'].copy(),
    obs = adata_completo.obs.copy(),
    var = adata_completo.var.copy()
    )

del adata_completo

print(adata.var.columns)
print(adata.obs.columns)

sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True) # a função sc.pp.calculate_qc_metrics vai calcular as métricas matemáticas de controle de qualidade (QC) para cada célula e cada gene do  conjunto de dados.
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True, save="BRUTO")

#-------------------------------------------------------------------------------------------------------------------------------

celula_saudavel = (adata.obs['pct_counts_mt'] <= 15).sum()
celula_morta = (adata.obs['pct_counts_mt'] > 15).sum()
total = adata.n_obs
print(celula_saudavel)
print(celula_morta)

adata.shape #(61747, 61541)
adata.obs.head()
adata.var.columns

print("reazliando a filtragem de mitocondrial")
adata = adata[adata.obs.pct_counts_mt < 15, :]
adata.shape#>>> adata.shape
(61546, 61541)
porcentagem_mantida = celula_saudavel/total *100

print("--- ANÁLISE DO CORTE MITOCONDRIAL (15% mt) ---")
print(f"Células saudáveis (<= 15%): {celula_saudavel}")
print(f"Células a serem excluídas (> 15%): {celula_morta}")
print(f"Você vai manter {porcentagem_mantida:.2f}% das suas células atuais.")

# ----------------------------RECORTE DOS GENES -----------------------------------------------
min_genes = 200
max_genes = 8000
total = adata.n_obs

adata.obs['n_genes_by_counts'].describe()
print(adata.obs['n_genes_by_counts'] >= 500)

adata.shape#(61546, 61541)

celulas_mantidas = ((adata.obs['n_genes_by_counts'] >= 200) & (adata.obs['n_genes_by_counts'] <= 8000)).sum()
print(celulas_mantidas) 
celulas_excluidas = total - celulas_mantidas
celulas_excluidas 

total
celulas_mantidas

print(f"ANTES do filtro", adata.shape)#(61546, 61541)
adata = adata[adata.obs.n_genes_by_counts < 8000, :]
sc.pp.filter_cells(adata, min_genes=200)
print(f"Depois do filtro:", adata.shape)#(61026, 61541)

porcentagem_mantida = celulas_mantidas/total *100

# 5. Imprime o relatório
print(f"--- ANÁLISE DO CORTE DE GENES ({min_genes} a {max_genes} genes) ---")
print(f"Células dentro do padrão: {celulas_mantidas}")
print(f"Células a serem excluídas (gotículas vazias ou doublets): {celulas_excluidas}")
print(f"Você vai manter {porcentagem_mantida:.2f}% das suas células atuais.")

#----------------------------------------CORTE DE TOTAL COUNTS ---------------------------

#ANÁLISE DE UMIS
min_umi = 0
max_umi = 500

# 3. Calcular a quantidade de células no intervalo de corte
total_celulas = adata.n_obs
celulas_no_intervalo = adata.obs['total_counts'].between(min_umi, max_umi).sum()

porcentagem = (celulas_no_intervalo / total_celulas) * 100

print(f"Total de células: {total_celulas}")
print(f"Células entre {min_umi} e {max_umi} UMIs: {celulas_no_intervalo}")
print(f"Porcentagem: {porcentagem:.2f}%")

# 4. Visualização com Histograma
plt.figure(figsize=(8, 5))
plt.hist(adata.obs['total_counts'], bins=100, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvspan(min_umi, max_umi, color='orange', alpha=0.3, label=f'Intervalo {min_umi}-{max_umi}')
plt.title('Distribuição de UMIs por Célula')
plt.xlabel('Total de UMIs (Counts)')
plt.ylabel('Número de Células')
plt.legend()
plt.savefig('analise_distribuicao_umis.png')

adata.obs['total_counts'].describe()
print(f"ANTES do filtro", adata.shape)
adata = adata[adata.obs['total_counts'] >= 500, :]
print(f"Depois do filtro:", adata.shape)

print(f"Células restantes após o corte de 500 UMIs: {adata.n_obs}")

#======================#==================#=========================#================++#

sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', color='pct_counts_mt', save="scatterTUDO_LIMPO")

sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True, save="LIMPO")

adata.obs[['n_genes_by_counts', 'total_counts', 'pct_counts_mt']].describe()

#============================7
#normalização

# 1. Aplica o log na base 2 (como você tinha escolhido antes)
sc.pp.log1p(adata, base=2)

adata.write_h5ad("GSE267301_filtradoenormalizado.h5ad")


adata = sc.read_h5ad("GSE267301_filtradoenormalizado.h5ad")

# Calculando quantidae de zeros na matriz
total_celulas = adata.n_obs * adata.n_vars
elementos_nao_zero = adata.X.count_nonzero()
porcentagem_zeros = 100 * (1-(elementos_nao_zero/total_celulas))
print(f"porcentagem de zeros na matriz: {porcentagem_zeros:.2f}%")

adata = sc.read_h5ad("BIOINFO/analise.pan_etal/GSE267301_filtradoenormalizado.h5ad")
#EXPORTANDO O ADATA.X
sio.mmwrite("Matriz_Panv2.mtx", adata.X)

adata.X = (adata.X > 0).astype(int)

# 3. Exporta a NOVA matriz binarizada para o R 
sio.mmwrite("Matriz_Pan_Binarizada.mtx", adata.X)
adata.obs.index.to_series().to_csv("barcodes_bin.tsv", sep='\t', index=False, header=False)
adata.var.index.to_series().to_csv("features_bin.tsv", sep='\t', index=False, header=False)

print(f"Matriz binarizada. Novo valor máximo: {adata.X.max()}")
adata.X.shape #(58326, 61541)
#----------------------------- 

import pandas as pd

print("\n--- Processando Rótulos (Cell_Type) ---")

# 1. Carrega o arquivo de anotações manuais
# Assumindo que a primeira coluna é o barcode da célula e a segunda é o tipo celular
anotacoes_df = pd.read_csv("BIOINFO/analise.pan_etal/anotacoes_tecidos.txt", sep='\t', header=None, names=['Barcode', 'cell_type'])

# Transforma em um dicionário para busca rápida: { '6-MTL_WM': 'Oligodendrocytes', ... }
dict_anotacoes = dict(zip(anotacoes_df['Barcode'], anotacoes_df['cell_type']))

# 2. Insere os nomes biológicos no adata garantindo a ordem exata das células
adata.obs['cell_type_text'] = adata.obs.index.map(dict_anotacoes)

# 3. Dicionário de tradução rigoroso (Texto -> Número de 1 a 8)
dicionario_map = {
    'Astrocytes': 1,
    'Oligodendrocytes': 2,
    'Exc. Neurons': 3,
    'Inh. Neurons': 4,
    'OPCs': 5,
    'Microglia': 6,
    'Endothelial': 7,
    'VLMCs': 8
}

# 4. Aplica a tradução numérica
adata.obs['Cell_Type_Num'] = adata.obs['cell_type_text'].map(dicionario_map)

# 5. Diagnóstico de Segurança Crítico
nulos = adata.obs['Cell_Type_Num'].isna().sum()
print(f"Células totais no adata: {adata.n_obs}")
print(f"Células sem rótulo após o mapeamento: {nulos}")

if nulos > 0:
    print("ALERTA: Algumas células não encontraram correspondência no arquivo ou no dicionário.")
    # Mostra quais nomes falharam para facilitar a correção
    print("Tipos não mapeados:", adata.obs[adata.obs['Cell_Type_Num'].isna()]['cell_type_text'].unique())
else:
    # 6. Exportação final isolada
    print("Nenhum erro de rótulo detectado. Exportando Rotulos_Pan_Binarizado.csv...")
    # Converte explicitamente para inteiro (evitando floats como 1.0) antes de salvar
    adata.obs['Cell_Type_Num'] = adata.obs['Cell_Type_Num'].astype(int)
    adata.obs[['Cell_Type_Num']].to_csv("Rotulos_Pan_Binarizado.csv", index=False)
    print("Exportação concluída com sucesso!")

import scipy.io as sio

print("--- Removendo Células Descartadas no QC ---")
# Filtra o objeto mantendo APENAS as células que receberam anotação (onde não é NaN)
adata = adata[~adata.obs['Cell_Type_Num'].isna()].copy()

print(f"Novo shape do adata (apenas células de alta qualidade): {adata.shape}")

# Converte a coluna para números inteiros (agora que não há mais NaNs)
adata.obs['Cell_Type_Num'] = adata.obs['Cell_Type_Num'].astype(int)

print("\n--- Exportando Arquivos Finais Alinhados ---")
# 1. Exporta a Matriz Esparsa (já binarizada e filtrada)
sio.mmwrite("Matriz_Pan_Binarizada_Limpa.mtx", adata.X)

# 2. Exporta os Rótulos Numéricos na mesma ordem
adata.obs[['Cell_Type_Num']].to_csv("Rotulos_Pan_Limpos.csv", index=False)

print("Sucesso absoluto! Matriz e Rótulos prontos.")