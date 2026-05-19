import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
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

sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True, save="BRUTO")
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True) # a função sc.pp.calculate_qc_metrics vai calcular as métricas matemáticas de controle de qualidade (QC) para cada célula e cada gene do  conjunto de dados.
#-------------------------------------------------------------------------------------------------------------------------------

celula_saudavel = (adata.obs['pct_counts_mt'] <= 15).sum()
celula_morta = (adata.obs['pct_counts_mt'] > 15).sum()
total = adata.n_obs
print(celula_saudavel)
print(celula_morta)

adata.shape #(61747, 61541)
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

sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', color='pct_counts_mt', save="scatterTUDO_LIMPO")

sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True, save="LIMPO")

adata.obs['n_genes_by_counts', 'total_counts', 'pct_counts_mt'].describe()
adata.obs[['n_genes_by_counts', 'total_counts', 'pct_counts_mt']].describe()

#------------------------------CHECAGEM 5K FUJITA ---------------------------------


listafujita_limpa = []
genes_oficial= []
genes_fujita = pd.read_csv("BIOINFO/analise.pan_etal/features.tsv.gz")
print(genes_fujita[:10]) 

#LISTAFUJITA
listafujita = genes_fujita.iloc[:, 0].tolist()

#LISTAFUJITA LIMPA
for gene in listafujita:
    gene_limpo = gene.split('\t')[0]                
    listafujita_limpa.append(gene_limpo)

#SALVANDO CODIGOS ORIGINAIS
codigos_original = []
for gene in adata.var['index1']:
    codigos_original.append(gene)

print(codigos_original[:10])
#ESCREVENDO O TXT DOS CÓDIGOS ORIGINAIS
with open("códigos_originais_Pan.txt", "w") as f:
    for gene in codigos_original: 
        f.write(f"{gene}\n")

#REESCREVENDO A COLUNA INDEX1

adata.var['index1'] = adata.var['index1'].str.split('.').str[0]
print(adata.var['index1'].values[:10])

for gene in listafujita_limpa:
    if gene in adata.var['index1']:
        genes_oficial.append(gene)

print(genes_oficial[:10])
len(genes_oficial)

genes5k = adata.var.nlargest(5000, 'total_counts')['index1']

genes5K = adata.var.sort_values(by='total_counts', ascending = False)['index1'].head(5000)

len(genes5k)

top5000_var['total_counts'].describe()

#MONTANDO TXT COM OS 5K MAIS FREQ E AS COUNTS TOTAIS

top5000_var = adata.var.nlargest(5000, 'total_counts')
top5000_var[['index1', 'total_counts']].to_csv("5kfujita-Pan", sep="\t", index=False)


tem_decimais = (adata.var['total_counts'] % 1 != 0).any()
print(tem_decimais)






