
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

adata_completo = sc.read_h5ad("/media/gustavo-k-sato/D61CF48E1CF46B3F1/Analise-RNA-seq-Pan_et_al---alzheimer/Documents/BIOINFO/analise.pan_etal/GSE267301_filtradoenormalizado.h5ad")

adata = adata_completo


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
adata.shape
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True) # a função sc.pp.calculate_qc_metrics vai calcular as métricas matemáticas de controle de qualidade (QC) para cada célula e cada gene do  conjunto de dados.
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True, save="BRUTO")

adata = sc.read_h5ad("segundodataset.h5ad")


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








DONOR = "Donor"
GRUPO = "Case_No_Case"            # rótulo caso/controle
CANDIDATAS = ["Case_No_Case", "NP.Diagnosis", "Braak.Group", "Braak.stage",
              "Amyloid", "APOE", "Sex", "Brain.Region", "Gray.vs.White", "SORT"]
ANOTACAO = "manual_annotation"


def _sec(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def explorar(obs):
    obs = obs.copy()
    obs.columns = obs.columns.str.strip()   # ' Braak.Group' tem espaço no início
    for c in obs.select_dtypes("category").columns:
        obs[c] = obs[c].astype(object)

    _sec("1. Visão geral")
    print(f"Células: {len(obs):,} | Doadores: {obs[DONOR].nunique()}")
    if "Unique.identifier" in obs:
        print(f"Amostras (Unique.identifier): {obs['Unique.identifier'].nunique()}")

    _sec("2. Células e doadores por categoria")
    for c in [c for c in CANDIDATAS if c in obs]:
        t = pd.DataFrame({
            "celulas": obs[c].value_counts(dropna=False),
            "doadores": obs.groupby(c, dropna=False)[DONOR].nunique(),
        })
        print(f"\n-- {c}\n{t}")

    _sec("3. Atributos constantes dentro do doador? (n de doadores com >1 valor)")
    cols = [c for c in CANDIDATAS + ["Age", "PMI.hr.", "Race", "RIN", "Brain.weight"]
            if c in obs]
    inconsist = {c: int((obs.groupby(DONOR)[c].nunique(dropna=False) > 1).sum())
                 for c in cols}
    print(pd.Series(inconsist, name="doadores_inconsistentes"))
    print("(0 = atributo de doador; >0 = varia por amostra/região ou há erro de rótulo)")

    _sec("4. Tabela por doador")
    donor_cols = [c for c in cols if inconsist[c] == 0]
    tab = obs.groupby(DONOR).agg({**{c: "first" for c in donor_cols}})
    tab["n_celulas"] = obs.groupby(DONOR).size()
    if "Brain.Region" in obs:
        tab["n_regioes"] = obs.groupby(DONOR)["Brain.Region"].nunique()
    print(tab.to_string())

    _sec("5. Cruzamentos no nível do doador")
    if GRUPO in tab:
        for c in ["NP.Diagnosis", "Braak.Group", "Braak.stage", "Amyloid", "APOE", "Sex"]:
            if c in tab:
                print(f"\n-- {GRUPO} x {c}\n{pd.crosstab(tab[GRUPO], tab[c], dropna=False)}")
        for c in ["Age", "PMI.hr."]:
            if c in tab:
                print(f"\n-- {c} por grupo\n"
                      f"{pd.to_numeric(tab[c], errors='coerce').groupby(tab[GRUPO]).describe()}")
        print(f"\n-- células por doador, por grupo\n{tab.groupby(GRUPO)['n_celulas'].describe()}")

    if "Brain.Region" in obs:
        _sec("6. Doador x região (n de células)")
        print(pd.crosstab(obs[DONOR], obs["Brain.Region"]).to_string())

    if ANOTACAO in obs and GRUPO in obs:
        _sec("7. Tipos celulares x grupo (células) e doadores com >=50 células por tipo")
        print(pd.crosstab(obs[ANOTACAO], obs[GRUPO], margins=True))
        por_doador = pd.crosstab(obs[DONOR], obs[ANOTACAO])
        print("\nDoadores com >=50 células em cada tipo:")
        print((por_doador >= 50).sum().rename("doadores"))

    return tab

explorar()

tab = explorar(adata.obs)