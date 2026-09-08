import scanpy as sc
import pandas as pd
import numpy as np


#=========================#===================#==========================================#
import pandas as pd
import scanpy as sc

print("1. Carregando a Pedra de Roseta (features.tsv)...")
df_roseta = pd.read_csv('features(1).tsv', sep='\t', header=None)

# Cria um dicionário de tradução: {ENSG: Gene Symbol}
# Ex: {'ENSG00000243485': 'MIR1302-10'}
dict_ensg_to_symbol = dict(zip(df_roseta[0], df_roseta[1]))

print("2. Carregando Features Originais do PAN...")
features_pan_raw = pd.read_csv("features_bin.tsv", sep='\t', header=None)[0].tolist()

print("3. Traduzindo o PAN para a nomenclatura do Matisse...")
features_pan_traduzidas = []

for gene in features_pan_raw:
    # Se o gene começar com ENSG, precisamos traduzir
    if str(gene).startswith("ENSG"):
        # Corta a string no ponto para remover a versão (ex: ENSG... .1 -> ENSG...)
        ensg_base = gene.split('.')[0]
        
        # Procura no dicionário. Se não achar, devolve o gene original
        novo_nome = dict_ensg_to_symbol.get(ensg_base, gene)
        features_pan_traduzidas.append(novo_nome)
    else:
        # Se já for um nome comum (Gene Symbol), mantém como está
        features_pan_traduzidas.append(gene)

print("4. Carregando o Matisse...")
adata_matisse = sc.read_h5ad("matrizFiltradaeNormalizadaM.h5ad", backed='r')
matisse_genes = adata_matisse.var_names.tolist()

print("\n5. Nova Intersecção (Cruzamento Traduzido)...")
genes_manter = set(matisse_genes).intersection(features_pan_traduzidas)
genes_ausentes = set(features_pan_traduzidas) - set(matisse_genes)

print(f" -> Genes no PAN: {len(features_pan_traduzidas)}")
print(f" -> Genes no Matisse: {len(matisse_genes)}")
print(f" -> NOVO Total de Genes em Comum: {len(genes_manter)}")
print(f" -> Genes Exclusivos do PAN (Perdidos): {len(genes_ausentes)}")

import pandas as pd
import scanpy as sc

print("1. Carregando e HIGIENIZANDO o Dicionário...")
df_roseta = pd.read_csv('features(1).tsv', sep='\t', header=None)

# Limpa espaços nas bordas e força MAIÚSCULO nas duas colunas
df_roseta[0] = df_roseta[0].astype(str).str.strip().str.upper()
df_roseta[1] = df_roseta[1].astype(str).str.strip().str.upper()

dict_ensg_to_symbol = dict(zip(df_roseta[0], df_roseta[1]))

print("2. Carregando e HIGIENIZANDO o PAN...")
features_pan_raw = pd.read_csv("features_bin.tsv", sep='\t', header=None)[0].tolist()

features_pan_traduzidas = []

for gene in features_pan_raw:
    # Higieniza o gene original
    g_clean = str(gene).strip().upper()
    
    if g_clean.startswith("ENSG"):
        ensg_base = g_clean.split('.')[0]
        # Procura no dicionário. Se não achar, mantém a base já limpa.
        novo_nome = dict_ensg_to_symbol.get(ensg_base, g_clean)
        features_pan_traduzidas.append(novo_nome)
    else:
        features_pan_traduzidas.append(g_clean)

print("3. Carregando e HIGIENIZANDO o Matisse...")
adata_matisse = sc.read_h5ad("matrizFiltradaeNormalizadaM.h5ad", backed='r')
# Força maiúsculo e limpa os genes do Matisse também
matisse_genes = [str(g).strip().upper() for g in adata_matisse.var_names]

print("\n4. AUDITORIA DE COLISÕES (DATA LEAKAGE METODOLÓGICO)...")
pan_total = len(features_pan_traduzidas)
pan_unicos = len(set(features_pan_traduzidas))
colisoes = pan_total - pan_unicos

print(f" -> PAN Original: {pan_total} colunas")
print(f" -> PAN Únicos após tradução: {pan_unicos} colunas")
if colisoes > 0:
    print(f" [ALERTA] A tradução gerou {colisoes} genes duplicados no PAN!")

print("\n5. INTERSECÇÃO BLINDADA...")
genes_manter = set(matisse_genes).intersection(set(features_pan_traduzidas))

print(f" -> NOVO Total de Genes em Comum: {len(genes_manter)}")


import pandas as pd
df = pd.read_csv('features(1).tsv', sep='\t', header=None)
print(df.head(20))
features_pan = pd.read_csv("features_bin.tsv", sep='\t', header=None)
print(features_pan.head(20))

print("--- Como ficou a lista NOVA na memória (strip + tradução) ---")
for i in range(50):
    print(f"Index {i}: {features_pan_traduzidas[i]}")

print("1. Carregando Matisse e Features do Modelo Ouro (PAN)...")
# Ajuste o nome do arquivo se necessário
adata_matisse = sc.read_h5ad("matrizFiltradaeNormalizadaM.h5ad", backed='r')
features_pan = pd.read_csv("features_bin.tsv", sep='\t', header=None)[0].tolist()

matisse_genes = adata_matisse.var_names.tolist(
    
)
num_celulas = adata_matisse.n_obs
num_genes_pan = len(features_pan)

print("\n2. Cruzando informações dos Datasets...")
pan_idx_map = {gene: idx for idx, gene in enumerate(features_pan)}

genes_manter = set(matisse_genes).intersection(features_pan)
genes_ausentes = set(features_pan) - set(matisse_genes)import scanpy as sc
import pandas as pd
import numpy as np

print("1. Carregando Matisse e Features do Modelo Ouro (PAN)...")
# Ajuste o nome do arquivo se necessário
adata_matisse = sc.read_h5ad("matrizFiltradaeNormalizadaM.h5ad", backed='r')
features_pan = pd.read_csv("features_bin.tsv", sep='\t', header=None)[0].tolist()

matisse_genes = adata_matisse.var_names.tolist(
    
)
num_celulas = adata_matisse.n_obs
num_genes_pan = len(features_pan)

print("\n2. Cruzando informações dos Datasets...")
pan_idx_map = {gene: idx for idx, gene in enumerate(features_pan)}

genes_manter = set(matisse_genes).intersection(features_pan)
genes_ausentes = set(features_pan) - set(matisse_genes)

# Isola os índices (0-based) das colunas que não existem no Matisse
indices_ausentes = [pan_idx_map[g] for g in genes_ausentes]

print(f" -> Células no Matisse: {num_celulas}")
print(f" -> Genes no PAN: {num_genes_pan}")
print(f" -> Genes no Matisse: {len(matisse_genes)}")
print(f" -> Genes em comum (Dados Reais): {len(genes_manter)}")
print(f" -> Genes Exclusivos do PAN (Receberão 0.5): {len(genes_ausentes)}")

print("\n3. Extraindo Dados Biológicos Reais...")
X_matisse = adata_matisse.to_memory().X.tocoo()

old_to_new_col = {i: pan_idx_map[g] for i, g in enumerate(matisse_genes) if g in genes_manter}

mask = np.isin(X_matisse.col, list(old_to_new_col.keys()))
linhas_reais = X_matisse.row[mask]
colunas_old = X_matisse.col[mask]
dados_reais = X_matisse.data[mask]

mapper = np.zeros(max(colunas_old.max() + 1, len(matisse_genes)), dtype=np.int32)
for old, new in old_to_new_col.items():
    mapper[old] = new
colunas_reais = mapper[colunas_old]

print("\n4. Calculando Volume Total da Matriz...")
total_real = len(dados_reais)
total_05 = len(genes_ausentes) * num_celulas
total_entradas = total_real + total_05

print(f" -> Fatos Reais: {total_real} entradas")
print(f" -> Ruído (0.5) Injetado: {total_05} entradas")
print(f" -> Total a ser escrito no disco: {total_entradas} linhas!")

print("\n5. Iniciando Streaming para o HD (Isso pode demorar alguns minutos)...")
nome_saida = "Matriz_Matisse_Alinhada_05.mtx"

with open(nome_saida, "w") as f:
    # 5.1 Escreve o Cabeçalho Oficial do formato MatrixMarket
    f.write("%%MatrixMarket matrix coordinate real general\n")
    f.write(f"{num_celulas} {num_genes_pan} {total_entradas}\n")
    
    # 5.2 Escreve os Dados Biológicos Reais (Somando +1 porque .mtx não usa base 0)
    print("   -> Escrevendo dados biológicos...")
    for r, c, d in zip(linhas_reais, colunas_reais, dados_reais):
        f.write(f"{r+1} {c+1} {d}\n")
        
    # 5.3 Escreve o 0.5 nos genes ausentes em lotes rápidos
    print("   -> Injetando 0.5 nos genes ausentes...")
    for i, col_idx in enumerate(indices_ausentes):
        col_mtx = col_idx + 1 # Ajuste para base 1
        
        # Cria um bloco de texto com o 0.5 para todas as células desta coluna
        bloco = [f"{r+1} {col_mtx} 0.5\n" for r in range(num_celulas)]
        f.writelines(bloco)
        
        if (i + 1) % 5000 == 0:
            print(f"      [{i+1}/{len(genes_ausentes)}] genes preenchidos com 0.5...")

print(f"\n[SUCESSO] Operação monumental concluída! Matriz pesada salva como '{nome_saida}'.")

# Isola os índices (0-based) das colunas que não existem no Matisse
indices_ausentes = [pan_idx_map[g] for g in genes_ausentes]

print(f" -> Células no Matisse: {num_celulas}")
print(f" -> Genes no PAN: {num_genes_pan}")
print(f" -> Genes no Matisse: {len(matisse_genes)}")
print(f" -> Genes em comum (Dados Reais): {len(genes_manter)}")
print(f" -> Genes Exclusivos do PAN (Receberão 0.5): {len(genes_ausentes)}")

print("\n3. Extraindo Dados Biológicos Reais...")
X_matisse = adata_matisse.to_memory().X.tocoo()

old_to_new_col = {i: pan_idx_map[g] for i, g in enumerate(matisse_genes) if g in genes_manter}

mask = np.isin(X_matisse.col, list(old_to_new_col.keys()))
linhas_reais = X_matisse.row[mask]
colunas_old = X_matisse.col[mask]
dados_reais = X_matisse.data[mask]

mapper = np.zeros(max(colunas_old.max() + 1, len(matisse_genes)), dtype=np.int32)
for old, new in old_to_new_col.items():
    mapper[old] = new
colunas_reais = mapper[colunas_old]

print("\n4. Calculando Volume Total da Matriz...")
total_real = len(dados_reais)
total_05 = len(genes_ausentes) * num_celulas
total_entradas = total_real + total_05

print(f" -> Fatos Reais: {total_real} entradas")
print(f" -> Ruído (0.5) Injetado: {total_05} entradas")
print(f" -> Total a ser escrito no disco: {total_entradas} linhas!")

print("\n5. Iniciando Streaming para o HD (Isso pode demorar alguns minutos)...")
nome_saida = "Matriz_Matisse_Alinhada_05.mtx"

with open(nome_saida, "w") as f:
    # 5.1 Escreve o Cabeçalho Oficial do formato MatrixMarket
    f.write("%%MatrixMarket matrix coordinate real general\n")
    f.write(f"{num_celulas} {num_genes_pan} {total_entradas}\n")
    
    # 5.2 Escreve os Dados Biológicos Reais (Somando +1 porque .mtx não usa base 0)
    print("   -> Escrevendo dados biológicos...")
    for r, c, d in zip(linhas_reais, colunas_reais, dados_reais):
        f.write(f"{r+1} {c+1} {d}\n")
        
    # 5.3 Escreve o 0.5 nos genes ausentes em lotes rápidos
    print("   -> Injetando 0.5 nos genes ausentes...")
    for i, col_idx in enumerate(indices_ausentes):
        col_mtx = col_idx + 1 # Ajuste para base 1
        
        # Cria um bloco de texto com o 0.5 para todas as células desta coluna
        bloco = [f"{r+1} {col_mtx} 0.5\n" for r in range(num_celulas)]
        f.writelines(bloco)
        
        if (i + 1) % 5000 == 0:
            print(f"      [{i+1}/{len(genes_ausentes)}] genes preenchidos com 0.5...")

print(f"\n[SUCESSO] Operação monumental concluída! Matriz pesada salva como '{nome_saida}'.")