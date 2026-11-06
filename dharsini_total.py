"""
Extração, Alinhamento Gênico, Remoção de Vazamento de Doadores e Binarização
do Dataset Externo GSE263468 para projeção no rSWeeP.

Requisitos atendidos:
  1. Mesmo QC estrito do treino (mito < 15%, 200 <= genes < 8000, UMIs >= 500) calculado na camada bruta 'counts'.
  2. Zero imputação e zero deleção de colunas: mantém exatamente os 61.541 genes na ordem exata do treino.
  3. Remove automaticamente os 6 doadores compartilhados com o GSE267301 (Pan et al.).
  4. Mapeia as 8 classes via Major_celltypes + author_annotation e remove os 133 Pericitos.
  5. Binariza (X > 0) sem distorção e exporta em formato Matrix Market (.mtx) + Rótulos + Metadados.
"""

import os
import gc
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.io as sio
from anndata.io import read_elem

# ============================================================================
# 1. CONFIGURAÇÃO DE CAMINHOS
# ============================================================================
CAMINHO_EXTERNO = "/home/gustavo-k-sato/Downloads/GSE263468_processed_data.h5ad"

# Caminhos candidatos para encontrar o dataset de treino (Pan et al.) e as features do treino
CANDIDATOS_PAN_H5AD = [
    "BIOINFO/analise.pan_etal/GSE267301_Full_Dataset.h5ad",
    "/media/gustavo-k-sato/D61CF48E1CF46B3F1/Analise-RNA-seq-Pan_et_al---alzheimer/BIOINFO/analise.pan_etal/GSE267301_Full_Dataset.h5ad",
]

CANDIDATOS_FEATURES = [
    "features_ENSG_puro.tsv",
    "/media/gustavo-k-sato/D61CF48E1CF46B3F1/Analise-RNA-seq-Pan_et_al---alzheimer/features_ENSG_puro.tsv",
    "/media/gustavo-k-sato/D61CF48E1CF46B3F1/Analise-RNA-seq-Pan_et_al---alzheimer/Documents/features_ENSG_puro.tsv",
]

PASTA_SAIDA = "export_GSE263468_SWeeP"
os.makedirs(PASTA_SAIDA, exist_ok=True)

CHUNK = 15000
MT_MAX, GENES_MIN, GENES_MAX, UMI_MIN = 15.0, 200, 8000, 500

# ============================================================================
# 2. FUNÇÕES AUXILIARES
# ============================================================================
def limpar_ensembl(serie):
    return (
        serie.fillna("MISSING")
        .astype(str)
        .str.split(".")
        .str[0]
        .str.strip()
        .str.upper()
    )

def csr_chunk(grupo, i, j, n_genes):
    p0, p1 = int(grupo["indptr"][i]), int(grupo["indptr"][j])
    return sp.csr_matrix(
        (grupo["data"][p0:p1], grupo["indices"][p0:p1], grupo["indptr"][i:j + 1] - p0),
        shape=(j - i, n_genes),
    )

def mapear_8_classes(obs_df):
    """
    Mapeia para os IDs de 1 a 8 idênticos ao treino (Pan et al.):
      1: Astrocytes | 2: Oligodendrocytes | 3: Exc. Neurons | 4: Inh. Neurons
      5: OPCs       | 6: Microglia        | 7: Endothelial  | 8: VLMCs
    Retorna NaN para Pericytes (133 células).
    """
    major = obs_df["Major_celltypes"].astype(str)
    auth = obs_df["author_annotation"].astype(str)

    mapa_direto = {
        "Astrocytes": 1,
        "Oligodendrocytes": 2,
        "Excitatory": 3,
        "Inhibitory": 4,
        "OPC": 5,
        "Microglia": 6,
    }
    rotulo_num = major.map(mapa_direto)

    # Desmembramento cirúrgico da classe 'Vascular' usando author_annotation
    mask_vasc = major == "Vascular"
    rotulo_num = np.where(mask_vasc & (auth == "Endothelial"), 7, rotulo_num)
    rotulo_num = np.where(mask_vasc & (auth == "VLMC"), 8, rotulo_num)
    # Pericytes permanecem como NaN e são filtrados no funil

    return pd.Series(rotulo_num, index=obs_df.index, dtype="float64")

# ============================================================================
# 3. CARREGAMENTO DE METADADOS E AUDITORIA DE DOADORES / GENES
# ============================================================================
print("[1/6] Abrindo GSE263468 e lendo metadados (obs e var)...")
f_ext = h5py.File(CAMINHO_EXTERNO, "r")
obs = read_elem(f_ext["obs"])
var = read_elem(f_ext["var"])
n_cells, n_genes = len(obs), len(var)
print(f"      Dimensões brutas: {n_cells:,} células x {n_genes:,} genes")

# --- 3.1 Identificação dos doadores sobrepostos com o treino (Pan et al.) ---
doadores_treino = set()
caminho_pan_encontrado = None
for cand in CANDIDATOS_PAN_H5AD:
    if os.path.exists(cand):
        caminho_pan_encontrado = cand
        break

if caminho_pan_encontrado:
    with h5py.File(caminho_pan_encontrado, "r") as f_pan:
        obs_pan = read_elem(f_pan["obs"])
        doadores_treino = set(obs_pan["Donor"].astype(str).str.strip().unique())
    print(f"[OK] Dataset de treino localizado ({caminho_pan_encontrado}).")
    print(f"     Doadores no treino ({len(doadores_treino)}): {sorted(doadores_treino)}")
else:
    # Fallback seguro: se o arquivo do Pan não estiver no caminho relativo,
    # mas sabemos pelos logs que os 5 confirmados são 108, 112, 116, 30, 46:
    raise FileNotFoundError(
        "Arquivo GSE267301_Full_Dataset.h5ad não encontrado nos caminhos candidatos. "
        "Ajuste CANDIDATOS_PAN_H5AD no topo do script para filtrar os 6 doadores em comum."
    )

doadores_ext = set(obs["Donor"].astype(str).str.strip().unique())
doadores_sobrepostos = sorted(doadores_treino & doadores_ext)
print(f"[ALERTA DE VAZAMENTO] Doadores em comum entre Treino e Externo ({len(doadores_sobrepostos)}): {doadores_sobrepostos}")

# --- 3.2 Verificação e Alinhamento dos 61.541 Ensembl IDs ---
print("\n[2/6] Verificando alinhamento dos 61.541 Ensembl IDs com o treino...")
genes_ext = limpar_ensembl(var["ENSEMBLE GENE"].reset_index(drop=True))

features_treino = None
for cand in CANDIDATOS_FEATURES:
    if os.path.exists(cand):
        features_treino = limpar_ensembl(pd.read_csv(cand, header=None, sep="\t")[0])
        print(f"[OK] Features do treino carregadas de: {cand} ({len(features_treino):,} genes)")
        break

if features_treino is None and caminho_pan_encontrado:
    print("     'features_ENSG_puro.tsv' não achado; extraindo referência diretamente do GSE267301_Full_Dataset.h5ad...")
    with h5py.File(caminho_pan_encontrado, "r") as f_pan:
        var_pan = read_elem(f_pan["var"])
        col_id = "index1" if "index1" in var_pan.columns else var_pan.columns[0]
        features_treino = limpar_ensembl(var_pan[col_id].reset_index(drop=True))

assert len(features_treino) == n_genes, f"Erro: Treino tem {len(features_treino)} genes, externo tem {n_genes}."
assert set(features_treino) == set(genes_ext), "Erro crítico: Conjunto de Ensembl IDs difere entre treino e externo!"

precisa_reordenar = not np.array_equal(features_treino.values, genes_ext.values)
if not precisa_reordenar:
    print("[OK] Ordem dos 61.541 genes é 100% IDÊNTICA entre treino e coorte externa! Nenhuma permutação necessária.")
    idx_permutacao = None
else:
    print("[AVISO] Conjunto de genes idêntico, mas em ordem diferente. Calculando vetor de permutação...")
    mapa_pos = {g: i for i, g in enumerate(genes_ext.values)}
    idx_permutacao = np.array([mapa_pos[g] for g in features_treino.values], dtype=np.int64)
    print("[OK] Vetor de permutação pronto (nenhuma coluna será apagada).")

# ============================================================================
# 4. VARREDURA EM BLOCOS: CÁLCULO DE QC NA CAMADA 'counts'
# ============================================================================
print("\n[3/6] Calculando métricas de QC na camada bruta 'counts'...")
idx_mt = np.where(var["mt"].astype(bool).values)[0]
print(f"      Genes mitocondriais confirmados: {len(idx_mt)}")

tot_counts = np.zeros(n_cells, dtype=np.int64)
n_genes_detec = np.zeros(n_cells, dtype=np.int64)
pct_mt = np.zeros(n_cells, dtype=np.float64)

grupo_counts = f_ext["layers"]["counts"]

for i in range(0, n_cells, CHUNK):
    j = min(i + CHUNK, n_cells)
    bloco = csr_chunk(grupo_counts, i, j, n_genes)
    
    t = np.asarray(bloco.sum(axis=1)).ravel()
    ng = np.asarray((bloco > 0).sum(axis=1)).ravel()
    mt = np.asarray(bloco[:, idx_mt].sum(axis=1)).ravel()
    
    tot_counts[i:j] = t
    n_genes_detec[i:j] = ng
    pct_mt[i:j] = np.where(t > 0, (mt / np.maximum(t, 1)) * 100.0, 0.0)

# ============================================================================
# 5. FUNIL DE FILTRAGEM (QC + REMOÇÃO DE PERICITOS + REMOÇÃO DE DOADORES TREINO)
# ============================================================================
print("\n[4/6] Aplicando funil de filtragem estrito...")
obs["Cell_Type_Num"] = mapear_8_classes(obs)
obs["Donor_str"] = obs["Donor"].astype(str).str.strip()

m_mito = pct_mt < MT_MAX
m_genes = (n_genes_detec >= GENES_MIN) & (n_genes_detec < GENES_MAX)
m_umi = tot_counts >= UMI_MIN
m_qc = m_mito & m_genes & m_umi

m_rotulo = obs["Cell_Type_Num"].notna().values
m_sem_vazamento = ~obs["Donor_str"].isin(doadores_sobrepostos).values

mask_final = m_qc & m_rotulo & m_sem_vazamento

print(f"  - Total inicial:                          {n_cells:,}")
print(f"  - Após QC (mito<15%, 200<=g<8000, u>=500): {m_qc.sum():,}")
print(f"  - Após remover Pericytes (133 s/ classe): {(m_qc & m_rotulo).sum():,}")
print(f"  - Após remover {len(doadores_sobrepostos)} doadores do treino:        {mask_final.sum():,}")

# Subconjunto Homólogo ao Treino (All Nuclei + 10X-Chr-V3)
mask_homologo = (
    mask_final
    & (obs["SORT"].astype(str) == "All Nuclei").values
    & (obs["Assay"].astype(str) == "10X-Chr-V3").values
)
print(f"  - Subconjunto Homólogo (All Nuclei + 10X-V3): {mask_homologo.sum():,}")
# ============================================================================
# 6. FUNÇÃO DE EXPORTAÇÃO POR STREAMING (ZERO ACÚMULO NA RAM)
# ============================================================================
def exportar_subconjunto_streaming(mask_booleana, prefixo, salvar_mtx=True):
    indices_validos = np.where(mask_booleana)[0]
    n_linhas_final = len(indices_validos)
    print(f"\n[Exportando '{prefixo}'] {n_linhas_final:,} células x {n_genes:,} genes...")

    obs_sub = obs.iloc[indices_validos].copy()
    obs_sub["Cell_Type_Num"] = obs_sub["Cell_Type_Num"].astype(int)

    # 1. Salva Rótulos Alinhados (1 a 8) idêntico ao padrão do treino
    caminho_rot = os.path.join(PASTA_SAIDA, f"Rotulos_{prefixo}.csv")
    obs_sub[["Cell_Type_Num"]].to_csv(caminho_rot, index=False)

    # 2. Salva Metadados Completos (útil para MLP e para Hopfield da Letícia)
    colunas_meta = [
        "Cell_Type_Num", "Major_celltypes", "author_annotation",
        "Donor", "Assay", "SORT", "Brain.Region", "Disease.Group", "Braak.stage"
    ]
    caminho_meta = os.path.join(PASTA_SAIDA, f"Metadados_{prefixo}.csv")
    obs_sub[colunas_meta].to_csv(caminho_meta, index=True)
    print(f"  -> Rótulos e Metadados salvos ({n_linhas_final:,} linhas).")
    print("  -> Distribuição das 8 classes neste dataset:")
    print(obs_sub["Cell_Type_Num"].value_counts().sort_index().to_string())

    # 3. Primeira passada rápida: contar total de não-zeros (nnz) para o cabeçalho do .mtx
    # Como ng_c já tem o número exato de genes > 0 por célula, a soma é instantânea!
    total_nnz = int(n_genes_detec[mask_booleana].sum())
    print(f"  -> Total exato de elementos não-zero (uns): {total_nnz:,}")

    if not salvar_mtx:
        return

    caminho_mtx = os.path.join(PASTA_SAIDA, f"Matriz_{prefixo}_Binarizada.mtx")
    print(f"  -> Iniciando gravação em STREAMING para: {caminho_mtx}")

    # 4. Escrita do .mtx em streaming bloco a bloco (sem usar sp.vstack na RAM)
    with open(caminho_mtx, "w", buffering=16 * 1024 * 1024) as f_mtx:
        # Cabeçalho padrão Matrix Market (coordenadas base 1)
        f_mtx.write("%%MatrixMarket matrix coordinate integer general\n")
        f_mtx.write(f"{n_linhas_final} {n_genes} {total_nnz}\n")

        offset_linha = 1  # Matrix Market usa índice começando em 1
        for i in range(0, n_cells, CHUNK):
            j = min(i + CHUNK, n_cells)
            mask_bloco = mask_booleana[i:j]
            n_aprovadas_bloco = int(mask_bloco.sum())
            if n_aprovadas_bloco == 0:
                continue

            bloco = csr_chunk(grupo_counts, i, j, n_genes)
            bloco_filtrado = bloco[mask_bloco, :]

            if precisa_reordenar:
                bloco_filtrado = bloco_filtrado[:, idx_permutacao]

            # Converte apenas este pequeno bloco para COO para extrair (linha, coluna)
            coo = bloco_filtrado.tocoo()
            # No formato binário, todos os valores não-zero valem exatamente 1
            df_chunk = pd.DataFrame({
                "row": coo.row + offset_linha,
                "col": coo.col + 1,
                "val": 1
            })
            df_chunk.to_csv(f_mtx, sep=" ", header=False, index=False)

            offset_linha += n_aprovadas_bloco
            del bloco, bloco_filtrado, coo, df_chunk
            gc.collect()
            print(f"     Streaming .mtx: {offset_linha - 1:,}/{n_linhas_final:,} células gravadas...", end="\r")

    print(f"\n  -> [OK] Matriz .mtx finalizada com sucesso: {caminho_mtx}")

# Executa para o subconjunto Homólogo (~85k células, rápido para testar no R)
print("\n[5/6] Exportando Coorte Externa Homóloga (All Nuclei + 10X-Chr-V3)...")
exportar_subconjunto_streaming(mask_homologo, "GSE263468_Homologo_10XV3_AllNuclei", salvar_mtx=True)

# Executa para o Dataset Geral Completo (~360k células, todas as químicas, sem doadores do treino)
print("\n[6/6] Exportando Dataset Geral Completo (Todas as químicas, Sem Vazamento)...")
exportar_subconjunto_streaming(mask_final, "GSE263468_Geral_SemVazamento", salvar_mtx=True)

f_ext.close()
print("\n======================================================================")
print("PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
print("======================================================================")