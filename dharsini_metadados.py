"""
Checagem de METADADOS do GSE263468 (somente obs; nenhuma camada da matriz é lida, roda em segundos).

Camadas de matriz relevantes (já verificadas no diagnóstico anterior):
  - 'counts'  == 'spliced' + 'unspliced'  (equivale ao nuclear_original do treino)
  - 'normalized' preserva o padrão de zeros de 'counts' (a binarização é a mesma)
Aqui o foco é o que decide se o dataset serve como coorte externa:
  1. esquema de obs
  2. rótulos: Major_celltypes, author_annotation, cell_type_ontology_term_id (e o que é 'Vascular')
  3. qualidade já aplicada pelos autores (doublets, droplet_FDR, sorting, ensaio) e consistência com o nosso QC
  4. estrutura doador > amostra > região, e variáveis constantes por doador
  5. composição das classes por região, grupo de doença, sorting
  6. nº de doadores por classe com >= 50/100/200 células (viabilidade do leave-one-donor-out)
  7. sobreposição de identificadores e barcodes com outras coortes (independência)

Saídas (na pasta do script):
  metadados_GSE263468.txt
  metadados_GSE263468_doadores.csv
  metadados_GSE263468_classe_por_doador.csv

Uso:  /usr/bin/python3 metadados_GSE263468.py
"""
import os
import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem

# ----------------------------------------------------------------------------
# CONFIGURAÇÃO
# ----------------------------------------------------------------------------
CAMINHO_H5AD = "/home/gustavo-k-sato/Downloads/GSE263468_processed_data.h5ad"
PASTA = os.path.dirname(os.path.abspath(__file__))
# CSV gerado pelo diagnóstico anterior (opcional); procura na pasta do script e na pasta acima
_CANDS_CSV = [os.path.join(PASTA, "diagnostico_GSE263468_qc.csv"),
              os.path.join(os.path.dirname(PASTA), "diagnostico_GSE263468_qc.csv")]
CSV_QC = next((c for c in _CANDS_CSV if os.path.exists(c)), _CANDS_CSV[0])

# coortes a comparar para independência (nome: lista de caminhos possíveis do .h5ad)
COORTES_COMPARACAO = {
    "Pan (GSE267301)": [
        "BIOINFO/analise.pan_etal/GSE267301_Full_Dataset.h5ad",
        os.path.join(PASTA, "BIOINFO/analise.pan_etal/GSE267301_Full_Dataset.h5ad"),
        os.path.join(PASTA, "GSE267301_Full_Dataset.h5ad"),
    ],
    # "Mathys": ["/caminho/completo/para/mathys.h5ad"],
}

COL_CLASSE, COL_FINA, COL_ONTO = "Major_celltypes", "author_annotation", "cell_type_ontology_term_id"
COL_DOADOR, COL_AMOSTRA, COL_REGIAO = "Donor", "sample", "Brain.Region"
LIMIARES = [10, 20, 50, 100, 200]

SAIDA_TXT = os.path.join(PASTA, "metadados_GSE263468.txt")
SAIDA_DOADORES = os.path.join(PASTA, "metadados_GSE263468_doadores.csv")
SAIDA_CLASSE_DOADOR = os.path.join(PASTA, "metadados_GSE263468_classe_por_doador.csv")

# ----------------------------------------------------------------------------
# UTILITÁRIOS
# ----------------------------------------------------------------------------
_arq = open(SAIDA_TXT, "w", encoding="utf-8")
CHECKLIST = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _arq.write(s + "\n")
    _arq.flush()


def titulo(t):
    log("\n" + "=" * 72)
    log(t)
    log("=" * 72)


def flag(status, msg):
    CHECKLIST.append((status, msg))
    log(f"[{status}] {msg}")


def vc(serie, n=40):
    v = serie.astype(str).value_counts(dropna=False)
    s = v.head(n).to_string()
    if len(v) > n:
        s += f"\n... (+{len(v) - n} valores)"
    return s


if not os.path.exists(CAMINHO_H5AD):
    raise SystemExit(f"Arquivo não encontrado: {CAMINHO_H5AD}")
with h5py.File(CAMINHO_H5AD, "r") as f:
    obs = read_elem(f["obs"])
n_cells = len(obs)
tem = lambda c: c in obs.columns

# ----------------------------------------------------------------------------
# 1. ESQUEMA
# ----------------------------------------------------------------------------
titulo("1. ESQUEMA DE obs")
log(f"{n_cells} células | {obs.shape[1]} colunas")
esq = pd.DataFrame({"dtype": obs.dtypes.astype(str), "valores_unicos": obs.nunique(dropna=False),
                    "nulos": obs.isna().sum()})
log(esq.to_string())

# ----------------------------------------------------------------------------
# 2. RÓTULOS
# ----------------------------------------------------------------------------
titulo("2. RÓTULOS CELULARES")
for c in [COL_CLASSE, COL_FINA, COL_ONTO]:
    if tem(c):
        log(f"\n-- {c} --\n" + vc(obs[c], 80))
    else:
        flag("AVISO", f"Coluna '{c}' ausente.")

if tem(COL_CLASSE) and tem(COL_FINA):
    log(f"\n-- {COL_FINA} dentro de cada categoria de {COL_CLASSE} (até 25 por categoria) --")
    for cls in obs[COL_CLASSE].astype(str).unique():
        sub = obs.loc[obs[COL_CLASSE].astype(str) == cls, COL_FINA]
        log(f"\n[{cls}] ({len(sub)} células)\n" + vc(sub, 25))

if tem(COL_CLASSE) and tem(COL_ONTO):
    log(f"\n-- {COL_ONTO} dentro de cada categoria de {COL_CLASSE} --")
    for cls in obs[COL_CLASSE].astype(str).unique():
        sub = obs.loc[obs[COL_CLASSE].astype(str) == cls, COL_ONTO]
        log(f"\n[{cls}]\n" + vc(sub, 10))

titulo("2b. 'Vascular' pode ser separado em Endothelial e VLMCs?")
if tem(COL_FINA):
    nomes = obs[COL_FINA].astype(str).value_counts()
    grupos = {
        "tipo endotelial": ["endo"],
        "tipo VLMC/meníngeo/fibroblasto": ["vlmc", "leptomen", "mening", "fibro"],
        "pericito/músculo liso/mural": ["peric", "smc", "smooth", "mural"],
        "vascular genérico": ["vascular"],
    }
    achou = {}
    for g, ps in grupos.items():
        sel = [n for n in nomes.index if any(p in n.lower() for p in ps)]
        achou[g] = sel
        log(f"{g}: {[(n, int(nomes[n])) for n in sel] if sel else 'nenhuma categoria'}")
    if achou["tipo endotelial"] and achou["tipo VLMC/meníngeo/fibroblasto"]:
        n_end = int(sum(nomes[n] for n in achou["tipo endotelial"]))
        n_vlmc = int(sum(nomes[n] for n in achou["tipo VLMC/meníngeo/fibroblasto"]))
        suficiente = min(n_end, n_vlmc) >= 500
        flag("OK" if suficiente else "AVISO",
             f"{COL_FINA} separa endotélio ({n_end} células) e VLMC ({n_vlmc}); "
             + ("contagens suficientes." if suficiente else
                "contagens muito baixas para F1 por classe na validação externa."))
    else:
        flag("AVISO", f"{COL_FINA} não separa claramente endotélio de VLMC: avaliação externa com 7 classes (Vascular fundida).")
else:
    flag("AVISO", f"Sem {COL_FINA}: Vascular não pode ser separada.")

# ----------------------------------------------------------------------------
# 3. QUALIDADE JÁ APLICADA PELOS AUTORES
# ----------------------------------------------------------------------------
titulo("3. QUALIDADE E ENSAIO")
num = [c for c in ["total_counts", "n_genes", "pct_counts_mt", "droplet_FDR"] if tem(c)]
if num:
    log(obs[num].apply(pd.to_numeric, errors="coerce").describe(percentiles=[.01, .25, .5, .75, .99]).to_string())
for c in ["doublefinder", "SORT", "Assay", "suspension_type", "is_primary_data", "Brain.weight", "Amyloid", "APOE_class"]:
    if tem(c):
        if obs[c].nunique(dropna=False) <= 15:
            log(f"\n-- {c} --\n" + vc(obs[c], 15))
        else:
            log(f"\n-- {c}: {obs[c].nunique()} valores distintos")

titulo("3b. Consistência entre as métricas dos autores e as calculadas por nós (counts)")
if os.path.exists(CSV_QC) and all(tem(c) for c in ["total_counts", "n_genes", "pct_counts_mt"]):
    primeira = pd.read_csv(CSV_QC, nrows=0).columns[0]
    q = pd.read_csv(CSV_QC, usecols=[primeira, "tot_c", "ng_c", "pct_c"], index_col=0)
    q.index = q.index.astype(str)
    a = obs[["total_counts", "n_genes", "pct_counts_mt"]].apply(pd.to_numeric, errors="coerce")
    a.index = a.index.astype(str)
    j = q.join(a, how="inner")
    log(f"Células pareadas por barcode: {len(j)} de {len(q)}")
    if len(j):
        for ours, theirs in [("tot_c", "total_counts"), ("ng_c", "n_genes"), ("pct_c", "pct_counts_mt")]:
            d = (j[ours] - j[theirs]).abs()
            log(f"{ours} vs {theirs}: diferença máxima={d.max():.4g} | mediana={d.median():.4g} | "
                f"correlação={j[ours].corr(j[theirs]):.6f}")
        ok = (j["tot_c"] - j["total_counts"]).abs().max() < 1e-6
        flag("OK" if ok else "AVISO",
             "total_counts dos autores == soma da camada counts." if ok else
             "total_counts dos autores difere da soma de counts: verificar de onde vêm as métricas deles.")
else:
    log(f"CSV de QC não encontrado em {CSV_QC} (rode antes o diagnóstico anterior) ou colunas ausentes.")

# ----------------------------------------------------------------------------
# 4. DOADOR > AMOSTRA > REGIÃO
# ----------------------------------------------------------------------------
titulo("4. ESTRUTURA DOADOR / AMOSTRA / REGIÃO")
cand = ["Brain.Region", "sample", "Braak.stage", "Disease.Group", "APOE_class", "Amyloid", "NP.Diagonis",
        "Age_group", "Sex", "PMI_group", "RIN_group", "SORT", "Assay", "suspension_type",
        "Unique.identifier", "disease_ontology_term_id"]
cols_d = [c for c in cand if tem(c)]
# NaN vira o texto "NA" (astype(str) sozinho preserva NaN float em versões novas do pandas e quebra o join)
meta = obs[[COL_DOADOR] + cols_d].astype(object).fillna("NA").astype(str)

n_doadores = meta[COL_DOADOR].nunique()
log(f"Doadores: {n_doadores}")
if tem(COL_AMOSTRA):
    amostras_por_doador = meta.groupby(COL_DOADOR)[COL_AMOSTRA].nunique()
    doador_por_amostra = meta.groupby(COL_AMOSTRA)[COL_DOADOR].nunique()
    log(f"Amostras: {meta[COL_AMOSTRA].nunique()} | amostras por doador: mediana={amostras_por_doador.median():.0f} "
        f"(mín {amostras_por_doador.min()}, máx {amostras_por_doador.max()})")
    if doador_por_amostra.max() == 1:
        flag("OK", "Cada amostra pertence a um único doador.")
    else:
        flag("ERRO", f"{(doador_por_amostra > 1).sum()} amostras aparecem em mais de um doador.")

nun = meta.groupby(COL_DOADOR)[cols_d].nunique()
const = pd.DataFrame({"doadores_com_mais_de_1_valor": (nun > 1).sum(), "max_valores_por_doador": nun.max()})
log("\nVariáveis constantes por doador? (0 = constante)\n" + const.to_string())

tab_d = meta.groupby(COL_DOADOR).agg(lambda s: " | ".join(sorted(s.unique())[:6]))
tab_d.insert(0, "n_celulas", meta.groupby(COL_DOADOR).size())
tab_d = tab_d.sort_values("n_celulas", ascending=False)
tab_d.to_csv(SAIDA_DOADORES)
mostrar = [c for c in ["n_celulas", COL_REGIAO, "Braak.stage", "Disease.Group", "APOE_class", "NP.Diagonis"] if c in tab_d.columns]
log("\nTabela por doador (completa em CSV):\n" + tab_d[mostrar].to_string())

for c in ["Disease.Group", "Braak.stage", "APOE_class"]:
    if c in tab_d.columns:
        log(f"\nDoadores por {c} (valores por doador):\n" + tab_d[c].value_counts().to_string())

# ----------------------------------------------------------------------------
# 5. COMPOSIÇÃO DAS CLASSES
# ----------------------------------------------------------------------------
titulo("5. COMPOSIÇÃO DAS CLASSES (proporção de cada classe dentro de cada nível, por coluna)")
for v in ["Brain.Region", "Disease.Group", "Braak.stage", "SORT", "Assay", "suspension_type"]:
    if tem(v) and tem(COL_CLASSE) and obs[v].nunique() <= 12:
        ct = pd.crosstab(obs[COL_CLASSE].astype(str), obs[v].astype(str), normalize="columns").round(3)
        log(f"\n-- {COL_CLASSE} x {v} --\n" + ct.to_string())

# ----------------------------------------------------------------------------
# 6. DOADORES POR CLASSE
# ----------------------------------------------------------------------------
titulo("6. DOADORES POR CLASSE (viabilidade do leave-one-donor-out)")
for coluna in [COL_CLASSE, COL_FINA]:
    if not tem(coluna) or obs[coluna].nunique() > 60:
        continue
    tab = pd.crosstab(obs[COL_DOADOR].astype(str), obs[coluna].astype(str))
    res = pd.DataFrame({"celulas": tab.sum(), "doadores>=1": (tab > 0).sum()})
    for L in LIMIARES:
        res[f"doadores>={L}"] = (tab >= L).sum()
    log(f"\n-- {coluna} --\n" + res.to_string())
    if coluna == COL_CLASSE:
        tab.to_csv(SAIDA_CLASSE_DOADOR)
        pobres = res.index[res["doadores>=50"] < 3].tolist()
        if pobres:
            flag("AVISO", f"Classes com menos de 3 doadores com >= 50 células: {pobres}")
        else:
            flag("OK", "Todas as classes têm >= 3 doadores com >= 50 células.")

titulo("6b. MAPEAMENTO PARA AS 8 CLASSES DO TREINO (via author_annotation)")
import re

NOMES8 = {1: "Astrocytes", 2: "Oligodendrocytes", 3: "Exc. Neurons", 4: "Inh. Neurons",
          5: "OPCs", 6: "Microglia", 7: "Endothelial", 8: "VLMCs"}


def mapear_fino(nome):
    n = str(nome)
    if re.match(r"^Ex\d+", n): return 3
    if re.match(r"^In\d+", n): return 4
    if n.startswith("Astrocyte"): return 1
    if n.startswith("Oligodendrocyte"): return 2
    if n == "OPC": return 5
    if n.startswith("Microglia"): return 6
    if n == "Endothelial": return 7
    if n == "VLMC": return 8
    return np.nan  # ex.: Pericytes, sem equivalente entre as 8 classes do treino


if tem(COL_FINA):
    cod = obs[COL_FINA].astype(str).map(mapear_fino)
    sem = obs.loc[cod.isna(), COL_FINA].astype(str).value_counts()
    log("Categorias SEM equivalente no treino (ficam fora do mapeamento):\n"
        + (sem.to_string() if len(sem) else "  nenhuma"))
    nome8 = cod.map(NOMES8).fillna("sem equivalente")
    log(f"\nCélulas mapeadas: {int(cod.notna().sum())} de {n_cells}")

    tabd = pd.crosstab(obs[COL_DOADOR].astype(str), nome8)
    res8 = pd.DataFrame({"celulas": tabd.sum()})
    for L in LIMIARES:
        res8[f"doadores>={L}"] = (tabd >= L).sum()
    log("\nClasses do treino x doadores:\n" + res8.to_string())

    for v in ["Assay", "SORT", "Brain.Region"]:
        if tem(v):
            log(f"\n-- células por classe x {v} --\n" + pd.crosstab(nome8, obs[v].astype(str)).to_string())

    mapa = pd.DataFrame({"classe_treino": cod.values, "classe_nome": nome8.values,
                         "author_annotation": obs[COL_FINA].astype(str).values,
                         COL_DOADOR: obs[COL_DOADOR].astype(str).values}, index=obs.index.astype(str))
    for v in [COL_AMOSTRA, "Assay", "SORT", COL_REGIAO]:
        if tem(v):
            mapa[v] = obs[v].astype(str).values
    mapa.to_csv(os.path.join(PASTA, "metadados_GSE263468_mapa_classes.csv"))

    mask = (res8.index != "sem equivalente") & (res8["doadores>=20"] < 3).values
    pobres = res8.index[mask].tolist()
    if pobres:
        flag("AVISO", f"Classes do treino com menos de 3 doadores com >= 20 células: {pobres}")
    else:
        flag("OK", "Todas as classes do treino têm >= 3 doadores com >= 20 células.")

# ----------------------------------------------------------------------------
# 7. INDEPENDÊNCIA ENTRE COORTES
# ----------------------------------------------------------------------------
titulo("7. SOBREPOSIÇÃO DE IDENTIFICADORES COM OUTRAS COORTES")
ids_ext = {c: set(obs[c].astype(str)) for c in [COL_DOADOR, COL_AMOSTRA, "Unique.identifier"] if tem(c)}
barcodes_ext = set(obs.index.astype(str))
if tem("cell_barcodes"):
    barcodes_ext |= set(obs["cell_barcodes"].astype(str))

for nome, caminhos in COORTES_COMPARACAO.items():
    p = next((c for c in caminhos if os.path.exists(c)), None)
    if p is None:
        flag("AVISO", f"{nome}: arquivo não encontrado; ajuste COORTES_COMPARACAO.")
        continue
    with h5py.File(p, "r") as g:
        o2 = read_elem(g["obs"])
    log(f"\n{nome}: {p} | {len(o2)} células | colunas: {o2.columns.tolist()}")
    palavras = ["donor", "patient", "subject", "individual", "sample", "id", "orig", "case"]
    cols2 = [c for c in o2.columns if any(w in c.lower() for w in palavras)]
    achou = False
    for c1, s1 in ids_ext.items():
        for c2 in cols2:
            inter = s1 & set(o2[c2].astype(str))
            if inter:
                achou = True
                log(f"  interseção {COL_DOADOR if c1 == COL_DOADOR else c1} x {c2}: {len(inter)} valores (ex.: {sorted(inter)[:5]})")
    log(f"  barcodes em comum (string completa): {len(barcodes_ext & set(o2.index.astype(str)))}")
    if not achou:
        log("  nenhuma interseção de identificadores de doador/amostra.")
    log("  Observação: ausência de interseção NÃO prova independência (nomenclaturas diferentes). "
        "Confirme no artigo/GEO de cada coorte se há doadores compartilhados.")
    nb = len(barcodes_ext & set(o2.index.astype(str)))
    flag("OK" if nb == 0 else "ERRO", f"{nome}: {nb} barcodes em comum.")

# ----------------------------------------------------------------------------
titulo("CHECKLIST FINAL")
for s, m in CHECKLIST:
    log(f"[{s}] {m}")
log(f"\nRelatório: {SAIDA_TXT}\nDoadores: {SAIDA_DOADORES}\nClasse x doador: {SAIDA_CLASSE_DOADOR}")
_arq.close()