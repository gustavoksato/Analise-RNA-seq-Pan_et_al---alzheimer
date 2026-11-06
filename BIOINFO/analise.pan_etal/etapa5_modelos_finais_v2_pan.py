#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETAPA 5 (v2, REFATORADA) - PRE-VERIFICACAO + MODELOS FINAIS CONGELADOS (3 BRACOS) - GSE267301 (Pan et al.)

Esta versao NAO altera nem sobrescreve a v1 (pasta etapa5_modelos_finais). Escreve em etapa5_modelos_finais_v2.
Com a mesma seed_split, os mesmos hiperparametros e o mesmo split 90/10 da v1, os modelos devem reproduzir os da v1
(o script confere e reporta); o que muda e a VERIFICACAO e a PROCEDENCIA dos artefatos.

PONTOS DA AUDITORIA TRATADOS
  P1  IDs "puros" podiam nao ser unicos (remocao de tudo apos o 1o ponto + upper, sem assert).
      -> checar_ids(): unicidade dos var_names originais (assert), IDs normalizados so por regex de Ensembl
         (remove ".versao" apenas de IDs ENSG; simbolos como AL627309.1 NAO sao truncados), unicidade da chave,
         lista de colisoes, comparacao com o esquema da v1, e escolha explicita da CHAVE de mapeamento externo.
  P2  Ordem dos genes x W (ponto critico). A v1 so conferia o h5ad contra a Etapa 1 (autoconsistencia).
      -> verificar_ordem_W(): recalcula (X_h5ad > 0) @ W em ~200 celulas e compara com as linhas do TSV SWeeP
         (Pearson por celula, erro relativo, controle com colunas embaralhadas). So "CONFIRMADA" libera o
         congelamento do braco SWeeP para inferencia externa. (O R multiplica a mtx binarizada por W sem
         padronizar: o TSV deve ser reproduzivel numericamente.)
  P2b Hipotese "baixa correspondencia com features_ENSG_puro.tsv = sufixo/caixa" -> diagnosticar_molde():
      posicoes iguais sob 3 esquemas de normalizacao + sobreposicao de CONJUNTO + ordenacao. Se o conjunto
      coincide e a posicao nao, a causa e ORDEM, nao sufixo/caixa.
  P3  Round-trip fraco -> autoteste PONTA A PONTA a partir do DISCO (npz/pt, em memoria nova): matriz bruta de 61.541
      colunas -> indices HVG -> PCA -> scaler -> MLP; SWeeP: TSV bruto (e, se W disponivel, h5ad->binarizacao->W).
      Tambem: pkl x npz equivalentes; simulacao de coorte externa (colunas permutadas e genes ausentes) pelo
      reindexar_para_pan() que a Etapa 6 vai reutilizar.
  P4  Manifesto sem afirmacao nao verificada sobre coorte externa; regras de genes ausentes/normalizacao
      registradas como decisoes a revisar, com a evidencia de normalizacao do h5ad (soma expm1 por celula).
  P5  Procedencia: SHA-256 de W (.rds), TSV, h5ad, metadados, listas de genes e deste script; pacotes instalados;
      hash do manifesto em arquivo separado.
  P6  Artefatos tambem em .npz (arrays puros, independentes da versao do sklearn) alem dos .pkl.
  P7  Pre-requisito: exige TESTE_CEGO_LOCK.json (Etapa 3) e LODO_LOCK.json (Etapa 4).
  P8  Desvio declarado: split 90/10 estratificado por (doador x classe) (as Etapas 3/4 estratificaram so por classe).

NAO FAZ (por escopo): sensibilidade do criterio de checkpoint (loss x F1), regularizacao do SWeeP 600D,
experimentos de genes ausentes/downsampling, coortes externas.

Uso:
    python etapa5_modelos_finais_v2_pan.py --so-preflight              # 1o: so verificacoes (rapido, sem treino)
    python etapa5_modelos_finais_v2_pan.py                             # preflight + treino + congelamento
    python etapa5_modelos_finais_v2_pan.py --W /caminho/orthbase_mproj_600d.rds
    python etapa5_modelos_finais_v2_pan.py --retomar
    python etapa5_modelos_finais_v2_pan.py --continuar-sem-verificacao-w   # treina mesmo sem W confirmada (SWeeP
                                                                           # fica BLOQUEADO para inferencia externa)
"""

import os
import re
import sys
import json
import time
import copy
import hashlib
import platform
import argparse
import datetime
import subprocess
import tempfile
from importlib import metadata as md

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

# ==============================================================================
# CAMINHOS E CONFIGURACAO
# ==============================================================================
BASE = "/media/gustavo-k-sato/D61CF48E1CF46B3F1/analisesPAN"
CAM_SWEEP_600 = "/media/gustavo-k-sato/D61CF48E1CF46B3F1/AIBIA/R_analises/SWeeP_600_Classes_Alzheimer_Oficial.tsv"
CAM_METADADOS = os.path.join(BASE, "Metadados_Pan_Limpos.csv")
CAM_H5AD_LOG = os.path.join(BASE, "Pan_58k_LogNorm_Para_BaselineHVG.h5ad")
CAM_MOLDE_ENSG = os.path.join(BASE, "features_ENSG_puro.tsv")
CAM_W_DEFAULT = ("/media/gustavo-k-sato/D61CF48E1CF46B3F1/AIBIA/R_analises/"
                 "resultados_SWeeP_GSE263468/orthbase_mproj_600d.rds")

DEFAULT_RUN = os.path.join(BASE, "resultados_MLP_GSE267301_v2", "run_20261102_201556")
SCRIPT_PATH = os.path.abspath(__file__) if "__file__" in globals() else None

FRAC_HOLDOUT_ES = 0.10
N_HVG = 2000
N_CEL_CHECK_W = 200          # celulas usadas na verificacao W x TSV
SEED_CHECK = 2026
LIM_R_MIN = 0.999            # Pearson minimo por celula para "CONFIRMADA"
LIM_ERR_REL_MED = 1e-3       # erro relativo mediano maximo para "CONFIRMADA"

NOMES_CLASSES = ["1: Astrocytes", "2: Oligodendrocytes", "3: Exc. Neurons", "4: Inh. Neurons",
                 "5: OPCs", "6: Microglia", "7: Endothelial", "8: VLMCs"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RE_ENSEMBL = re.compile(r"^(ENS[A-Z]{0,6}G\d{11})(?:\.\d+)?$")

# ------------------------------------------------------------------------------
# Escrita SEM overwrite
# ------------------------------------------------------------------------------
RETOMANDO = False


def _checar_novo(caminho):
    if os.path.exists(caminho):
        if RETOMANDO:
            print(f"   [retomada] ja existe, mantido: {os.path.basename(caminho)}")
            return False
        raise FileExistsError(f"Recusando sobrescrever: {caminho}")
    return True


def salvar_df(df, caminho, **kw):
    if _checar_novo(caminho):
        df.to_csv(caminho, **kw)


def salvar_json(obj, caminho):
    if _checar_novo(caminho):
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def salvar_txt(linhas, caminho):
    if _checar_novo(caminho):
        with open(caminho, "w", encoding="utf-8") as f:
            f.write("\n".join(map(str, linhas)) + "\n")


def salvar_npy(arr, caminho):
    if _checar_novo(caminho):
        np.save(caminho, arr)


def salvar_npz(caminho, **arrs):
    if _checar_novo(caminho):
        np.savez(caminho, **arrs)


def salvar_pkl(obj, caminho):
    if _checar_novo(caminho):
        joblib.dump(obj, caminho)


def salvar_pt(obj, caminho):
    if _checar_novo(caminho):
        torch.save(obj, caminho)


def salvar_fig(fig, caminho):
    if _checar_novo(caminho):
        fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fixar_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def versao(pacote):
    try:
        return md.version(pacote)
    except Exception:
        return "?"


def sha256_arquivo(caminho, bloco=1 << 24):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for b in iter(lambda: f.read(bloco), b""):
            h.update(b)
    return h.hexdigest()


def sha256_lista(strings):
    h = hashlib.sha256()
    for s in strings:
        h.update((str(s) + "\n").encode("utf-8"))
    return h.hexdigest()


# ==============================================================================
# RUN DA ETAPA 2, PASTAS, LOCK, BRACOS
# ==============================================================================
def carregar_run(run_dir):
    run_dir = os.path.abspath(run_dir)
    for n in ["decisao_hiperparametros.json", "config_etapas_0_a_2.json"]:
        if not os.path.exists(os.path.join(run_dir, n)):
            raise FileNotFoundError(f"Arquivo da Etapa 2 ausente: {os.path.join(run_dir, n)}")
    with open(os.path.join(run_dir, "decisao_hiperparametros.json"), encoding="utf-8") as f:
        decisao = json.load(f)
    with open(os.path.join(run_dir, "config_etapas_0_a_2.json"), encoding="utf-8") as f:
        cfg = json.load(f)["cfg"]
    assert decisao["pesos_escolhido"] in ("Balanced", "Sem_Pesos"), "Decisao de pesos invalida"
    assert int(decisao["k_escolhido"]) in cfg["k_lista"], "k escolhido fora da lista da Etapa 2"
    # P7: o modelo final so e congelado depois de avaliado (Etapas 3 e 4 executadas nesta run)
    for lock in [os.path.join(run_dir, "etapa3_teste_cego", "TESTE_CEGO_LOCK.json"),
                 os.path.join(run_dir, "etapa4_lodo", "LODO_LOCK.json")]:
        if not os.path.exists(lock):
            raise FileNotFoundError(f"Pre-requisito ausente (Etapa 3/4 nao executada nesta run): {lock}")
    return decisao, cfg


def preparar_pastas(run_dir, ts, so_preflight):
    run_dir = os.path.abspath(run_dir)
    e5 = os.path.join(run_dir, "etapa5_modelos_finais_v2")
    P = {"run": run_dir, "etapa5": e5, "v1": os.path.join(run_dir, "etapa5_modelos_finais")}
    subs = ["preflight_" + ts] if so_preflight else ["preflight_" + ts, "artefatos_congelados", "seeds", "meta", "figuras"]
    for sub in subs:
        key = "preflight" if sub.startswith("preflight_") else sub
        P[key] = os.path.join(e5, sub)
        os.makedirs(P[key], exist_ok=True)
    return P


def criar_lock(P, decisao, bracos):
    cam = os.path.join(P["etapa5"], "MODELOS_FINAIS_V2_LOCK.json")
    if os.path.exists(cam):
        if not RETOMANDO:
            raise FileExistsError("A Etapa 5 v2 desta run JA FOI INICIADA (MODELOS_FINAIS_V2_LOCK.json existe). "
                                  "Use --retomar para continuar uma execucao interrompida.")
        print("   [retomada] lock existente mantido.")
        return
    with open(cam, "x", encoding="utf-8") as f:
        json.dump({"iniciado_em": datetime.datetime.now().isoformat(timespec="seconds"),
                   "pesos": decisao["pesos_escolhido"], "k": decisao["k_escolhido"],
                   "bracos": [b["nome"] for b in bracos]}, f, indent=2)


def definir_bracos(decisao):
    k = int(decisao["k_escolhido"])
    return [
        {"nome": "Baseline_HVG2000_PCA50", "tipo": "hvg_log", "dim": 50},
        {"nome": "SWeeP_600D_Puro", "tipo": "sweep_puro", "dim": 600},
        {"nome": f"SWeeP_600D_PCA{k}", "tipo": "sweep_pca", "dim": k},
    ]


# ==============================================================================
# DADOS (o h5ad fica em memoria durante toda a execucao)
# ==============================================================================
def carregar_e_auditar():
    for nome, cam in [("SWeeP TSV", CAM_SWEEP_600), ("Metadados", CAM_METADADOS), ("h5ad LogNorm", CAM_H5AD_LOG)]:
        if not os.path.exists(cam):
            raise FileNotFoundError(f"{nome} nao encontrado: {cam}")
    df_sw = pd.read_csv(CAM_SWEEP_600, sep="\t")
    y_tsv = df_sw["Cell_Type"].values.astype(int)
    X_sweep = df_sw.drop(columns=["Cell_Type"]).values.astype(np.float32)
    del df_sw
    df_meta = pd.read_csv(CAM_METADADOS)
    adata = sc.read_h5ad(CAM_H5AD_LOG)
    obs_index = adata.obs.index.values.astype(str)

    assert X_sweep.shape[0] == len(y_tsv) == len(df_meta) == adata.n_obs == 58326, "Numero de celulas != 58326"
    assert X_sweep.shape[1] == 600 and adata.n_vars == 61541, "Dimensoes invalidas"
    assert np.array_equal(obs_index, df_meta["Barcode"].astype(str).values), "Barcodes do h5ad != metadados"
    assert np.array_equal(y_tsv, df_meta["Cell_Type_Num"].values.astype(int)), "Rotulos do TSV != metadados"
    assert not np.isnan(X_sweep).any(), "NaN detectado na matriz SWeeP"
    assert len(set(obs_index)) == len(obs_index), "Barcodes duplicados"
    donors = df_meta["Donor"].astype(str).values
    y = y_tsv - 1
    print(f"   -> Auditoria OK: {len(y)} celulas, {len(np.unique(donors))} doadores, {adata.n_vars} genes.")
    return X_sweep, y, donors, adata


def linhas_de(Xall, idx):
    M = Xall[idx]
    return M


def para_denso(M):
    return (M.toarray() if sp.issparse(M) else np.asarray(M))


# ==============================================================================
# P1 - IDs DOS GENES
# ==============================================================================
def normalizar_id(s):
    """Remove '.versao' SOMENTE de IDs Ensembl (ENSG...). Simbolos (AL627309.1) ficam intactos. Maiusculas."""
    s = str(s).strip().upper()
    m = RE_ENSEMBL.match(s)
    return m.group(1) if m else s


def esquema_v1(s):
    """Esquema da v1 (para comparacao): remove tudo apos o 1o ponto + upper."""
    return re.sub(r"\..*$", "", str(s)).strip().upper()


def checar_ids(var_names):
    raw = np.asarray(var_names).astype(str)
    assert len(raw) == len(set(raw)), "var_names ORIGINAIS do h5ad nao sao unicos!"
    key = np.array([normalizar_id(s) for s in raw])
    v1 = np.array([esquema_v1(s) for s in raw])
    n_ens = int(sum(bool(RE_ENSEMBL.match(str(s).strip().upper())) for s in raw))
    n_com_ponto = int(sum("." in s for s in raw))
    vc_key, vc_v1 = pd.Series(key).value_counts(), pd.Series(v1).value_counts()
    colis_key, colis_v1 = vc_key[vc_key > 1], vc_v1[vc_v1 > 1]
    chave_unica = bool(len(colis_key) == 0)
    if chave_unica:
        chave_nome, chaves = "ensembl_sem_versao_upper (regex ENSG)", key
    else:
        chave_nome, chaves = "var_names_originais (normalizacao gera colisoes)", raw
    resumo = {
        "n_genes": int(len(raw)),
        "var_names_originais_unicos": True,
        "n_ids_formato_ensembl": n_ens,
        "frac_ids_formato_ensembl": n_ens / len(raw),
        "n_ids_com_ponto": n_com_ponto,
        "exemplos_originais": list(raw[:3]),
        "n_unicos_chave_regex": int(len(set(key))),
        "n_colisoes_chave_regex": int(len(colis_key)),
        "n_unicos_esquema_v1": int(len(set(v1))),
        "n_colisoes_esquema_v1": int(len(colis_v1)),
        "n_ids_que_diferem_regex_vs_v1": int(np.sum(key != v1)),
        "chave_de_mapeamento_escolhida": chave_nome,
        "chave_e_unica": chave_unica,
    }
    df_col = pd.DataFrame({
        "chave": list(colis_key.index) + list(colis_v1.index),
        "n": list(colis_key.values) + list(colis_v1.values),
        "esquema": ["regex"] * len(colis_key) + ["v1"] * len(colis_v1)})
    return raw, chaves, key, resumo, df_col


# ==============================================================================
# P2b - DIAGNOSTICO CONTRA features_ENSG_puro.tsv
# ==============================================================================
def diagnosticar_molde(raw, cam_molde):
    if not os.path.exists(cam_molde):
        return None, "molde nao encontrado"
    molde = pd.read_csv(cam_molde, sep="\t", header=None)[0].astype(str).values
    esquemas = {"original (sem normalizar)": (lambda s: str(s)),
                "v1 (strip apos 1o ponto, sem upper)": (lambda s: re.sub(r"\..*$", "", str(s))),
                "v1 + upper": esquema_v1,
                "regex Ensembl + upper": normalizar_id}
    linhas = []
    for nome, fn in esquemas.items():
        a = np.array([fn(s) for s in raw])
        b = np.array([fn(s) for s in molde])
        igual_pos = int(np.sum(a == b)) if len(a) == len(b) else None
        conj = len(set(a) & set(b))
        linhas.append({"Esquema": nome, "N_h5ad": len(a), "N_molde": len(b),
                       "Posicoes_iguais": igual_pos, "Sobreposicao_de_CONJUNTO": conj,
                       "Frac_conjunto_h5ad": conj / max(len(set(a)), 1)})
    a = np.array([normalizar_id(s) for s in raw])
    b = np.array([normalizar_id(s) for s in molde])
    pos_b = {g: i for i, g in enumerate(b)}
    comuns = [(i, pos_b[g]) for i, g in enumerate(a) if g in pos_b]
    spear = float("nan")
    if len(comuns) > 10:
        spear = float(pd.Series([c[0] for c in comuns]).corr(pd.Series([c[1] for c in comuns]), method="spearman"))
    extra = {"h5ad_ordenado_alfabeticamente": bool(np.all(a[:-1] <= a[1:])),
             "molde_ordenado_alfabeticamente": bool(np.all(b[:-1] <= b[1:])),
             "spearman_posicoes_nos_genes_comuns": spear}
    df = pd.DataFrame(linhas)
    reg = df[df.Esquema == "regex Ensembl + upper"].iloc[0]
    raw_pos = df[df.Esquema == "original (sem normalizar)"].iloc[0]
    n = max(len(raw), 1)
    if reg.Frac_conjunto_h5ad >= 0.99 and (reg.Posicoes_iguais or 0) / n < 0.5:
        interp = ("MESMO CONJUNTO de genes, ORDEM DIFERENTE: sufixo de versao e caixa NAO explicam a baixa "
                  "correspondencia posicional.")
    elif (reg.Posicoes_iguais or 0) - (raw_pos.Posicoes_iguais or 0) > 0.3 * n:
        interp = "A normalizacao de IDs (sufixo/caixa) explica a maior parte da diferenca posicional."
    else:
        interp = "Conjuntos tambem diferem; correspondencia parcial entre anotacoes. Ver tabela."
    extra["interpretacao"] = interp
    return (df, extra), interp


# ==============================================================================
# P2 - ORDEM DOS GENES x W  (reproduz o TSV SWeeP a partir do h5ad)
# ==============================================================================
_R_CONVERSOR = r"""
suppressMessages(library(Matrix))
a <- commandArgs(TRUE)
W <- readRDS(a[1])
if (is.data.frame(W)) W <- as.matrix(W)
if (!is.matrix(W)) W <- as.matrix(W)
d <- dim(W)
con <- file(a[2], "wb")
writeBin(as.integer(d), con, size = 4)
writeBin(as.numeric(W), con, size = 8)
close(con)
"""


def carregar_W(caminho, pasta_tmp):
    ext = os.path.splitext(caminho)[1].lower()
    if ext == ".npy":
        W = np.load(caminho)
    elif ext == ".npz":
        z = np.load(caminho)
        W = z[z.files[0]]
    elif ext in (".csv", ".tsv", ".txt"):
        W = pd.read_csv(caminho, sep=None, engine="python", header=None).values
    elif ext == ".rds":
        W = None
        try:
            rs = os.path.join(pasta_tmp, "converter_W.R")
            out = os.path.join(pasta_tmp, "W_convertida.bin")
            with open(rs, "w") as f:
                f.write(_R_CONVERSOR)
            subprocess.run(["Rscript", rs, caminho, out], check=True, capture_output=True, text=True)
            raw = np.fromfile(out, dtype="<i4", count=2)
            d0, d1 = int(raw[0]), int(raw[1])
            vals = np.fromfile(out, dtype="<f8", offset=8)
            W = vals.reshape((d0, d1), order="F")
        except Exception as e:  # noqa
            print(f"   [AVISO] Conversao via Rscript falhou ({type(e).__name__}); tentando pyreadr...")
            try:
                import pyreadr
                obj = list(pyreadr.read_r(caminho).values())[0]
                W = np.asarray(obj.values, dtype=np.float64)
            except Exception as e2:  # noqa
                raise RuntimeError(
                    "Nao consegui ler a W (.rds). Exporte-a em R e passe o arquivo com --W:\n"
                    "   W <- readRDS('orthbase_mproj_600d.rds'); "
                    "write.table(as.matrix(W), 'W.tsv', sep='\\t', row.names=FALSE, col.names=FALSE)\n"
                    "   (ou use numpy: np.save). Detalhe: " + repr(e2))
    else:
        raise ValueError(f"Extensao de W nao suportada: {ext}")
    W = np.asarray(W)
    assert W.ndim == 2, f"W nao e 2D: {W.shape}"
    if W.shape == (600, 61541):
        W = W.T
    assert W.shape == (61541, 600), f"Dimensao inesperada da W: {W.shape}"
    return np.ascontiguousarray(W, dtype=np.float64)


def _pearson_linhas(A, B):
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    num = (A * B).sum(1)
    den = np.sqrt((A * A).sum(1) * (B * B).sum(1))
    return np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)


def verificar_ordem_W(Xall, X_tsv, W, idx):
    """Recalcula (X>0)@W e compara com as linhas do TSV SWeeP. Inclui controle (colunas embaralhadas) e hipotese
    alternativa (valores log, nao binarizados)."""
    Xs = Xall[idx]
    B = (Xs > 0)
    B = B.astype(np.float64) if sp.issparse(B) else np.asarray(B, dtype=np.float64)
    T = X_tsv[idx].astype(np.float64)
    P = np.asarray(B @ W)
    r = _pearson_linhas(P, T)
    err = np.linalg.norm(P - T, axis=1) / np.maximum(np.linalg.norm(T, axis=1), 1e-300)
    rng = np.random.default_rng(SEED_CHECK)
    perm = rng.permutation(W.shape[0])
    Wp = W[perm, :]                                   # equivale a embaralhar a ordem dos genes
    r_ctrl = _pearson_linhas(np.asarray(B @ Wp), T)
    L = Xs.astype(np.float64)
    r_alt = _pearson_linhas(np.asarray(L @ W), T)
    df = pd.DataFrame({"idx_celula": idx, "Pearson": r, "Erro_Relativo": err,
                       "MaxAbsDiff": np.abs(P - T).max(1), "Pearson_Controle_Embaralhado": r_ctrl,
                       "Pearson_Alt_ValoresLog_NaoBinarizados": r_alt})
    resumo = {"n_celulas": int(len(idx)),
              "pearson_mediano": float(np.nanmedian(r)), "pearson_minimo": float(np.nanmin(r)),
              "erro_relativo_mediano": float(np.median(err)), "erro_relativo_maximo": float(err.max()),
              "max_abs_diff": float(np.abs(P - T).max()),
              "controle_embaralhado_pearson_mediano": float(np.nanmedian(r_ctrl)),
              "alt_log_nao_binarizado_pearson_mediano": float(np.nanmedian(r_alt))}
    if resumo["pearson_minimo"] >= LIM_R_MIN and resumo["erro_relativo_mediano"] <= LIM_ERR_REL_MED:
        resumo["veredito"] = "CONFIRMADA"
    elif resumo["pearson_mediano"] >= 0.99:
        resumo["veredito"] = "INCONCLUSIVA (alta correlacao, mas valores diferem: ha transformacao adicional?)"
    else:
        resumo["veredito"] = "FALHA (a ordem dos genes do h5ad NAO reproduz o TSV)"
    return df, resumo


def orto_W(W):
    G = W.T @ W
    return float(np.abs(G - np.eye(G.shape[0])).max())


# ==============================================================================
# P4 - EVIDENCIA DE NORMALIZACAO DO h5ad
# ==============================================================================
def evidencia_normalizacao(adata, n=500, seed=SEED_CHECK):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(adata.n_obs, size=min(n, adata.n_obs), replace=False))
    M = para_denso(adata.X[idx]).astype(np.float64)
    soma = np.expm1(M).sum(1)
    cv = float(np.std(soma) / max(np.mean(soma), 1e-300))
    return {"n_celulas_amostra": int(len(idx)), "X_min": float(M.min()), "X_max": float(M.max()),
            "soma_expm1_por_celula_mediana": float(np.median(soma)),
            "soma_expm1_por_celula_min": float(soma.min()), "soma_expm1_por_celula_max": float(soma.max()),
            "cv_soma_expm1": cv,
            "interpretacao": ("compativel com normalize_total(target_sum=%.0f) + log1p" % round(np.median(soma))
                              if cv < 1e-3 else "soma por celula NAO e constante: normalizacao nao e simples "
                                                "normalize_total+log1p; conferir preparar_pan_GSE267301.py"),
            "target_sum_inferido": float(round(np.median(soma))) if cv < 1e-3 else None,
            "uns_keys": list(adata.uns.keys())}


# ==============================================================================
# UTILITARIOS DE INFERENCIA EXTERNA (reutilizaveis na Etapa 6)
# ==============================================================================
def reindexar_para_pan(X_ext, ids_ext, chaves_pan, chave_fn=normalizar_id):
    """Leva uma matriz externa (n x g_ext, CONTAGENS BRUTAS) para a ordem exata dos 61.541 genes do Pan.
    Genes ausentes -> coluna zero; IDs externos duplicados na mesma chave -> contagens SOMADAS.
    Retorna (csr n x 61541, mascara_presentes[61541] bool, resumo)."""
    chaves_pan = list(chaves_pan)
    pos = {g: i for i, g in enumerate(chaves_pan)}
    assert len(pos) == len(chaves_pan), "chaves do Pan nao unicas"
    ids_ext = [chave_fn(s) for s in ids_ext]
    ext_idx, pan_idx = [], []
    for j, g in enumerate(ids_ext):
        i = pos.get(g)
        if i is not None:
            ext_idx.append(j)
            pan_idx.append(i)
    g_ext = len(ids_ext)
    M = sp.csr_matrix((np.ones(len(ext_idx), dtype=np.float32), (ext_idx, pan_idx)), shape=(g_ext, len(chaves_pan)))
    Xs = X_ext if sp.issparse(X_ext) else sp.csr_matrix(X_ext)
    X_pan = (Xs @ M).tocsr()
    presentes = np.zeros(len(chaves_pan), dtype=bool)
    presentes[pan_idx] = True
    resumo = {"genes_externos": g_ext, "genes_pan_cobertos": int(presentes.sum()),
              "cobertura_pan": float(presentes.mean()),
              "genes_externos_sem_correspondencia": int(g_ext - len(ext_idx)),
              "ids_externos_duplicados_na_chave": int(len(ext_idx) - len(set(pan_idx)))}
    return X_pan, presentes, resumo


def cobertura_hvg(presentes_pan, hvg_cols):
    c = np.asarray(presentes_pan)[hvg_cols]
    return {"hvg_total": int(len(hvg_cols)), "hvg_presentes": int(c.sum()), "cobertura_hvg": float(c.mean())}


def normalizar_log1p(X_counts, target_sum):
    """normalize_total + log1p (mesma definicao do Pan, com target_sum inferido no manifesto)."""
    X = X_counts.tocsr().astype(np.float32) if sp.issparse(X_counts) else sp.csr_matrix(X_counts, dtype=np.float32)
    s = np.asarray(X.sum(1)).ravel()
    s[s == 0] = 1.0
    X = sp.diags((target_sum / s).astype(np.float32)) @ X
    X.data = np.log1p(X.data)
    return X.tocsr()


def projetar_sweep(X_counts_pan, W):
    """Binariza (x>0) e projeta com a MESMA W. Entrada: contagens JA na ordem do Pan."""
    B = (X_counts_pan > 0)
    B = B.astype(np.float32) if sp.issparse(B) else np.asarray(B, dtype=np.float32)
    return np.asarray(B @ W.astype(np.float32))


# ==============================================================================
# SPLIT 90/10 (identico ao da v1: estratificado por doador x classe)
# ==============================================================================
def criar_split_90_10(y, donors, seed_split, P):
    cam = os.path.join(P["etapa5"], "split_final_90_10_early_stopping.npz")
    if os.path.exists(cam) and RETOMANDO:
        sp_ = np.load(cam)
        print(f"   [retomada] Split carregado: treino={len(sp_['idx_treino_final'])} | val_es={len(sp_['idx_val_es'])}")
        return sp_["idx_treino_final"], sp_["idx_val_es"]
    estrato = np.array([f"{d}_{c}" for d, c in zip(donors, y)])
    idx_tr, idx_va = train_test_split(np.arange(len(y)), test_size=FRAC_HOLDOUT_ES,
                                      random_state=seed_split, stratify=estrato)
    assert len(np.intersect1d(idx_tr, idx_va)) == 0
    assert set(np.unique(y[idx_tr])) == set(range(8)) and set(np.unique(y[idx_va])) == set(range(8))
    assert set(np.unique(donors[idx_tr])) == set(np.unique(donors))
    assert set(np.unique(donors[idx_va])) == set(np.unique(donors))
    if _checar_novo(cam):
        np.savez(cam, idx_treino_final=idx_tr, idx_val_es=idx_va)
    print(f"   -> Split 90/10 (doador x classe): treino={len(idx_tr)} | hold-out ES={len(idx_va)}")
    return idx_tr, idx_va


# ==============================================================================
# MLP E TREINO
# ==============================================================================
class ClassificadorCelular(nn.Module):
    def __init__(self, input_dim, num_classes=8):
        super().__init__()
        hidden1 = 128 if input_dim > 100 else 64
        self.camadas = nn.Sequential(
            nn.Linear(input_dim, hidden1), nn.ReLU(),
            nn.Linear(hidden1, 32), nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.camadas(x)


def prever_modelo(modelo, Z, batch=2048):
    modelo.eval()
    saidas = []
    with torch.no_grad():
        for i in range(0, len(Z), batch):
            saidas.append(modelo(torch.tensor(Z[i:i + batch], dtype=torch.float32).to(DEVICE)).argmax(1).cpu().numpy())
    return np.concatenate(saidas).astype(np.int64)


def treinar_modelo_final(Xtr, ytr, Xva, yva, usar_pesos, seed, cfg):
    fixar_seed(seed)
    t0 = time.time()
    nc = cfg["n_classes"]
    modelo = ClassificadorCelular(Xtr.shape[1], nc).to(DEVICE)
    pesos = np.ones(nc, dtype=np.float32)
    if usar_pesos:
        pres = np.unique(ytr)
        pesos[pres] = compute_class_weight("balanced", classes=pres, y=ytr)
        loss_tr_fn = nn.CrossEntropyLoss(weight=torch.tensor(pesos, dtype=torch.float32).to(DEVICE))
    else:
        loss_tr_fn = nn.CrossEntropyLoss()
    loss_va_fn = nn.CrossEntropyLoss()

    opt = torch.optim.Adam(modelo.parameters(), lr=cfg["lr"])
    ld_tr = DataLoader(TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long)),
                       batch_size=cfg["batch_treino"], shuffle=True)
    ld_va = DataLoader(TensorDataset(torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva, dtype=torch.long)),
                       batch_size=cfg["batch_eval"], shuffle=False)

    melhor, estado, ep_ideal, sem_melhora = float("inf"), None, 0, 0
    hist_tr, hist_va = [], []
    for ep in range(cfg["epocas_max"]):
        modelo.train()
        acc_tr = 0.0
        for bx, by in ld_tr:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            loss = loss_tr_fn(modelo(bx), by)
            loss.backward()
            opt.step()
            acc_tr += loss.item() * bx.size(0)
        hist_tr.append(acc_tr / len(ld_tr.dataset))
        modelo.eval()
        acc_va = 0.0
        with torch.no_grad():
            for bx, by in ld_va:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                acc_va += loss_va_fn(modelo(bx), by).item() * bx.size(0)
        lv = acc_va / len(ld_va.dataset)
        hist_va.append(lv)
        if lv < melhor:
            melhor, ep_ideal, sem_melhora = lv, ep + 1, 0
            estado = {k: v.detach().cpu().clone() for k, v in copy.deepcopy(modelo.state_dict()).items()}
        else:
            sem_melhora += 1
            if sem_melhora >= cfg["paciencia"]:
                break

    modelo.load_state_dict({k: v.to(DEVICE) for k, v in estado.items()})
    p_va = prever_modelo(modelo, Xva, cfg["batch_eval"])
    return {
        "state_dict": estado, "y_pred_val_es": p_va,
        "pesos_classe_treino": pesos.tolist(),
        "F1_Macro_Val_ES": float(f1_score(yva, p_va, average="macro", labels=list(range(nc)), zero_division=0)),
        "Acc_Bal_Val_ES": float(balanced_accuracy_score(yva, p_va)),
        "Acc_Val_ES": float(accuracy_score(yva, p_va)),
        "Loss_Val_CE": float(melhor), "Epoca_Ideal": int(ep_ideal), "Epocas_Rodadas": int(len(hist_va)),
        "hist_treino": hist_tr, "hist_val": hist_va, "Tempo_s": float(time.time() - t0),
    }


# ==============================================================================
# TRANSFORMADORES: ajuste, congelamento (.pkl + .npz) e aplicacao a partir de arrays
# ==============================================================================
def ajustar_transformador(braco, X_sweep, adata, idx_tr, cfg):
    tipo, seed_split = braco["tipo"], cfg["seed_split"]
    if tipo == "sweep_puro":
        sc_ = StandardScaler().fit(X_sweep[idx_tr])
        return {"scaler": sc_}, {"scaler_mean": sc_.mean_, "scaler_scale": sc_.scale_}
    if tipo == "sweep_pca":
        k = braco["dim"]
        pca = PCA(n_components=cfg["k_max_pca"], random_state=seed_split).fit(X_sweep[idx_tr])
        sc_ = StandardScaler().fit(pca.transform(X_sweep[idx_tr])[:, :k])
        return ({"pca": pca, "scaler": sc_, "k": k},
                {"pca_mean": pca.mean_, "pca_components": pca.components_[:k],
                 "pca_var_ratio": pca.explained_variance_ratio_[:k],
                 "scaler_mean": sc_.mean_, "scaler_scale": sc_.scale_})
    if tipo == "hvg_log":
        ad_tr = adata[idx_tr, :].copy()
        sc.pp.highly_variable_genes(ad_tr, n_top_genes=N_HVG, flavor="seurat")
        mask = ad_tr.var["highly_variable"].values.astype(bool)
        del ad_tr
        assert int(mask.sum()) == N_HVG, f"Esperado {N_HVG} HVGs, obteve {int(mask.sum())}"
        cols = np.where(mask)[0].astype(np.int64)
        H = para_denso(adata.X[idx_tr][:, cols]).astype(np.float32)
        pca = PCA(n_components=50, random_state=seed_split).fit(H)
        sc_ = StandardScaler().fit(pca.transform(H))
        del H
        return ({"pca": pca, "scaler": sc_, "hvg_cols": cols},
                {"hvg_cols": cols, "hvg_mask": mask, "pca_mean": pca.mean_, "pca_components": pca.components_,
                 "pca_var_ratio": pca.explained_variance_ratio_,
                 "scaler_mean": sc_.mean_, "scaler_scale": sc_.scale_})
    raise ValueError(tipo)


def aplicar_sklearn(obj, tipo, entrada):
    if tipo == "hvg_log":
        H = para_denso(entrada[:, obj["hvg_cols"]]).astype(np.float32)
        return obj["scaler"].transform(obj["pca"].transform(H)).astype(np.float32)
    if tipo == "sweep_pca":
        return obj["scaler"].transform(obj["pca"].transform(entrada)[:, :obj["k"]]).astype(np.float32)
    return obj["scaler"].transform(entrada).astype(np.float32)


def aplicar_npz(A, tipo, entrada):
    """Aplica o transformador SO a partir dos arrays do .npz (independente de sklearn)."""
    if tipo == "hvg_log":
        Z = para_denso(entrada[:, A["hvg_cols"]]).astype(np.float32)
        Z = (Z - A["pca_mean"]) @ A["pca_components"].T
    elif tipo == "sweep_pca":
        Z = (np.asarray(entrada, dtype=np.float32) - A["pca_mean"]) @ A["pca_components"].T
    else:
        Z = np.asarray(entrada, dtype=np.float32)
    return ((Z - A["scaler_mean"]) / A["scaler_scale"]).astype(np.float32)


def carregar_pipeline_disco(pasta_braco, cam_pt):
    """Recarrega transformador (npz) + MLP (pt) do DISCO. Reutilizavel na Etapa 6."""
    with np.load(os.path.join(pasta_braco, "transformador.npz"), allow_pickle=False) as z:
        A = {k: z[k] for k in z.files}
    estado = torch.load(cam_pt, map_location=DEVICE, weights_only=True)
    modelo = ClassificadorCelular(estado["camadas.0.weight"].shape[1], 8).to(DEVICE)
    modelo.load_state_dict(estado)
    modelo.eval()
    return A, modelo


def congelar_braco(braco, X_sweep, adata, idx_tr, idx_va, cfg, P, chaves, raw_ids):
    nome, tipo = braco["nome"], braco["tipo"]
    pasta = os.path.join(P["artefatos_congelados"], nome)
    os.makedirs(pasta, exist_ok=True)
    t0 = time.time()
    obj, arrs = ajustar_transformador(braco, X_sweep, adata, idx_tr, cfg)
    cam_pkl, cam_npz = os.path.join(pasta, "transformador_sklearn.pkl"), os.path.join(pasta, "transformador.npz")
    # Em retomada, o ajuste e REFEITO em memoria (determinístico) e conferido contra o npz ja salvo
    if os.path.exists(cam_npz) and RETOMANDO:
        with np.load(cam_npz, allow_pickle=False) as z:
            for k, v in arrs.items():
                assert np.allclose(z[k], v, atol=1e-5), f"Retomada: transformador refeito difere do salvo ({nome}/{k})"
    salvar_pkl(obj, cam_pkl)
    salvar_npz(cam_npz, **arrs)
    info = {"transformador_npz": f"{nome}/transformador.npz", "transformador_pkl": f"{nome}/transformador_sklearn.pkl"}
    if tipo == "hvg_log":
        cols = arrs["hvg_cols"]
        salvar_npy(cols, os.path.join(pasta, "hvg2000_indices_colunas.npy"))
        salvar_npy(arrs["hvg_mask"], os.path.join(pasta, "hvg2000_mask_61541.npy"))
        salvar_txt(np.asarray(raw_ids)[cols], os.path.join(pasta, "hvg2000_ids_originais.txt"))
        salvar_txt(np.asarray(chaves)[cols], os.path.join(pasta, "hvg2000_chaves_mapeamento.txt"))
        info.update({"hvg_indices_npy": f"{nome}/hvg2000_indices_colunas.npy",
                     "hvg_chaves_txt": f"{nome}/hvg2000_chaves_mapeamento.txt",
                     "var_explicada_50_%": float(arrs["pca_var_ratio"].sum() * 100)})
    elif tipo == "sweep_pca":
        info["var_explicada_k_%"] = float(arrs["pca_var_ratio"].sum() * 100)
    Xtr = aplicar_npz(arrs, tipo, (adata.X[idx_tr] if tipo == "hvg_log" else X_sweep[idx_tr]))
    Xva = aplicar_npz(arrs, tipo, (adata.X[idx_va] if tipo == "hvg_log" else X_sweep[idx_va]))
    assert np.isfinite(Xtr).all() and np.isfinite(Xva).all()
    info["Tempo_Prep_s"] = float(time.time() - t0)
    return Xtr, Xva, info, pasta


# ==============================================================================
# AUTOTESTES POS-CONGELAMENTO (a partir do DISCO)
# ==============================================================================
def autoteste_braco(braco, pasta, cam_pt, X_sweep, adata, idx_va, p_va_treino, chaves, raw_ids, W, rng):
    tipo = braco["tipo"]
    A, modelo = carregar_pipeline_disco(pasta, cam_pt)
    bruto_va = adata.X[idx_va] if tipo == "hvg_log" else X_sweep[idx_va]
    linhas = []

    # (a) predicao a partir do disco == predicao feita no treino
    p_disco = prever_modelo(modelo, aplicar_npz(A, tipo, bruto_va))
    linhas.append({"Teste": "ponta_a_ponta_disco_vs_treino", "Concordancia": float((p_disco == p_va_treino).mean())})

    # (b) .pkl == .npz
    obj = joblib.load(os.path.join(pasta, "transformador_sklearn.pkl"))
    d = float(np.abs(aplicar_sklearn(obj, tipo, bruto_va) - aplicar_npz(A, tipo, bruto_va)).max())
    linhas.append({"Teste": "pkl_vs_npz_max_abs_diff", "Concordancia": 1.0 if d < 1e-4 else 0.0, "Detalhe": d})

    # (c) simulacao de coorte externa: colunas permutadas -> reindexar_para_pan -> mesma predicao
    sub = np.sort(rng.choice(len(idx_va), size=min(1500, len(idx_va)), replace=False))
    if tipo == "hvg_log":
        Xc = adata.X[idx_va[sub]].tocsc() if sp.issparse(adata.X) else sp.csc_matrix(adata.X[idx_va[sub]])
        perm = rng.permutation(Xc.shape[1])
        X_perm = Xc[:, perm].tocsr()
        ids_perm = np.asarray(raw_ids)[perm]
        chave_fn = normalizar_id if "regex" in P_CHAVE["nome"] else (lambda s: str(s))
        X_back, presentes, res = reindexar_para_pan(X_perm, ids_perm, chaves, chave_fn)
        p_ref = prever_modelo(modelo, aplicar_npz(A, tipo, adata.X[idx_va[sub]]))
        p_sim = prever_modelo(modelo, aplicar_npz(A, tipo, X_back))
        linhas.append({"Teste": "reindexacao_colunas_permutadas", "Concordancia": float((p_ref == p_sim).mean()),
                       "Detalhe": res["cobertura_pan"]})
        # genes ausentes: remove 5% dos genes -> cobertura reportada corretamente
        manter = np.sort(rng.choice(X_perm.shape[1], size=int(0.95 * X_perm.shape[1]), replace=False))
        _, pres2, res2 = reindexar_para_pan(X_perm[:, manter], ids_perm[manter], chaves, chave_fn)
        cov = cobertura_hvg(pres2, A["hvg_cols"])
        ok = abs(res2["cobertura_pan"] - 0.95) < 0.005
        linhas.append({"Teste": "cobertura_com_5pct_genes_removidos", "Concordancia": 1.0 if ok else 0.0,
                       "Detalhe": f"cobertura_pan={res2['cobertura_pan']:.4f}; cobertura_HVG={cov['cobertura_hvg']:.4f}"})
    elif W is not None:
        # caminho SWeeP completo: h5ad (colunas permutadas) -> reindexa -> binariza -> W -> pipeline
        Xc = adata.X[idx_va[sub]].tocsc() if sp.issparse(adata.X) else sp.csc_matrix(adata.X[idx_va[sub]])
        perm = rng.permutation(Xc.shape[1])
        chave_fn = normalizar_id if "regex" in P_CHAVE["nome"] else (lambda s: str(s))
        X_back, _, res = reindexar_para_pan(Xc[:, perm].tocsr(), np.asarray(raw_ids)[perm], chaves, chave_fn)
        Z = projetar_sweep(X_back, W)
        p_sim = prever_modelo(modelo, aplicar_npz(A, tipo, Z))
        p_ref = prever_modelo(modelo, aplicar_npz(A, tipo, X_sweep[idx_va[sub]]))
        linhas.append({"Teste": "h5ad_permutado->reindexa->binariza->W->MLP_vs_TSV",
                       "Concordancia": float((p_ref == p_sim).mean()), "Detalhe": res["cobertura_pan"]})
    df = pd.DataFrame(linhas)
    df.insert(0, "Braco", braco["nome"])
    return df


P_CHAVE = {"nome": ""}     # preenchido na main (nome da chave de mapeamento escolhida)


# ==============================================================================
# FIGURAS E COMPARACOES
# ==============================================================================
def plotar_curva_final(P, nome, meta):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, len(meta["hist_treino"]) + 1), meta["hist_treino"], label="Loss treino (90%)", lw=2)
    ax.plot(range(1, len(meta["hist_val"]) + 1), meta["hist_val"], label="Loss hold-out ES (10%, CE s/ pesos)", lw=2)
    ax.axvline(meta["Epoca_Ideal"], color="red", ls="--", label=f"Checkpoint (epoca {meta['Epoca_Ideal']})")
    ax.set_xlabel("Epocas")
    ax.set_ylabel("Cross-entropy")
    ax.set_title(f"Modelo final congelado - {nome}")
    ax.legend()
    ax.grid(True, ls=":", alpha=0.6)
    salvar_fig(fig, os.path.join(P["figuras"], f"curva_loss_final_{nome}.png"))


def comparar_com_v1(P, df_tab, idx_tr, idx_va):
    """Reprodutibilidade: mesmos split/seeds/hiperparametros => mesmos resultados da v1 (se existir)."""
    out = {"split_identico_v1": None}
    cam_sp = os.path.join(P["v1"], "split_final_90_10_early_stopping.npz")
    if os.path.exists(cam_sp):
        z = np.load(cam_sp)
        out["split_identico_v1"] = bool(np.array_equal(z["idx_treino_final"], idx_tr) and
                                        np.array_equal(z["idx_val_es"], idx_va))
    linhas = []
    for _, r in df_tab.iterrows():
        cam = os.path.join(P["v1"], "meta", f"meta_final_{r['Representacao']}_seed{int(r['Seed'])}.json")
        if os.path.exists(cam):
            with open(cam, encoding="utf-8") as f:
                m = json.load(f)
            linhas.append({"Representacao": r["Representacao"], "Seed": int(r["Seed"]),
                           "Epoca_v1": m["Epoca_Ideal"], "Epoca_v2": int(r["Epoca_Ideal"]),
                           "F1_v1": m["F1_Macro_Val_ES"], "F1_v2": float(r["F1_Macro_Holdout_ES"]),
                           "Dif_F1": float(r["F1_Macro_Holdout_ES"]) - m["F1_Macro_Val_ES"]})
    return out, (pd.DataFrame(linhas) if linhas else None)


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    global RETOMANDO
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN, help="Pasta run_* da Etapa 2")
    ap.add_argument("--W", default=CAM_W_DEFAULT, help="Base ortonormal (.rds/.npy/.npz/.tsv) 61541x600")
    ap.add_argument("--so-preflight", action="store_true", help="So executa as verificacoes P1/P2/P4 (sem treino)")
    ap.add_argument("--retomar", action="store_true", help="Continua uma execucao interrompida")
    ap.add_argument("--continuar-sem-verificacao-w", action="store_true",
                    help="Treina mesmo sem W confirmada; SWeeP fica BLOQUEADO para inferencia externa no manifesto")
    ap.add_argument("--sem-hash-grande", action="store_true", help="Nao calcula SHA-256 do TSV e do h5ad (lento)")
    args = ap.parse_args()
    RETOMANDO = args.retomar
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    decisao, cfg = carregar_run(args.run)
    P = preparar_pastas(args.run, ts, args.so_preflight)
    bracos = definir_bracos(decisao)
    usar_pesos = decisao["pesos_escolhido"] == "Balanced"
    seeds = cfg["seeds_treino"]
    seed_principal = seeds[0]

    print("=" * 78)
    print("ETAPA 5 (v2) - PRE-VERIFICACAO + MODELOS FINAIS CONGELADOS - GSE267301")
    print("=" * 78)
    print(f"[INFO] Dispositivo={DEVICE} | Pesos={decisao['pesos_escolhido']} | k={decisao['k_escolhido']} | seeds={seeds}")

    print("\n[1] Carregando dados e auditando alinhamento")
    X_sweep, y, donors, adata = carregar_e_auditar()

    # ------------------------------------------------------------------ P1
    print("\n[2] P1 - IDs dos genes")
    raw_ids, chaves, key_regex, res_ids, df_col = checar_ids(adata.var_names)
    P_CHAVE["nome"] = res_ids["chave_de_mapeamento_escolhida"]
    for k_, v_ in res_ids.items():
        print(f"   {k_}: {v_}")
    salvar_json(res_ids, os.path.join(P["preflight"], "P1_ids_resumo.json"))
    salvar_df(df_col, os.path.join(P["preflight"], "P1_colisoes.csv"), index=False)
    if not res_ids["chave_e_unica"]:
        print("   [AVISO] A normalizacao gera colisoes: chave de mapeamento = var_names ORIGINAIS.")
    if res_ids["n_colisoes_esquema_v1"] > 0:
        print(f"   [ACHADO] O esquema da v1 gerava {res_ids['n_colisoes_esquema_v1']} colisoes: "
              "o SHA-256 'genes_puros' da v1 NAO deve ser usado.")

    # ------------------------------------------------------------------ P2b
    print("\n[3] P2b - h5ad x features_ENSG_puro.tsv (hipotese: sufixo/caixa)")
    diag, interp = diagnosticar_molde(raw_ids, CAM_MOLDE_ENSG)
    if diag is not None:
        df_m, extra = diag
        print(df_m.to_string(index=False))
        print(f"   {extra}")
        salvar_df(df_m, os.path.join(P["preflight"], "P2b_molde_vs_h5ad.csv"), index=False)
        salvar_json(extra, os.path.join(P["preflight"], "P2b_molde_interpretacao.json"))
    else:
        print(f"   [pulado] {interp}")

    # ------------------------------------------------------------------ P2
    print("\n[4] P2 - Ordem dos genes x W (reproduzir o TSV SWeeP a partir do h5ad)")
    W, status_w, res_w = None, "NAO_VERIFICADO", {}
    if args.W and os.path.exists(args.W):
        tmp = tempfile.mkdtemp(prefix="w_", dir=P["preflight"])
        W = carregar_W(args.W, tmp)
        print(f"   -> W carregada: {W.shape} | max|W'W - I| = {orto_W(W):.2e}")
        rng_c = np.random.default_rng(SEED_CHECK)
        idx_c = np.sort(rng_c.choice(adata.n_obs, size=N_CEL_CHECK_W, replace=False))
        df_w, res_w = verificar_ordem_W(adata.X, X_sweep, W, idx_c)
        res_w["max_abs_W_T_W_menos_I"] = orto_W(W)
        status_w = res_w["veredito"]
        salvar_df(df_w, os.path.join(P["preflight"], "P2_W_vs_TSV_por_celula.csv"), index=False)
        salvar_json(res_w, os.path.join(P["preflight"], "P2_W_vs_TSV_resumo.json"))
        for k_, v_ in res_w.items():
            print(f"   {k_}: {v_}")
    else:
        print(f"   [AVISO] W nao encontrada em: {args.W}. Passe --W com o arquivo correto.")
    print(f"   >>> STATUS DA ORDEM W x GENES: {status_w}")

    # ------------------------------------------------------------------ P4
    print("\n[5] P4 - Evidencia de normalizacao do h5ad")
    ev_norm = evidencia_normalizacao(adata)
    for k_, v_ in ev_norm.items():
        print(f"   {k_}: {v_}")
    salvar_json(ev_norm, os.path.join(P["preflight"], "P4_normalizacao_evidencia.json"))

    w_ok = status_w == "CONFIRMADA"
    if args.so_preflight:
        print("\n[--so-preflight] Verificacoes concluidas. Nada foi treinado. Pasta:", P["preflight"])
        return
    if not w_ok and not args.continuar_sem_verificacao_w:
        print("\n[PARADA] A ordem dos genes x W nao esta CONFIRMADA. Resolva (ver P2/P2b) ou use "
              "--continuar-sem-verificacao-w (SWeeP ficara bloqueado para inferencia externa).")
        sys.exit(2)

    criar_lock(P, decisao, bracos)

    # ------------------------------------------------------------------ split + referencias
    print("\n[6] Split 90/10 e referencias genomicas")
    idx_tr, idx_va = criar_split_90_10(y, donors, cfg["seed_split"], P)
    ytr, yva = y[idx_tr], y[idx_va]
    art = P["artefatos_congelados"]
    salvar_txt(raw_ids, os.path.join(art, "genes_pan_61541_ids_originais.txt"))
    salvar_txt(chaves, os.path.join(art, "genes_pan_61541_chaves_mapeamento.txt"))

    # ------------------------------------------------------------------ treino e congelamento
    print("\n[7] Ajustando transformadores (90%), treinando e congelando")
    linhas_tab, info_art, auto = [], {}, []
    rng_t = np.random.default_rng(SEED_CHECK)
    for b in bracos:
        nome = b["nome"]
        print(f"\n--- {nome} ({b['dim']}D) ---")
        Xtr, Xva, info, pasta = congelar_braco(b, X_sweep, adata, idx_tr, idx_va, cfg, P, chaves, raw_ids)
        info_art[nome] = info
        p_va_canon = None
        for s in seeds:
            cam_meta = os.path.join(P["meta"], f"meta_final_{nome}_seed{s}.json")
            cam_pt = os.path.join(P["seeds"], f"mlp_{nome}_seed{s}.pt")
            cam_pva = os.path.join(P["seeds"], f"ypred_holdout_{nome}_seed{s}.npy")
            if RETOMANDO and os.path.exists(cam_meta) and os.path.exists(cam_pt) and os.path.exists(cam_pva):
                with open(cam_meta, encoding="utf-8") as f:
                    m = json.load(f)
                print(f"   [retomada seed {s}] ep={m['Epoca_Ideal']}/{m['Epocas_Rodadas']} | F1={m['F1_Macro_Val_ES']:.4f}")
            else:
                r = treinar_modelo_final(Xtr, ytr, Xva, yva, usar_pesos, s, cfg)
                estado, yp = r.pop("state_dict"), r.pop("y_pred_val_es")
                salvar_pt(estado, cam_pt)
                salvar_npy(yp, cam_pva)
                if s == seed_principal:
                    salvar_pt(estado, os.path.join(pasta, "mlp_canonico.pt"))
                    plotar_curva_final(P, nome, r)
                m = {"Representacao": nome, "Dimensao": b["dim"], "Pesos": decisao["pesos_escolhido"], "Seed": s,
                     "N_Treino_Final": int(len(idx_tr)), "N_Holdout_ES": int(len(idx_va)),
                     "Tempo_Prep_s": info["Tempo_Prep_s"], **r}
                salvar_json(m, cam_meta)
                print(f"   [seed {s}] ep={m['Epoca_Ideal']}/{m['Epocas_Rodadas']} | LossES={m['Loss_Val_CE']:.4f} | "
                      f"F1 hold-out ES={m['F1_Macro_Val_ES']:.4f} | {m['Tempo_s']:.0f}s")
            if s == seed_principal:
                p_va_canon = np.load(cam_pva)
            linhas_tab.append({"Representacao": nome, "Dimensao": b["dim"], "Pesos": m["Pesos"], "Seed": s,
                               "N_Treino_Final": m["N_Treino_Final"], "N_Holdout_ES": m["N_Holdout_ES"],
                               "Epoca_Ideal": m["Epoca_Ideal"], "Epocas_Rodadas": m["Epocas_Rodadas"],
                               "Loss_Holdout_CE": m["Loss_Val_CE"], "F1_Macro_Holdout_ES": m["F1_Macro_Val_ES"],
                               "Acc_Bal_Holdout_ES": m["Acc_Bal_Val_ES"], "Acc_Holdout_ES": m["Acc_Val_ES"],
                               "Tempo_Treino_s": m["Tempo_s"]})
        del Xtr, Xva

        # autoteste do modelo canonico a partir do DISCO
        print("   -> autoteste a partir do disco...")
        df_a = autoteste_braco(b, pasta, os.path.join(pasta, "mlp_canonico.pt"), X_sweep, adata, idx_va,
                               p_va_canon, chaves, raw_ids, W if w_ok else None, rng_t)
        auto.append(df_a)
        for _, r_ in df_a.iterrows():
            print(f"      {r_['Teste']}: {r_['Concordancia']:.5f}")
            assert r_["Concordancia"] >= 0.9995, f"AUTOTESTE FALHOU: {nome} / {r_['Teste']}"

    # ------------------------------------------------------------------ tabelas
    print("\n[8] Tabelas")
    df_tab = pd.DataFrame(linhas_tab)
    salvar_df(df_tab, os.path.join(P["etapa5"], "Tabela5_Treino_Final_POR_SEED.csv"), index=False)
    cols_res = ["Epoca_Ideal", "Loss_Holdout_CE", "F1_Macro_Holdout_ES", "Acc_Bal_Holdout_ES"]
    df_res = df_tab.groupby("Representacao", sort=False)[cols_res].agg(["mean", "std"])
    df_res.columns = [f"{a}_{b_}" for a, b_ in df_res.columns]
    salvar_df(df_res.reset_index(), os.path.join(P["etapa5"], "Tabela5a_Resumo_Modelos_Finais.csv"), index=False)
    salvar_df(pd.concat(auto, ignore_index=True), os.path.join(P["etapa5"], "Tabela5d_Autoteste_Ponta_a_Ponta.csv"),
              index=False)
    info_v1, df_v1 = comparar_com_v1(P, df_tab, idx_tr, idx_va)
    if df_v1 is not None:
        salvar_df(df_v1, os.path.join(P["etapa5"], "Tabela5c_Reprodutibilidade_vs_v1.csv"), index=False)
        print(f"   Split identico a v1: {info_v1['split_identico_v1']} | "
              f"max|dF1| vs v1 = {df_v1['Dif_F1'].abs().max():.5f}")

    # HVG final x Etapa 3 / folds LODO (como na v1)
    cam_f = os.path.join(art, "Baseline_HVG2000_PCA50", "hvg2000_ids_originais.txt")
    with open(cam_f, encoding="utf-8") as f:
        set_final = set(l.strip() for l in f if l.strip())
    linhas_j = []
    cam_e3 = os.path.join(args.run, "etapa3_teste_cego", "hvg2000_ensembl_ids_treino70.txt")
    fontes = [("Final_90pct_vs_Etapa3_70pct", cam_e3)]
    pasta_e4 = os.path.join(args.run, "etapa4_lodo", "hvg_ids")
    if os.path.isdir(pasta_e4):
        fontes += [(f"Final_90pct_vs_{a[:-4]}", os.path.join(pasta_e4, a)) for a in sorted(os.listdir(pasta_e4))
                   if a.startswith("hvg2000_fold") and a.endswith(".txt")]
    for rot, cam in fontes:
        if os.path.exists(cam):
            with open(cam, encoding="utf-8") as f:
                s_ = set(l.strip() for l in f if l.strip())
            inter = len(set_final & s_)
            linhas_j.append({"Comparacao": rot, "Intersecao_Genes": inter, "Jaccard": inter / max(len(set_final | s_), 1)})
    if linhas_j:
        salvar_df(pd.DataFrame(linhas_j), os.path.join(P["etapa5"], "Tabela5b_Comparacao_HVG_Final.csv"), index=False)

    # ------------------------------------------------------------------ proveniencia e manifesto
    print("\n[9] Proveniencia e manifesto v2")
    hashes = {"Metadados_Pan_Limpos.csv": sha256_arquivo(CAM_METADADOS),
              "genes_ids_originais": sha256_lista(raw_ids), "genes_chaves_mapeamento": sha256_lista(chaves)}
    if args.W and os.path.exists(args.W):
        hashes["W_" + os.path.basename(args.W)] = sha256_arquivo(args.W)
    if not args.sem_hash_grande:
        hashes["SWeeP_600_TSV"] = sha256_arquivo(CAM_SWEEP_600)
        hashes["h5ad_LogNorm"] = sha256_arquivo(CAM_H5AD_LOG)
    if SCRIPT_PATH and os.path.exists(SCRIPT_PATH):
        hashes["este_script"] = sha256_arquivo(SCRIPT_PATH)
    pacotes = sorted(f"{d.metadata['Name']}=={d.version}" for d in md.distributions() if d.metadata["Name"])
    salvar_txt(pacotes, os.path.join(P["etapa5"], "ambiente_pacotes.txt"))

    sweep_liberado = bool(w_ok)
    manifesto = {
        "versao_manifesto": "v2",
        "gerado_em": datetime.datetime.now().isoformat(timespec="seconds"),
        "coorte_referencia": "GSE267301 (Pan et al.)",
        "n_celulas_total": int(len(y)), "n_celulas_treino_final_90pct": int(len(idx_tr)),
        "n_celulas_holdout_es_10pct": int(len(idx_va)), "n_doadores": int(len(np.unique(donors))),
        "split": "90/10 estratificado por (doador x classe); hold-out usado so para early stopping",
        "seed_canonica_principal": int(seed_principal), "seeds_disponiveis": seeds,
        "esquema_pesos_congelado": decisao["pesos_escolhido"], "k_pca_sweep_congelado": int(decisao["k_escolhido"]),
        "dicionario_classes_base0_pytorch": {str(i): NOMES_CLASSES[i] for i in range(8)},
        "genes": {"n_genes": 61541, "chave_de_mapeamento": res_ids["chave_de_mapeamento_escolhida"],
                  "arquivo_chaves": "genes_pan_61541_chaves_mapeamento.txt",
                  "arquivo_ids_originais": "genes_pan_61541_ids_originais.txt",
                  "resumo_ids": res_ids},
        "verificacao_ordem_W": {"status": status_w, "resumo": res_w, "W_arquivo": args.W,
                                "sweep_liberado_para_inferencia_externa": sweep_liberado},
        "normalizacao_h5ad_evidencia": ev_norm,
        "reprodutibilidade_vs_v1": info_v1,
        "regras_inferencia_externa_A_REVISAR_antes_de_usar": {
            "nota": ("Estas regras sao DECISOES do autor, nao fatos sobre nenhuma coorte especifica. Devem ser "
                     "revisadas com o orientador e fixadas (com timestamp e hash deste arquivo) ANTES de qualquer "
                     "inferencia externa."),
            "ordem_dos_passos": ("1) reindexar CONTAGENS BRUTAS para a ordem do Pan com reindexar_para_pan() "
                                 "(genes ausentes = 0; IDs duplicados somados); 2) HVG: normalizar_log1p com o "
                                 "target_sum inferido do Pan; SWeeP: binarizar (x>0) e projetar com a MESMA W; "
                                 "3) transformador .npz; 4) MLP."),
            "normalizacao_Baseline_HVG": ("Primaria: normalizar APOS reindexar (universo de genes = Pan). "
                                          "Sensibilidade: normalizar antes de reindexar. Reportar ambas."),
            "genes_ausentes": ("Reportar cobertura dos 61.541 genes e dos 2.000 HVGs (cobertura_hvg). Limiar "
                               "minimo de cobertura a definir ANTES da inferencia."),
            "classes_ausentes_no_ground_truth": ("Reportar F1-macro restrito as classes presentes e a taxa de "
                                                 "predicoes nas classes ausentes."),
            "SWeeP": ("Bloqueado para inferencia externa enquanto verificacao_ordem_W.status != CONFIRMADA."),
            "escolha_do_modelo": ("Modelo canonico = seed %d (definida a priori). Ensemble das %d seeds = analise "
                                  "de sensibilidade." % (seed_principal, len(seeds))),
        },
        "artefatos_por_braco": {b["nome"]: {**info_art[b["nome"]],
                                            "mlp_canonico_pt": f"{b['nome']}/mlp_canonico.pt"} for b in bracos},
        "sha256": hashes,
        "ambiente": {"python": sys.version, "plataforma": platform.platform(), "torch": torch.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "scanpy": versao("scanpy"),
                     "scikit-learn": versao("scikit-learn"), "joblib": versao("joblib"),
                     "lista_completa": "ambiente_pacotes.txt"},
    }
    cam_man = os.path.join(P["etapa5"], "MANIFESTO_INFERENCIA_EXTERNA_V2.json")
    salvar_json(manifesto, cam_man)
    salvar_json(manifesto, os.path.join(art, "MANIFESTO_INFERENCIA_EXTERNA_V2.json"))
    salvar_txt([sha256_arquivo(cam_man) + "  MANIFESTO_INFERENCIA_EXTERNA_V2.json"],
               os.path.join(P["etapa5"], "MANIFESTO_V2.sha256"))

    pd.set_option("display.width", 220)
    print("\n[RESUMO - hold-out de early stopping (NAO e metrica de generalizacao; ver Etapas 3 e 4)]")
    print(df_res.round(4).to_string())
    print("\n" + "=" * 78)
    print(f"ETAPA 5 v2 CONCLUIDA. Ordem W x genes: {status_w} | SWeeP liberado p/ externo: {sweep_liberado}")
    print(f"Artefatos: {art}")
    print("=" * 78)


if __name__ == "__main__":
    main()