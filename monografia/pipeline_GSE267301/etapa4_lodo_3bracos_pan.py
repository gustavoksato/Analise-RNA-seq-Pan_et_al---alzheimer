#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETAPA 4 - LODO (Leave-One-Donor-Out) NO GSE267301 (Pan et al.) - 3 BRACOS PRINCIPAIS

Objetivo: estimar o desempenho em DOADOR NUNCA VISTO e comparar os 3 bracos nos MESMOS folds.

Entrada (da run da Etapa 2): decisao_hiperparametros.json (pesos, k) e config_etapas_0_a_2.json.
Hiperparametros ficam FIXOS em todos os folds (limitacao a declarar: escolhidos na validacao do split celular).

Para cada um dos 8 doadores retidos (fold = 1 doador):
    - treino+validacao = os outros 7 doadores; validacao interna = 15% estratificada (mesma para todos os bracos/seeds)
    - HVG (2.000, seurat), PCA e StandardScaler sao AJUSTADOS so no treino interno do fold
    - checkpoint pela loss de validacao interna (CE sem pesos), paciencia da Etapa 2
    - o doador retido so e usado para predicao
    - bracos: Baseline_HVG2000_PCA50 | SWeeP_600D_Puro | SWeeP_600D_PCA<k>  x seeds da Etapa 2

Saidas (etapa4_lodo/): predicoes por fold, predicoes OOF (58.326 celulas) por braco/seed, metricas por doador,
metricas pooled OOF (todos os doadores e sem ALSP), IC por bootstrap de DOADORES, bootstrap pareado entre bracos,
diferenca por doador, comparacao com o teste cego da Etapa 3 e estabilidade dos HVGs entre folds.

Politica de escrita: nenhum arquivo existente e sobrescrito. Resultados de cada fold x braco x seed sao salvos
assim que terminam; --retomar continua uma execucao interrompida.

Uso:
    python etapa4_lodo_3bracos_pan.py
    python etapa4_lodo_3bracos_pan.py --run /caminho/run_AAAAMMDD_HHMMSS
    python etapa4_lodo_3bracos_pan.py --retomar
"""

import os
import sys
import json
import time
import copy
import platform
import argparse
import datetime
from importlib import metadata as md

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
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

# ==============================================================================
# CAMINHOS E CONFIGURACAO
# ==============================================================================
BASE = "/media/gustavo-k-sato/D61CF48E1CF46B3F1/analisesPAN"
CAM_SWEEP_600 = "/media/gustavo-k-sato/D61CF48E1CF46B3F1/AIBIA/R_analises/SWeeP_600_Classes_Alzheimer_Oficial.tsv"
CAM_METADADOS = os.path.join(BASE, "Metadados_Pan_Limpos.csv")
CAM_H5AD_LOG = os.path.join(BASE, "Pan_58k_LogNorm_Para_BaselineHVG.h5ad")

DEFAULT_RUN = os.path.join(BASE, "resultados_MLP_GSE267301_v2", "run_20261102_201556")

N_HVG = 2000
FRAC_VAL_INTERNA = 0.15
N_BOOT_DOADOR = 10000
SEED_BOOT = 12345
DOADOR_EXCLUIR = "ALSP"      # escopo "Sem_ALSP": 7 doadores AD/controle
LIMIAR_SUPORTE = 20          # combinacoes doador x classe com menos celulas que isso sao sinalizadas

NOMES_CLASSES = ["1: Astrocytes", "2: Oligodendrocytes", "3: Exc. Neurons", "4: Inh. Neurons",
                 "5: OPCs", "6: Microglia", "7: Endothelial", "8: VLMCs"]
CURTO = [n.split(":")[1].strip() for n in NOMES_CLASSES]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    return decisao, cfg


def preparar_pastas(run_dir):
    run_dir = os.path.abspath(run_dir)
    e4 = os.path.join(run_dir, "etapa4_lodo")
    P = {"run": run_dir, "etapa4": e4, "etapa3": os.path.join(run_dir, "etapa3_teste_cego")}
    for sub in ["predicoes", "meta", "hvg_ids", "oof", "figuras"]:
        P[sub] = os.path.join(e4, sub)
        os.makedirs(P[sub], exist_ok=True)
    return P


def criar_lock(P, decisao, bracos):
    cam = os.path.join(P["etapa4"], "LODO_LOCK.json")
    if os.path.exists(cam):
        if not RETOMANDO:
            raise FileExistsError("A Etapa 4 desta run JA FOI INICIADA (LODO_LOCK.json existe). "
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
# DADOS, AUDITORIA E FOLDS
# ==============================================================================
def carregar_e_auditar():
    for nome, cam in [("SWeeP TSV", CAM_SWEEP_600), ("Metadados", CAM_METADADOS), ("h5ad LogNorm", CAM_H5AD_LOG)]:
        if not os.path.exists(cam):
            raise FileNotFoundError(f"{nome} nao encontrado: {cam}")
    df_sw = pd.read_csv(CAM_SWEEP_600, sep="\t")
    y_tsv = df_sw["Cell_Type"].values.astype(int)
    X = df_sw.drop(columns=["Cell_Type"]).values.astype(np.float32)
    del df_sw
    df_meta = pd.read_csv(CAM_METADADOS)

    ad = sc.read_h5ad(CAM_H5AD_LOG, backed="r")
    obs_index = ad.obs.index.values.astype(str)
    n_obs, n_vars = ad.n_obs, ad.n_vars
    ad.file.close()
    del ad

    for col in ["Barcode", "Donor", "Cell_Type_Num"]:
        assert col in df_meta.columns, f"Coluna ausente nos metadados: {col}"
    assert X.shape[0] == len(y_tsv) == len(df_meta) == n_obs, "Tamanhos divergentes entre SWeeP, metadados e h5ad"
    assert np.array_equal(obs_index, df_meta["Barcode"].astype(str).values), "Barcodes do h5ad != metadados"
    assert np.array_equal(y_tsv, df_meta["Cell_Type_Num"].values.astype(int)), "Rotulos do TSV != metadados"
    assert X.shape[1] == 600 and n_vars == 61541
    assert not np.isnan(X).any()
    donors = df_meta["Donor"].astype(str).values
    print(f"   -> Auditoria OK: {X.shape[0]} celulas, {len(np.unique(donors))} doadores.")
    return X, y_tsv - 1, donors, df_meta


def montar_folds(y, donors):
    doadores = sorted(np.unique(donors))
    assert DOADOR_EXCLUIR in doadores, f"Doador {DOADOR_EXCLUIR} nao encontrado: {doadores}"
    folds = []
    for i, d in enumerate(doadores, 1):
        idx_te = np.where(donors == d)[0]
        idx_trval = np.where(donors != d)[0]
        assert len(np.intersect1d(donors[idx_trval], [d])) == 0, "Doador retido vazou para o treino!"
        assert set(np.unique(y[idx_te])) == set(range(8)), f"Doador {d} sem todas as 8 classes no teste"
        assert set(np.unique(y[idx_trval])) == set(range(8)), f"Treino do fold {d} sem todas as classes"
        folds.append({"fold": i, "doador": d, "idx_te": idx_te, "idx_trval": idx_trval})
    return folds


def dividir_interno(idx_trval, y, seed_split):
    return train_test_split(idx_trval, test_size=FRAC_VAL_INTERNA, random_state=seed_split, stratify=y[idx_trval])


# ==============================================================================
# FEATURES POR BRACO (ajustadas so no treino interno do fold)
# ==============================================================================
CACHE = {"adata": None}


def obter_adata():
    if CACHE["adata"] is None:
        print("   -> Carregando h5ad log-norm completo (uma unica vez)...")
        CACHE["adata"] = sc.read_h5ad(CAM_H5AD_LOG)
    return CACHE["adata"]


def construir_features(braco, X, fo, idx_tr, idx_va, idx_te, cfg, P):
    tipo, seed_split = braco["tipo"], cfg["seed_split"]
    if tipo == "sweep_puro":
        sc_ = StandardScaler()
        return sc_.fit_transform(X[idx_tr]), sc_.transform(X[idx_va]), sc_.transform(X[idx_te])
    if tipo == "sweep_pca":
        pca = PCA(n_components=cfg["k_max_pca"], random_state=seed_split)
        Ztr = pca.fit_transform(X[idx_tr])
        Zva, Zte = pca.transform(X[idx_va]), pca.transform(X[idx_te])
        k = braco["dim"]
        sc_ = StandardScaler()
        return sc_.fit_transform(Ztr[:, :k]), sc_.transform(Zva[:, :k]), sc_.transform(Zte[:, :k])
    if tipo == "hvg_log":
        adata = obter_adata()
        ad_tr = adata[idx_tr, :].copy()
        sc.pp.highly_variable_genes(ad_tr, n_top_genes=N_HVG, flavor="seurat")
        mask = ad_tr.var["highly_variable"].values
        del ad_tr
        assert int(mask.sum()) == N_HVG, f"HVGs selecionados: {int(mask.sum())}"
        cols = np.where(mask)[0]
        salvar_txt(list(adata.var_names[mask].astype(str)),
                   os.path.join(P["hvg_ids"], f"hvg2000_fold{fo['fold']}_{fo['doador']}.txt"))
        Xall = adata.X

        def fatia(idx):
            M = Xall[idx][:, cols]
            return (M.toarray() if sp.issparse(M) else np.asarray(M)).astype(np.float32)

        Htr, Hva, Hte = fatia(idx_tr), fatia(idx_va), fatia(idx_te)
        pca = PCA(n_components=50, random_state=seed_split)
        Ztr = pca.fit_transform(Htr)
        Zva, Zte = pca.transform(Hva), pca.transform(Hte)
        del Htr, Hva, Hte
        sc_ = StandardScaler()
        return sc_.fit_transform(Ztr), sc_.transform(Zva), sc_.transform(Zte)
    raise ValueError(f"Tipo de braco desconhecido: {tipo}")


# ==============================================================================
# MLP E TREINO (nao recebe y do doador retido)
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


def treinar_e_predizer(Xtr, ytr, Xva, yva, Xte, usar_pesos, seed, cfg):
    fixar_seed(seed)
    t0 = time.time()
    nc = cfg["n_classes"]
    modelo = ClassificadorCelular(Xtr.shape[1], nc).to(DEVICE)

    if usar_pesos:
        pres = np.unique(ytr)
        pesos = np.ones(nc, dtype=np.float32)
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
    ld_te = DataLoader(TensorDataset(torch.tensor(Xte, dtype=torch.float32)),
                       batch_size=cfg["batch_eval"], shuffle=False)

    melhor, estado, ep_ideal, sem_melhora = float("inf"), None, 0, 0
    hist_va = []
    for ep in range(cfg["epocas_max"]):
        modelo.train()
        for bx, by in ld_tr:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            loss_tr_fn(modelo(bx), by).backward()
            opt.step()
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
    modelo.eval()
    with torch.no_grad():
        p_va = np.concatenate([modelo(bx.to(DEVICE)).argmax(1).cpu().numpy() for bx, _ in ld_va])
        p_te = np.concatenate([modelo(bx.to(DEVICE)).argmax(1).cpu().numpy() for (bx,) in ld_te])

    return {
        "y_pred": p_te.astype(np.int64),
        "F1_Macro_Val": float(f1_score(yva, p_va, average="macro", labels=list(range(nc)), zero_division=0)),
        "Loss_Val_CE": float(melhor),
        "Epoca_Ideal": int(ep_ideal),
        "Epocas_Rodadas": int(len(hist_va)),
        "Tempo_s": float(time.time() - t0),
    }


# ==============================================================================
# METRICAS A PARTIR DE MATRIZES DE CONFUSAO (linhas = real, colunas = predito)
# ==============================================================================
def cm_de(y, p):
    return np.bincount(y * 8 + p, minlength=64).reshape(8, 8)


def stats_de_cm(cm):
    """Aceita (8,8) ou (...,8,8). F1/recall/precisao 0 quando o denominador e 0."""
    cm = np.asarray(cm, dtype=float)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    sup = cm.sum(-1)
    prd = cm.sum(-2)
    den = sup + prd                       # = 2*tp + fp + fn
    f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)
    rec = np.where(sup > 0, tp / np.maximum(sup, 1), 0.0)
    prec = np.where(prd > 0, tp / np.maximum(prd, 1), 0.0)
    tot = sup.sum(-1)
    return {"f1": f1, "rec": rec, "prec": prec, "sup": sup,
            "F1_Macro": f1.mean(-1), "Acuracia_Balanceada": rec.mean(-1),
            "F1_Weighted": (f1 * sup).sum(-1) / np.maximum(tot, 1),
            "Acuracia": tp.sum(-1) / np.maximum(tot, 1)}


def bootstrap_doadores(cms, idx_escopo, n_boot, seed):
    """Reamostra DOADORES com reposicao (mesmas reamostras para todos os bracos/seeds => pareado) e
    recalcula F1-macro e acuracia balanceada pooled somando as matrizes de confusao por doador."""
    rng = np.random.default_rng(seed)
    D = len(idx_escopo)
    S = rng.integers(0, D, size=(n_boot, D))
    out = {}
    for chave, cm_d in cms.items():
        cm_s = cm_d[idx_escopo]                      # (D,8,8)
        st = stats_de_cm(cm_s[S].sum(axis=1))        # (B,8,8) -> metricas (B,)
        out[chave] = {"F1_Macro": st["F1_Macro"], "Acuracia_Balanceada": st["Acuracia_Balanceada"]}
    return out


def resumo_doadores(df_meta, doadores):
    cols = [c for c in ["Case_No_Case", "NP.Diagnosis", "Brain.Region", "Sex", "Age"] if c in df_meta.columns]
    linhas = []
    for d in doadores:
        sub = df_meta[df_meta["Donor"].astype(str) == d]
        linha = {"Doador": d, "N_Celulas": len(sub)}
        for c in cols:
            linha[c] = ";".join(sorted(map(str, sub[c].dropna().unique())))
        linhas.append(linha)
    return pd.DataFrame(linhas)


# ==============================================================================
# CONSOLIDACAO (le predicoes salvas; nao treina nada)
# ==============================================================================
def consolidar(P, bracos, seeds, folds, y, donors, df_meta, seed_principal):
    doadores = [fo["doador"] for fo in folds]
    D, n = len(doadores), len(y)
    nomes = [b["nome"] for b in bracos]
    escopos = {"Todos": list(range(D)),
               "Sem_ALSP": [j for j, d in enumerate(doadores) if d != DOADOR_EXCLUIR]}

    fold_por_celula = np.empty(n, dtype=int)
    for fo in folds:
        fold_por_celula[fo["idx_te"]] = fo["fold"]
    salvar_npy(fold_por_celula, os.path.join(P["oof"], "fold_por_celula.npy"))
    salvar_npy(y, os.path.join(P["oof"], "y_true.npy"))
    salvar_npy(np.asarray(donors, dtype=str), os.path.join(P["oof"], "doador_por_celula.npy"))

    oof, cms, mt = {}, {}, {}
    for nm in nomes:
        for s in seeds:
            arr = np.full(n, -1, dtype=np.int64)
            cm_d = np.zeros((D, 8, 8), dtype=np.int64)
            for j, fo in enumerate(folds):
                tag = f"fold{fo['fold']}_{fo['doador']}_{nm}_seed{s}"
                yp = np.load(os.path.join(P["predicoes"], f"ypred_{tag}.npy"))
                assert len(yp) == len(fo["idx_te"]), f"Predicao com tamanho errado: {tag}"
                arr[fo["idx_te"]] = yp
                cm_d[j] = cm_de(y[fo["idx_te"]], yp)
                with open(os.path.join(P["meta"], f"meta_{tag}.json"), encoding="utf-8") as f:
                    mt[(nm, s, fo["doador"])] = json.load(f)
            assert (arr >= 0).all(), f"OOF incompleto: {nm} seed {s}"
            salvar_npy(arr, os.path.join(P["oof"], f"y_pred_oof_{nm}_seed{s}.npy"))
            oof[(nm, s)], cms[(nm, s)] = arr, cm_d

    # ---------------- por doador ----------------
    linhas_d = []
    for (nm, s), cm_d in cms.items():
        st = stats_de_cm(cm_d)
        assert (st["sup"] > 0).all(), "Doador sem alguma classe no teste"
        for j, d in enumerate(doadores):
            m = mt[(nm, s, d)]
            linha = {"Representacao": nm, "Seed": s, "Doador": d, "N_Celulas": int(cm_d[j].sum()),
                     "Acuracia": st["Acuracia"][j], "Acuracia_Balanceada": st["Acuracia_Balanceada"][j],
                     "F1_Weighted": st["F1_Weighted"][j], "F1_Macro": st["F1_Macro"][j],
                     "Epoca_Ideal": m["Epoca_Ideal"], "F1_Macro_Val_Interna": m["F1_Macro_Val"]}
            for ci, cn in enumerate(CURTO):
                linha[f"F1_{cn}"] = st["f1"][j, ci]
                linha[f"N_{cn}"] = int(st["sup"][j, ci])
            linhas_d.append(linha)
    df_d = pd.DataFrame(linhas_d)
    salvar_df(df_d, os.path.join(P["etapa4"], "Tabela4b_Por_Doador_POR_SEED.csv"), index=False)

    pivot = df_d.pivot_table(index="Doador", columns="Representacao", values="F1_Macro", aggfunc="mean")[nomes]
    pivot = pivot.reindex(doadores)
    base = df_d[(df_d["Representacao"] == nomes[0]) & (df_d["Seed"] == seeds[0])].set_index("Doador")
    pivot.insert(0, "N_Celulas", base["N_Celulas"])
    pivot.insert(1, "N_Endothelial", base["N_Endothelial"])
    pivot.insert(2, "N_VLMCs", base["N_VLMCs"])
    pivot.insert(3, "N_Microglia", base["N_Microglia"])
    salvar_df(pivot.reset_index(), os.path.join(P["etapa4"], "Tabela4c_F1Macro_Por_Doador_Media_Seeds.csv"), index=False)
    salvar_df(resumo_doadores(df_meta, doadores), os.path.join(P["etapa4"], "Tabela4_Doadores_Metadados.csv"), index=False)

    suporte = df_d[(df_d["Representacao"] == nomes[0]) & (df_d["Seed"] == seeds[0])][
        ["Doador"] + [f"N_{c}" for c in CURTO]]
    baixo = [{"Doador": r["Doador"], "Classe": c, "N_Celulas": int(r[f"N_{c}"])}
             for _, r in suporte.iterrows() for c in CURTO if r[f"N_{c}"] < LIMIAR_SUPORTE]
    salvar_df(pd.DataFrame(baixo, columns=["Doador", "Classe", "N_Celulas"]),
              os.path.join(P["etapa4"], "Tabela4c2_Doador_Classe_suporte_baixo.csv"), index=False)

    pares = [(a, b) for i, a in enumerate(nomes) for b in nomes[i + 1:]]
    linhas_dif = []
    for a, b in pares:
        dlt = pivot[a] - pivot[b]
        for d in doadores:
            linhas_dif.append({"Braco_A": a, "Braco_B": b, "Doador": d, "Delta_F1Macro_A_menos_B": dlt[d]})
        linhas_dif.append({"Braco_A": a, "Braco_B": b, "Doador": "N_DOADORES_A_GANHA",
                           "Delta_F1Macro_A_menos_B": int((dlt > 0).sum())})
    salvar_df(pd.DataFrame(linhas_dif), os.path.join(P["etapa4"], "Tabela4e_Diferenca_Por_Doador.csv"), index=False)

    # ---------------- pooled OOF (todos / sem ALSP) ----------------
    linhas_p, linhas_c = [], []
    est = {}
    for esc, idx in escopos.items():
        for (nm, s), cm_d in cms.items():
            cm_sum = cm_d[idx].sum(0)
            st = stats_de_cm(cm_sum)
            est[(esc, nm, s)] = {"F1_Macro": float(st["F1_Macro"]), "Acuracia_Balanceada": float(st["Acuracia_Balanceada"])}
            linhas_p.append({"Representacao": nm, "Seed": s, "Escopo": esc, "N_Celulas": int(cm_sum.sum()),
                             "N_Doadores": len(idx), "Acuracia": float(st["Acuracia"]),
                             "Acuracia_Balanceada": float(st["Acuracia_Balanceada"]),
                             "F1_Macro": float(st["F1_Macro"]), "F1_Weighted": float(st["F1_Weighted"])})
            for ci, cn in enumerate(NOMES_CLASSES):
                linhas_c.append({"Representacao": nm, "Seed": s, "Escopo": esc, "Classe": cn,
                                 "Precisao": st["prec"][ci], "Recall": st["rec"][ci], "F1": st["f1"][ci],
                                 "Suporte": int(st["sup"][ci])})
    df_p = pd.DataFrame(linhas_p)
    salvar_df(df_p, os.path.join(P["etapa4"], "Tabela4_LODO_Pooled_POR_SEED.csv"), index=False)
    salvar_df(pd.DataFrame(linhas_c), os.path.join(P["etapa4"], "Tabela4d_Pooled_Por_Classe_LONG.csv"), index=False)

    cols = ["Acuracia", "Acuracia_Balanceada", "F1_Macro", "F1_Weighted"]
    res = df_p.groupby(["Escopo", "Representacao"], sort=False)[cols].agg(["mean", "std"])
    res.columns = [f"{a}_{b}" for a, b in res.columns]
    res = res.reset_index()
    salvar_df(res, os.path.join(P["etapa4"], "Tabela4a_LODO_Pooled_Resumo_Media_DP.csv"), index=False)

    # ---------------- bootstrap por doador ----------------
    linhas_ic, linhas_par = [], []
    for k_esc, (esc, idx) in enumerate(escopos.items()):
        boot = bootstrap_doadores(cms, idx, N_BOOT_DOADOR, SEED_BOOT + k_esc)
        for (nm, s), bm in boot.items():
            for met in ["F1_Macro", "Acuracia_Balanceada"]:
                linhas_ic.append({"Representacao": nm, "Seed": s, "Escopo": esc, "Metrica": met,
                                  "Estimativa": est[(esc, nm, s)][met],
                                  "IC95_inf": float(np.percentile(bm[met], 2.5)),
                                  "IC95_sup": float(np.percentile(bm[met], 97.5)),
                                  "N_Doadores": len(idx), "N_boot": N_BOOT_DOADOR})
        for s in seeds:
            for a, b in pares:
                for met in ["F1_Macro", "Acuracia_Balanceada"]:
                    dlt = boot[(a, s)][met] - boot[(b, s)][met]
                    linhas_par.append({"Seed": s, "Escopo": esc, "Metrica": met, "Braco_A": a, "Braco_B": b,
                                       "Delta_A_menos_B": est[(esc, a, s)][met] - est[(esc, b, s)][met],
                                       "IC95_inf": float(np.percentile(dlt, 2.5)),
                                       "IC95_sup": float(np.percentile(dlt, 97.5)),
                                       "Frac_boot_delta_positivo": float((dlt > 0).mean()),
                                       "N_Doadores": len(idx), "N_boot": N_BOOT_DOADOR})
    df_ic = pd.DataFrame(linhas_ic)
    df_par = pd.DataFrame(linhas_par)
    salvar_df(df_ic, os.path.join(P["etapa4"], "Tabela4f_IC_Bootstrap_Doadores.csv"), index=False)
    salvar_df(df_par, os.path.join(P["etapa4"], "Tabela4g_Bootstrap_Pareado_Doadores.csv"), index=False)

    # ---------------- comparacao com o teste cego da Etapa 3 ----------------
    comp = None
    cam3 = os.path.join(P["etapa3"], "Tabela2a_Resumo_Media_DP.csv")
    if os.path.exists(cam3):
        t3 = pd.read_csv(cam3)
        lodo = res[res["Escopo"] == "Todos"].set_index("Representacao")
        linhas = []
        for nm in nomes:
            if nm in set(t3["Representacao"]):
                r3 = t3[t3["Representacao"] == nm].iloc[0]
                linhas.append({"Representacao": nm,
                               "F1Macro_TesteCego_Etapa3": r3["F1_Macro_mean"],
                               "F1Macro_LODO_Pooled": lodo.loc[nm, "F1_Macro_mean"],
                               "Delta_LODO_menos_Teste": lodo.loc[nm, "F1_Macro_mean"] - r3["F1_Macro_mean"],
                               "AccBal_TesteCego_Etapa3": r3["Acuracia_Balanceada_mean"],
                               "AccBal_LODO_Pooled": lodo.loc[nm, "Acuracia_Balanceada_mean"]})
        comp = pd.DataFrame(linhas)
        salvar_df(comp, os.path.join(P["etapa4"], "Tabela4h_Comparacao_Etapa3_vs_LODO.csv"), index=False)

    # ---------------- estabilidade dos HVGs entre folds ----------------
    jac = None
    conjuntos = {}
    for fo in folds:
        cam = os.path.join(P["hvg_ids"], f"hvg2000_fold{fo['fold']}_{fo['doador']}.txt")
        if os.path.exists(cam):
            with open(cam, encoding="utf-8") as f:
                conjuntos[fo["doador"]] = set(l.strip() for l in f if l.strip())
    if len(conjuntos) >= 2:
        ds = list(conjuntos)
        jac = pd.DataFrame([[len(conjuntos[a] & conjuntos[b]) / len(conjuntos[a] | conjuntos[b]) for b in ds]
                            for a in ds], index=ds, columns=ds)
        salvar_df(jac.reset_index(), os.path.join(P["etapa4"], "Tabela4i_HVG_Jaccard_Entre_Folds.csv"), index=False)

    # ---------------- figuras ----------------
    for nm in nomes:
        cm_all = cms[(nm, seed_principal)].sum(0)
        cmn = cm_all / np.maximum(cm_all.sum(1, keepdims=True), 1) * 100
        fig, ax = plt.subplots(figsize=(9.5, 7.5))
        sns.heatmap(cmn, annot=True, fmt=".1f", cmap="Blues", xticklabels=NOMES_CLASSES, yticklabels=NOMES_CLASSES, ax=ax)
        ax.set_xlabel("Predicao da MLP")
        ax.set_ylabel("Classe real (Pan et al.)")
        ax.set_title(f"LODO pooled OOF (%) - {nm} - seed {seed_principal}")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        salvar_fig(fig, os.path.join(P["figuras"], f"matriz_confusao_OOF_{nm}.png"))

    fig, ax = plt.subplots(figsize=(10, 4.8))
    larg = 0.8 / len(nomes)
    xs = np.arange(len(doadores))
    for i, nm in enumerate(nomes):
        ax.bar(xs + i * larg, pivot.loc[doadores, nm].values, width=larg, label=nm)
    ax.set_xticks(xs + larg * (len(nomes) - 1) / 2)
    ax.set_xticklabels(doadores)
    ax.set_ylim(max(0.0, np.nanmin(pivot[nomes].values) - 0.03), 1.0)
    ax.set_ylabel("F1-macro por doador (media das seeds)")
    ax.set_xlabel("Doador retido")
    ax.set_title("LODO: F1-macro por doador")
    ax.legend()
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    salvar_fig(fig, os.path.join(P["figuras"], "F1macro_por_doador.png"))

    return res, pivot, df_ic, df_par, comp, jac, df_p


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    global RETOMANDO
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN, help="Pasta run_* da Etapa 2")
    ap.add_argument("--retomar", action="store_true", help="Continua uma execucao interrompida da Etapa 4")
    args = ap.parse_args()
    RETOMANDO = args.retomar

    decisao, cfg = carregar_run(args.run)
    P = preparar_pastas(args.run)
    bracos = definir_bracos(decisao)
    usar_pesos = decisao["pesos_escolhido"] == "Balanced"
    seeds = cfg["seeds_treino"]
    seed_principal = seeds[0]

    print("=" * 78)
    print("ETAPA 4 - LODO (1 doador retido por fold; hiperparametros fixos da Etapa 2)")
    print("=" * 78)
    print(f"[INFO] Dispositivo: {DEVICE}")
    print(f"[INFO] Run da Etapa 2: {P['run']}")
    print(f"[INFO] Pesos={decisao['pesos_escolhido']} | k={decisao['k_escolhido']} | seeds={seeds}")
    print(f"[INFO] Bracos: {[b['nome'] for b in bracos]}")

    criar_lock(P, decisao, bracos)
    salvar_json({
        "decisao_etapa2": decisao, "cfg_etapa2": cfg, "bracos": bracos, "FRAC_VAL_INTERNA": FRAC_VAL_INTERNA,
        "N_BOOT_DOADOR": N_BOOT_DOADOR, "SEED_BOOT": SEED_BOOT, "DOADOR_EXCLUIR": DOADOR_EXCLUIR,
        "ambiente": {"python": sys.version, "plataforma": platform.platform(), "torch": torch.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "scanpy": versao("scanpy"),
                     "scikit-learn": versao("scikit-learn"), "device": str(DEVICE)},
    }, os.path.join(P["etapa4"], "config_etapa4.json"))

    print("\n[1/3] Carregando dados, auditando e montando folds...")
    X, y, donors, df_meta = carregar_e_auditar()
    folds = montar_folds(y, donors)
    salvar_json([{"fold": f["fold"], "doador": f["doador"], "n_teste": len(f["idx_te"]),
                  "n_trval": len(f["idx_trval"])} for f in folds], os.path.join(P["etapa4"], "folds.json"))
    print("   -> Folds: " + " | ".join(f"{f['fold']}:{f['doador']} ({len(f['idx_te'])})" for f in folds))
    print(f"   -> {len(folds)} folds x {len(bracos)} bracos x {len(seeds)} seeds = "
          f"{len(folds) * len(bracos) * len(seeds)} treinos")

    print("\n[2/3] Treinando por fold...")
    for fo in folds:
        f, d = fo["fold"], fo["doador"]
        idx_tr, idx_va = dividir_interno(fo["idx_trval"], y, cfg["seed_split"])
        idx_te = fo["idx_te"]
        assert d not in set(donors[idx_tr]) and d not in set(donors[idx_va]), "Doador retido no treino/validacao!"
        salvar_json({"fold": f, "doador": d, "n_treino": len(idx_tr), "n_val_interna": len(idx_va),
                     "n_teste_doador": len(idx_te)}, os.path.join(P["meta"], f"fold{f}_{d}_info.json"))
        print(f"\n=== Fold {f}/{len(folds)} | doador retido: {d} ({len(idx_te)} celulas) | "
              f"treino={len(idx_tr)} val={len(idx_va)} ===")
        for b in bracos:
            pend = [s for s in seeds
                    if not (os.path.exists(os.path.join(P["predicoes"], f"ypred_fold{f}_{d}_{b['nome']}_seed{s}.npy"))
                            and os.path.exists(os.path.join(P["meta"], f"meta_fold{f}_{d}_{b['nome']}_seed{s}.json")))]
            if not pend:
                print(f"   {b['nome']}: completo (pulando)")
                continue
            t0 = time.time()
            Xtr, Xva, Xte = construir_features(b, X, fo, idx_tr, idx_va, idx_te, cfg, P)
            print(f"   {b['nome']} ({b['dim']}D): features em {time.time() - t0:.0f}s | seeds pendentes {pend}")
            for s in pend:
                r = treinar_e_predizer(Xtr, y[idx_tr], Xva, y[idx_va], Xte, usar_pesos, s, cfg)
                yp = r.pop("y_pred")
                tag = f"fold{f}_{d}_{b['nome']}_seed{s}"
                salvar_npy(yp, os.path.join(P["predicoes"], f"ypred_{tag}.npy"))
                r.update({"Pesos": decisao["pesos_escolhido"], "Braco": b["nome"], "Seed": s, "Doador": d, "Fold": f})
                salvar_json(r, os.path.join(P["meta"], f"meta_{tag}.json"))
                acc = float((yp == y[idx_te]).mean())
                f1m = float(f1_score(y[idx_te], yp, average="macro", labels=list(range(8)), zero_division=0))
                print(f"      [seed {s:>4}] epoca={r['Epoca_Ideal']:02d}/{r['Epocas_Rodadas']:02d} | "
                      f"F1-macro val interna={r['F1_Macro_Val']:.4f} | doador: acc={acc:.4f} F1-macro={f1m:.4f} | "
                      f"{r['Tempo_s']:.0f}s")
            del Xtr, Xva, Xte
    CACHE["adata"] = None

    print("\n[3/3] Consolidando OOF, tabelas, bootstrap por doador e figuras...")
    res, pivot, df_ic, df_par, comp, jac, df_p = consolidar(P, bracos, seeds, folds, y, donors, df_meta, seed_principal)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    print("\n[LODO POOLED OOF - media e DP entre seeds]")
    print(res.round(4).to_string(index=False))
    print("\n[F1-MACRO POR DOADOR - media das seeds]")
    print(pivot.round(4).to_string())
    ic_sp = df_ic[(df_ic["Seed"] == seed_principal) & (df_ic["Metrica"] == "F1_Macro")]
    print(f"\n[IC 95% BOOTSTRAP DE DOADORES - F1-macro pooled - seed {seed_principal}]")
    print(ic_sp.round(4).to_string(index=False))
    par_sp = df_par[(df_par["Seed"] == seed_principal) & (df_par["Metrica"] == "F1_Macro")]
    print(f"\n[BOOTSTRAP PAREADO POR DOADOR - dF1-macro - seed {seed_principal}]")
    print(par_sp.round(4).to_string(index=False))
    if comp is not None:
        print("\n[COMPARACAO: teste cego (split por celula, Etapa 3) vs LODO pooled]")
        print(comp.round(4).to_string(index=False))
        print("   (treinos e conjuntos de avaliacao diferentes: so a agregacao de metrica e a mesma)")
    if jac is not None:
        mask = ~np.eye(len(jac), dtype=bool)
        print(f"\n[ESTABILIDADE DOS HVGs] Jaccard medio entre folds: {jac.values[mask].mean():.3f} "
              f"(min {jac.values[mask].min():.3f})")

    salvar_json({"concluido_em": datetime.datetime.now().isoformat(timespec="seconds"),
                 "bracos": [b["nome"] for b in bracos], "seeds": seeds,
                 "obs": "Hiperparametros fixos (escolhidos na validacao do split celular). Validacao interna "
                        "(15%) vem dos 7 doadores de treino. IC por bootstrap de 8 doadores e largo e ilustrativo."},
                os.path.join(P["etapa4"], "etapa4_concluida.json"))
    print("\n" + "=" * 78)
    print(f"ETAPA 4 CONCLUIDA. Saidas em: {P['etapa4']}")
    print("=" * 78)


if __name__ == "__main__":
    main()




