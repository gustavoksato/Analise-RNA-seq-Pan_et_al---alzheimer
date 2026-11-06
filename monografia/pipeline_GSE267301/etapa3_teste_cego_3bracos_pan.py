#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETAPA 3 - TESTE CEGO (GSE267301 / Pan et al.) - 3 BRACOS PRINCIPAIS x 3 SEEDS

Le da run da Etapa 2:
    decisao_hiperparametros.json   (pesos e k, escolhidos so na validacao)
    config_etapas_0_a_2.json       (hiperparametros de treino, seeds, seed do split)
    split_indices_70_15_15.npz     (indices de treino/validacao/teste)

Bracos (RODAR_ABLACOES = False):
    1. Baseline_HVG2000_PCA50   2.000 HVGs (log-norm, ajustados SO no treino) -> PCA50 -> scaler
    2. SWeeP_600D_Puro          SWeeP 600D -> scaler
    3. SWeeP_600D_PCA<k>        SWeeP 600D -> PCA50 (so treino) -> primeiras k componentes -> scaler
Todos usam o mesmo esquema de pesos (decisao unica da Etapa 2).

Protocolo do teste cego:
    - Early stopping/checkpoint pela loss de VALIDACAO (CE sem pesos, paciencia da Etapa 2).
    - O modelo NUNCA ve o teste durante o treino (a funcao de treino nao recebe y_teste).
    - Cada (braco, seed) e avaliado uma unica vez; predicoes salvas em .npy.
    - Um arquivo TESTE_CEGO_LOCK.json impede reexecutar a Etapa 3 do zero na mesma run.

Politica de escrita: nenhum arquivo existente e sobrescrito.

Uso:
    python etapa3_teste_cego_3bracos_pan.py                 # usa DEFAULT_RUN
    python etapa3_teste_cego_3bracos_pan.py --run /caminho/run_AAAAMMDD_HHMMSS
    python etapa3_teste_cego_3bracos_pan.py --retomar       # continua execucao interrompida
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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support, confusion_matrix,
                             classification_report)

# ==============================================================================
# CAMINHOS E CONFIGURACAO
# ==============================================================================
BASE = "/media/gustavo-k-sato/D61CF48E1CF46B3F1/analisesPAN"
CAM_SWEEP_600 = "/media/gustavo-k-sato/D61CF48E1CF46B3F1/AIBIA/R_analises/SWeeP_600_Classes_Alzheimer_Oficial.tsv"
CAM_METADADOS = os.path.join(BASE, "Metadados_Pan_Limpos.csv")
CAM_H5AD_LOG = os.path.join(BASE, "Pan_58k_LogNorm_Para_BaselineHVG.h5ad")

DEFAULT_RUN = os.path.join(BASE, "resultados_MLP_GSE267301_v2", "run_20261102_201556")

RODAR_ABLACOES = False   # True adiciona Ablacao_HVG2000_Binario_PCA50 (pendente de decisao do orientador)
N_BOOT = 2000            # reamostragens do bootstrap pareado (nivel de celula)
SEED_BOOT = 12345
N_HVG = 2000

NOMES_CLASSES = ["1: Astrocytes", "2: Oligodendrocytes", "3: Exc. Neurons", "4: Inh. Neurons",
                 "5: OPCs", "6: Microglia", "7: Endothelial", "8: VLMCs"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------------------
# Escrita SEM overwrite
# ------------------------------------------------------------------------------
RETOMANDO = False


def _checar_novo(caminho):
    """True se pode escrever; False se deve pular (retomada); erro se sobrescreveria."""
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


def salvar_txt(texto, caminho):
    if _checar_novo(caminho):
        with open(caminho, "w", encoding="utf-8") as f:
            f.write(texto if isinstance(texto, str) else "\n".join(map(str, texto)) + "\n")


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
# LEITURA DA RUN DA ETAPA 2
# ==============================================================================
def carregar_run(run_dir):
    run_dir = os.path.abspath(run_dir)
    nec = ["decisao_hiperparametros.json", "config_etapas_0_a_2.json", "split_indices_70_15_15.npz"]
    for n in nec:
        if not os.path.exists(os.path.join(run_dir, n)):
            raise FileNotFoundError(f"Arquivo da Etapa 2 ausente: {os.path.join(run_dir, n)}")
    with open(os.path.join(run_dir, nec[0]), encoding="utf-8") as f:
        decisao = json.load(f)
    with open(os.path.join(run_dir, nec[1]), encoding="utf-8") as f:
        cfg = json.load(f)["cfg"]
    sp_ = np.load(os.path.join(run_dir, nec[2]))
    assert decisao["pesos_escolhido"] in ("Balanced", "Sem_Pesos"), "Decisao de pesos invalida"
    assert int(decisao["k_escolhido"]) in cfg["k_lista"], "k escolhido fora da lista da Etapa 2"
    assert decisao.get("teste_cego_tocado") is False, "A decisao da Etapa 2 diz que o teste ja foi tocado!"
    return decisao, cfg, sp_["idx_treino"], sp_["idx_val"], sp_["idx_teste"]


def preparar_pastas(run_dir):
    run_dir = os.path.abspath(run_dir)
    e3 = os.path.join(run_dir, "etapa3_teste_cego")
    P = {"run": run_dir, "etapa3": e3}
    for sub in ["predicoes", "meta", "figuras", "relatorios"]:
        P[sub] = os.path.join(e3, sub)
        os.makedirs(P[sub], exist_ok=True)
    return P


def criar_lock(P, decisao, bracos):
    cam = os.path.join(P["etapa3"], "TESTE_CEGO_LOCK.json")
    if os.path.exists(cam):
        if not RETOMANDO:
            raise FileExistsError(
                "O teste cego desta run JA FOI INICIADO (TESTE_CEGO_LOCK.json existe). "
                "Para continuar uma execucao interrompida use --retomar. "
                "Para refazer do zero, crie uma nova run (Etapa 2) e nao reutilize esta.")
        print("   [retomada] lock existente mantido.")
        return
    with open(cam, "x", encoding="utf-8") as f:   # modo 'x': falha se existir
        json.dump({"iniciado_em": datetime.datetime.now().isoformat(timespec="seconds"),
                   "pesos": decisao["pesos_escolhido"], "k": decisao["k_escolhido"],
                   "bracos": [b["nome"] for b in bracos]}, f, indent=2)


def definir_bracos(decisao):
    k = int(decisao["k_escolhido"])
    bracos = [
        {"nome": "Baseline_HVG2000_PCA50", "tipo": "hvg_log", "dim": 50},
        {"nome": "SWeeP_600D_Puro", "tipo": "sweep_puro", "dim": 600},
        {"nome": f"SWeeP_600D_PCA{k}", "tipo": "sweep_pca", "dim": k},
    ]
    if RODAR_ABLACOES:
        bracos.append({"nome": "Ablacao_HVG2000_Binario_PCA50", "tipo": "hvg_bin", "dim": 50})
    return bracos


# ==============================================================================
# DADOS E AUDITORIA (mesmos asserts da Etapa 1)
# ==============================================================================
def carregar_e_auditar(idx_tr, idx_va, idx_te):
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

    assert X.shape[0] == len(y_tsv) == len(df_meta) == n_obs, "Tamanhos divergentes entre SWeeP, metadados e h5ad"
    assert np.array_equal(obs_index, df_meta["Barcode"].astype(str).values), "Barcodes do h5ad != metadados"
    assert np.array_equal(y_tsv, df_meta["Cell_Type_Num"].values.astype(int)), "Rotulos do TSV != metadados"
    assert X.shape[1] == 600 and n_vars == 61541
    assert not np.isnan(X).any()
    y = y_tsv - 1

    todos = np.concatenate([idx_tr, idx_va, idx_te])
    assert len(todos) == len(y) and len(np.unique(todos)) == len(y), "Split nao e uma particao das celulas"
    for nome, idx in [("treino", idx_tr), ("validacao", idx_va), ("teste", idx_te)]:
        assert set(np.unique(y[idx])) == set(range(8)), f"Classes ausentes no {nome}"
    print(f"   -> Auditoria OK. Split: treino={len(idx_tr)} | validacao={len(idx_va)} | teste cego={len(idx_te)}")
    return X, y


# ==============================================================================
# FEATURES POR BRACO (tudo ajustado SO no treino)
# ==============================================================================
CACHE_HVG = {}


def preparar_hvg(idx_tr, idx_va, idx_te, cfg, P):
    if "H" in CACHE_HVG:
        return CACHE_HVG["H"]
    print("   -> Carregando h5ad log-norm e selecionando 2.000 HVGs SOMENTE no treino...")
    t0 = time.time()
    adata = sc.read_h5ad(CAM_H5AD_LOG)
    ad_tr = adata[idx_tr, :].copy()
    sc.pp.highly_variable_genes(ad_tr, n_top_genes=N_HVG, flavor="seurat")
    mask = ad_tr.var["highly_variable"].values
    del ad_tr
    assert int(mask.sum()) == N_HVG, f"HVGs selecionados: {int(mask.sum())}"
    cols = np.where(mask)[0]
    salvar_txt(list(adata.var_names[mask].astype(str)),
               os.path.join(P["etapa3"], "hvg2000_ensembl_ids_treino70.txt"))
    Xall = adata.X

    def fatia(idx):
        M = Xall[idx][:, cols]
        return (M.toarray() if sp.issparse(M) else np.asarray(M)).astype(np.float32)

    H = (fatia(idx_tr), fatia(idx_va), fatia(idx_te))
    del adata, Xall
    print(f"      concluido em {time.time() - t0:.0f}s | matrizes HVG: {H[0].shape}, {H[1].shape}, {H[2].shape}")
    CACHE_HVG["H"] = H
    return H


def construir_features(braco, X_sweep, idx_tr, idx_va, idx_te, cfg, P):
    tipo = braco["tipo"]
    seed_split = cfg["seed_split"]
    if tipo == "sweep_puro":
        sc_ = StandardScaler()
        return (sc_.fit_transform(X_sweep[idx_tr]), sc_.transform(X_sweep[idx_va]), sc_.transform(X_sweep[idx_te]))
    if tipo == "sweep_pca":
        # Mesmo procedimento da Etapa 2: PCA de 50 ajustado no treino; usa as k primeiras componentes.
        pca = PCA(n_components=cfg["k_max_pca"], random_state=seed_split)
        Ztr = pca.fit_transform(X_sweep[idx_tr])
        Zva, Zte = pca.transform(X_sweep[idx_va]), pca.transform(X_sweep[idx_te])
        k = braco["dim"]
        sc_ = StandardScaler()
        return sc_.fit_transform(Ztr[:, :k]), sc_.transform(Zva[:, :k]), sc_.transform(Zte[:, :k])
    if tipo in ("hvg_log", "hvg_bin"):
        Htr, Hva, Hte = preparar_hvg(idx_tr, idx_va, idx_te, cfg, P)
        if tipo == "hvg_bin":
            Htr, Hva, Hte = [(h > 0).astype(np.float32) for h in (Htr, Hva, Hte)]
        pca = PCA(n_components=50, random_state=seed_split)
        Ztr = pca.fit_transform(Htr)
        Zva, Zte = pca.transform(Hva), pca.transform(Hte)
        sc_ = StandardScaler()
        return sc_.fit_transform(Ztr), sc_.transform(Zva), sc_.transform(Zte)
    raise ValueError(f"Tipo de braco desconhecido: {tipo}")


# ==============================================================================
# MLP E TREINO (nao recebe y_teste)
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
    loss_va_fn = nn.CrossEntropyLoss()   # sem pesos: mesma escala em qualquer esquema

    opt = torch.optim.Adam(modelo.parameters(), lr=cfg["lr"])
    ld_tr = DataLoader(TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long)),
                       batch_size=cfg["batch_treino"], shuffle=True)
    ld_va = DataLoader(TensorDataset(torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva, dtype=torch.long)),
                       batch_size=cfg["batch_eval"], shuffle=False)
    ld_te = DataLoader(TensorDataset(torch.tensor(Xte, dtype=torch.float32)),
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
    modelo.eval()
    with torch.no_grad():
        p_va = np.concatenate([modelo(bx.to(DEVICE)).argmax(1).cpu().numpy() for bx, _ in ld_va])
        p_te = np.concatenate([modelo(bx.to(DEVICE)).argmax(1).cpu().numpy() for (bx,) in ld_te])

    return {
        "y_pred_teste": p_te.astype(np.int64),
        "F1_Macro_Val": float(f1_score(yva, p_va, average="macro", labels=list(range(nc)), zero_division=0)),
        "Loss_Val_CE": float(melhor),
        "Epoca_Ideal": int(ep_ideal),
        "Epocas_Rodadas": int(len(hist_va)),
        "hist_treino": hist_tr,
        "hist_val": hist_va,
        "Tempo_s": float(time.time() - t0),
    }


# ==============================================================================
# METRICAS, TABELAS E BOOTSTRAP
# ==============================================================================
def metricas(y, yp):
    labs = list(range(8))
    p, r, f, sup = precision_recall_fscore_support(y, yp, labels=labs, zero_division=0)
    return {
        "Acuracia": accuracy_score(y, yp),
        "Acuracia_Balanceada": balanced_accuracy_score(y, yp),
        "F1_Macro": f1_score(y, yp, average="macro", labels=labs, zero_division=0),
        "F1_Weighted": f1_score(y, yp, average="weighted", labels=labs, zero_division=0),
        "prec": p, "rec": r, "f1": f, "sup": sup,
    }


def f1_macro_rapido(y, p, idx):
    cm = np.bincount(y[idx] * 8 + p[idx], minlength=64).reshape(8, 8)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    den = 2 * tp + fp + fn
    return float(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0).mean())


def bootstrap_pareado(y, preds, nomes, seeds, n_boot, seed_boot):
    """Reamostra CELULAS do teste (mesmas reamostras para todos os bracos). Nao considera
    agrupamento por doador: o teste e split por celula; a generalizacao entre doadores e o LODO."""
    rng = np.random.default_rng(seed_boot)
    n = len(y)
    todos = np.arange(n)
    pares = [(a, b) for i, a in enumerate(nomes) for b in nomes[i + 1:]]
    linhas = []
    for s in seeds:
        f1b = {nm: np.empty(n_boot) for nm in nomes}
        for bi in range(n_boot):
            idx = rng.integers(0, n, n)
            for nm in nomes:
                f1b[nm][bi] = f1_macro_rapido(y, preds[(nm, s)], idx)
        for a, b in pares:
            obs = f1_macro_rapido(y, preds[(a, s)], todos) - f1_macro_rapido(y, preds[(b, s)], todos)
            d = f1b[a] - f1b[b]
            linhas.append({"Seed": s, "Braco_A": a, "Braco_B": b, "Delta_F1Macro_A_menos_B": obs,
                           "IC95_inf": float(np.percentile(d, 2.5)), "IC95_sup": float(np.percentile(d, 97.5)),
                           "Frac_boot_delta_positivo": float((d > 0).mean()), "N_boot": n_boot})
    return pd.DataFrame(linhas)


def figuras_seed_principal(P, nome, y, yp, meta):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, len(meta["hist_treino"]) + 1), meta["hist_treino"], label="Loss de treino", lw=2)
    ax.plot(range(1, len(meta["hist_val"]) + 1), meta["hist_val"], label="Loss de validacao (CE sem pesos)", lw=2)
    ax.axvline(meta["Epoca_Ideal"], color="red", ls="--", label=f"Checkpoint (epoca {meta['Epoca_Ideal']})")
    ax.set_xlabel("Epocas")
    ax.set_ylabel("Cross-entropy")
    ax.set_title(f"Convergencia - {nome}")
    ax.legend()
    ax.grid(True, ls=":", alpha=0.6)
    salvar_fig(fig, os.path.join(P["figuras"], f"curva_loss_{nome}.png"))

    labs = list(range(8))
    cmn = confusion_matrix(y, yp, labels=labs, normalize="true") * 100
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    sns.heatmap(cmn, annot=True, fmt=".1f", cmap="Blues", xticklabels=NOMES_CLASSES, yticklabels=NOMES_CLASSES, ax=ax)
    ax.set_xlabel("Predicao da MLP")
    ax.set_ylabel("Classe real (Pan et al.)")
    ax.set_title(f"Matriz de confusao normalizada por linha (%) - {nome}")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    salvar_fig(fig, os.path.join(P["figuras"], f"matriz_confusao_norm_{nome}.png"))

    cma = confusion_matrix(y, yp, labels=labs)
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    sns.heatmap(cma, annot=True, fmt="d", cmap="Blues", xticklabels=NOMES_CLASSES, yticklabels=NOMES_CLASSES, ax=ax)
    ax.set_xlabel("Predicao da MLP")
    ax.set_ylabel("Classe real (Pan et al.)")
    ax.set_title(f"Matriz de confusao absoluta - {nome}")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    salvar_fig(fig, os.path.join(P["figuras"], f"matriz_confusao_abs_{nome}.png"))


def montar_tabelas(P, bracos, seeds, y_te, seed_principal):
    nomes = [b["nome"] for b in bracos]
    preds, linhas, longa = {}, [], []
    for b in bracos:
        for s in seeds:
            tag = f"{b['nome']}_seed{s}"
            yp = np.load(os.path.join(P["predicoes"], f"ypred_{tag}.npy"))
            assert len(yp) == len(y_te), f"Predicao com tamanho errado: {tag}"
            with open(os.path.join(P["meta"], f"meta_{tag}.json"), encoding="utf-8") as f:
                meta = json.load(f)
            preds[(b["nome"], s)] = yp
            m = metricas(y_te, yp)
            linha = {"Representacao": b["nome"], "Dimensao": b["dim"], "Pesos": meta["Pesos"], "Seed": s,
                     "Epoca_Ideal": meta["Epoca_Ideal"], "Epocas_Rodadas": meta["Epocas_Rodadas"],
                     "F1_Macro_Val": meta["F1_Macro_Val"], "Tempo_Treino_s": meta["Tempo_s"]}
            for kk in ["Acuracia", "Acuracia_Balanceada", "F1_Macro", "F1_Weighted"]:
                linha[kk] = m[kk]
            for ci, nc in enumerate(NOMES_CLASSES):
                linha[f"F1_{nc.split(':')[1].strip()}"] = m["f1"][ci]
                longa.append({"Representacao": b["nome"], "Seed": s, "Classe": nc,
                              "Precisao": m["prec"][ci], "Recall": m["rec"][ci], "F1": m["f1"][ci],
                              "Suporte": int(m["sup"][ci])})
            linhas.append(linha)
            if s == seed_principal:
                salvar_txt(classification_report(y_te, yp, labels=list(range(8)), target_names=NOMES_CLASSES, digits=4,
                                                 zero_division=0),
                           os.path.join(P["relatorios"], f"relatorio_{b['nome']}_seed{s}.txt"))
                figuras_seed_principal(P, b["nome"], y_te, yp, meta)

    df = pd.DataFrame(linhas)
    salvar_df(df, os.path.join(P["etapa3"], "Tabela2_Teste_Cego_POR_SEED.csv"), index=False)

    cols = ["Acuracia", "Acuracia_Balanceada", "F1_Macro", "F1_Weighted", "Epoca_Ideal", "F1_Macro_Val"]
    res = df.groupby("Representacao", sort=False)[cols].agg(["mean", "std"])
    res.columns = [f"{a}_{b}" for a, b in res.columns]
    salvar_df(res.reset_index(), os.path.join(P["etapa3"], "Tabela2a_Resumo_Media_DP.csv"), index=False)

    dl = pd.DataFrame(longa)
    salvar_df(dl, os.path.join(P["etapa3"], "Tabela2b_Metricas_Por_Classe_LONG.csv"), index=False)
    piv = dl.pivot_table(index="Classe", columns="Representacao", values="F1", aggfunc="mean")[nomes]
    sup = dl[dl["Representacao"] == nomes[0]].groupby("Classe")["Suporte"].first()
    piv.insert(0, "Suporte_teste", sup)
    salvar_df(piv.reset_index(), os.path.join(P["etapa3"], "Tabela2c_F1_Por_Classe_Media_Seeds.csv"), index=False)

    boot = bootstrap_pareado(y_te, preds, nomes, seeds, N_BOOT, SEED_BOOT)
    salvar_df(boot, os.path.join(P["etapa3"], "Tabela2d_Bootstrap_Pareado_Celulas.csv"), index=False)
    return df, res, piv, boot


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    global RETOMANDO
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN, help="Pasta run_* da Etapa 2")
    ap.add_argument("--retomar", action="store_true", help="Continua uma execucao interrompida da Etapa 3")
    args = ap.parse_args()
    RETOMANDO = args.retomar

    decisao, cfg, idx_tr, idx_va, idx_te = carregar_run(args.run)
    P = preparar_pastas(args.run)
    bracos = definir_bracos(decisao)
    usar_pesos = decisao["pesos_escolhido"] == "Balanced"
    seeds = cfg["seeds_treino"]
    seed_principal = seeds[0]

    print("=" * 78)
    print("ETAPA 3 - TESTE CEGO (cada braco x seed e avaliado UMA vez; configuracao congelada)")
    print("=" * 78)
    print(f"[INFO] Dispositivo: {DEVICE}")
    print(f"[INFO] Run da Etapa 2: {P['run']}")
    print(f"[INFO] Pesos={decisao['pesos_escolhido']} | k={decisao['k_escolhido']} | seeds={seeds}")
    print(f"[INFO] Bracos: {[b['nome'] for b in bracos]} (RODAR_ABLACOES={RODAR_ABLACOES})")

    criar_lock(P, decisao, bracos)
    salvar_json({
        "decisao_etapa2": decisao, "cfg_etapa2": cfg, "bracos": bracos, "RODAR_ABLACOES": RODAR_ABLACOES,
        "N_BOOT": N_BOOT, "SEED_BOOT": SEED_BOOT,
        "ambiente": {"python": sys.version, "plataforma": platform.platform(), "torch": torch.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "scanpy": versao("scanpy"),
                     "scikit-learn": versao("scikit-learn"), "device": str(DEVICE)},
    }, os.path.join(P["etapa3"], "config_etapa3.json"))

    print("\n[1/3] Carregando dados e auditando alinhamento...")
    X, y = carregar_e_auditar(idx_tr, idx_va, idx_te)
    ytr, yva, yte = y[idx_tr], y[idx_va], y[idx_te]
    salvar_npy(idx_te, os.path.join(P["predicoes"], "idx_teste.npy"))
    salvar_npy(yte, os.path.join(P["predicoes"], "y_teste.npy"))

    print("\n[2/3] Treinando e avaliando os bracos...")
    for b in bracos:
        pend = [s for s in seeds
                if not (os.path.exists(os.path.join(P["predicoes"], f"ypred_{b['nome']}_seed{s}.npy"))
                        and os.path.exists(os.path.join(P["meta"], f"meta_{b['nome']}_seed{s}.json")))]
        print(f"\n--- {b['nome']} ({b['dim']}D) | pendentes: {pend if pend else 'nenhuma'} ---")
        if not pend:
            continue
        Xtr, Xva, Xte = construir_features(b, X, idx_tr, idx_va, idx_te, cfg, P)
        for s in pend:
            r = treinar_e_predizer(Xtr, ytr, Xva, yva, Xte, usar_pesos, s, cfg)
            yp = r.pop("y_pred_teste")
            tag = f"{b['nome']}_seed{s}"
            salvar_npy(yp, os.path.join(P["predicoes"], f"ypred_{tag}.npy"))
            r["Pesos"] = decisao["pesos_escolhido"]
            r["Braco"] = b["nome"]
            r["Seed"] = s
            salvar_json(r, os.path.join(P["meta"], f"meta_{tag}.json"))
            f1_te = f1_score(yte, yp, average="macro", labels=list(range(8)), zero_division=0)
            print(f"   [seed {s:>4}] epoca={r['Epoca_Ideal']:02d}/{r['Epocas_Rodadas']:02d} | "
                  f"F1-macro val={r['F1_Macro_Val']:.4f} | F1-macro TESTE={f1_te:.4f} | {r['Tempo_s']:.0f}s")
        del Xtr, Xva, Xte
    CACHE_HVG.clear()

    print("\n[3/3] Consolidando tabelas, figuras e bootstrap pareado...")
    df, res, piv, boot = montar_tabelas(P, bracos, seeds, yte, seed_principal)

    pd.set_option("display.width", 220)
    print("\n[RESUMO TESTE CEGO - MEDIA e DP entre seeds]")
    print(res.round(4).to_string())
    print("\n[F1 POR CLASSE - media das seeds]")
    print(piv.round(4).to_string())
    print(f"\n[BOOTSTRAP PAREADO (celulas, {N_BOOT} reamostras) - seed {seed_principal}]")
    print(boot[boot["Seed"] == seed_principal].round(4).to_string(index=False))

    salvar_json({"concluido_em": datetime.datetime.now().isoformat(timespec="seconds"),
                 "bracos": [b["nome"] for b in bracos], "seeds": seeds,
                 "obs": "DP entre seeds nao inclui variancia do split; bootstrap e por celula, nao por doador."},
                os.path.join(P["etapa3"], "etapa3_concluida.json"))
    print("\n" + "=" * 78)
    print(f"ETAPA 3 CONCLUIDA. Saidas em: {P['etapa3']}")
    print("=" * 78)


if __name__ == "__main__":
    main()