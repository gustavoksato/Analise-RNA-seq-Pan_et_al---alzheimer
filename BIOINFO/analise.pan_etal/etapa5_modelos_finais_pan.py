#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETAPA 5 - TREINAMENTO E CONGELAMENTO DOS 3 MODELOS FINAIS NO GSE267301 (Pan et al.)

Objetivo:
    Ajustar e congelar os 3 modelos finais de producao (Baseline_HVG2000_PCA50,
    SWeeP_600D_Puro e SWeeP_600D_PCA<k>) utilizando todos os 8 doadores do GSE267301
    (58.326 celulas), com particao 90% treino final (52.493 celulas) e 10% hold-out
    estratificado (5.833 celulas) usado EXCLUSIVAMENTE para checkpoint de early stopping.

Importante (governanca metodologica):
    - As metricas de desempenho interno do trabalho sao as da Etapa 3 (Teste Cego) e
      Etapa 4 (LODO). Os modelos gerados nesta Etapa 5 NUNCA sao avaliados internamente
      como metrica de resultado: existem estritamente como artefatos congelados para
      inferencia cega nas coortes externas (GSE263468 / Dharsini e GSE305140 / Lattke).
    - Todos os transformadores (2.000 HVGs seurat, PCA e StandardScaler) sao ajustados
      estritamente nos 90% de treino final (idx_treino_final), sem vazamento do hold-out.
    - Salva artefatos completos para os 3 bracos (incluindo mascara, indices, IDs Ensembl,
      PCA, StandardScaler e pesos da MLP do Baseline HVG), alem de verificar a integridade
      de serializacao (round-trip check) de cada arquivo salvo.

Politica de escrita:
    Nenhum arquivo existente e sobrescrito. Usa MODELOS_FINAIS_LOCK.json e suporta --retomar.

Uso:
    python etapa5_modelos_finais_pan.py
    python etapa5_modelos_finais_pan.py --run /caminho/run_AAAAMMDD_HHMMSS
    python etapa5_modelos_finais_pan.py --retomar
"""

import os
import sys
import json
import time
import copy
import hashlib
import platform
import argparse
import datetime
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

DEFAULT_RUN = os.path.join(BASE, "resultados_MLP_GSE267301_v2", "run_20261102_201556")

FRAC_HOLDOUT_ES = 0.10   # 10% estratificado (por classe x doador) exclusivo para early stopping
N_HVG = 2000

NOMES_CLASSES = [
    "1: Astrocytes", "2: Oligodendrocytes", "3: Exc. Neurons", "4: Inh. Neurons",
    "5: OPCs", "6: Microglia", "7: Endothelial", "8: VLMCs"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------------------
# Utilitarios de escrita SEM overwrite
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


def sha256_lista(strings):
    h = hashlib.sha256()
    for s in strings:
        h.update((str(s) + "\n").encode("utf-8"))
    return h.hexdigest()


# ==============================================================================
# LEITURA DA RUN E PREPARACAO DE PASTAS
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
    e5 = os.path.join(run_dir, "etapa5_modelos_finais")
    P = {
        "run": run_dir,
        "etapa1": os.path.join(run_dir, "etapa1_auditoria"),
        "etapa3": os.path.join(run_dir, "etapa3_teste_cego"),
        "etapa4": os.path.join(run_dir, "etapa4_lodo"),
        "etapa5": e5,
    }
    for sub in ["artefatos_congelados", "seeds", "meta", "figuras"]:
        P[sub] = os.path.join(e5, sub)
        os.makedirs(P[sub], exist_ok=True)
    return P


def criar_lock(P, decisao, bracos):
    cam = os.path.join(P["etapa5"], "MODELOS_FINAIS_LOCK.json")
    if os.path.exists(cam):
        if not RETOMANDO:
            raise FileExistsError(
                "A Etapa 5 desta run JA FOI INICIADA (MODELOS_FINAIS_LOCK.json existe). "
                "Use --retomar para continuar uma execucao interrompida."
            )
        print("   [retomada] lock existente mantido.")
        return
    with open(cam, "x", encoding="utf-8") as f:
        json.dump({
            "iniciado_em": datetime.datetime.now().isoformat(timespec="seconds"),
            "pesos": decisao["pesos_escolhido"],
            "k": decisao["k_escolhido"],
            "bracos": [b["nome"] for b in bracos],
        }, f, indent=2)


def definir_bracos(decisao):
    k = int(decisao["k_escolhido"])
    return [
        {"nome": "Baseline_HVG2000_PCA50", "tipo": "hvg_log", "dim": 50},
        {"nome": "SWeeP_600D_Puro", "tipo": "sweep_puro", "dim": 600},
        {"nome": f"SWeeP_600D_PCA{k}", "tipo": "sweep_pca", "dim": k},
    ]


# ==============================================================================
# AUDITORIA DOS DADOS E SPLIT 90/10 DE EARLY STOPPING
# ==============================================================================
def carregar_e_auditar(P):
    for nome, cam in [("SWeeP TSV", CAM_SWEEP_600), ("Metadados", CAM_METADADOS), ("h5ad LogNorm", CAM_H5AD_LOG)]:
        if not os.path.exists(cam):
            raise FileNotFoundError(f"{nome} nao encontrado: {cam}")

    df_sw = pd.read_csv(CAM_SWEEP_600, sep="\t")
    y_tsv = df_sw["Cell_Type"].values.astype(int)
    X_sweep = df_sw.drop(columns=["Cell_Type"]).values.astype(np.float32)
    del df_sw
    df_meta = pd.read_csv(CAM_METADADOS)

    ad = sc.read_h5ad(CAM_H5AD_LOG, backed="r")
    obs_index = ad.obs.index.values.astype(str)
    var_names = ad.var_names.values.astype(str)
    n_obs, n_vars = ad.n_obs, ad.n_vars
    ad.file.close()
    del ad

    assert X_sweep.shape[0] == len(y_tsv) == len(df_meta) == n_obs == 58326, "Numero de celulas != 58326"
    assert X_sweep.shape[1] == 600 and n_vars == 61541, "Dimensoes invalidas"
    assert np.array_equal(obs_index, df_meta["Barcode"].astype(str).values), "Barcodes do h5ad != metadados"
    assert np.array_equal(y_tsv, df_meta["Cell_Type_Num"].values.astype(int)), "Rotulos do TSV != metadados"
    assert not np.isnan(X_sweep).any(), "NaN detectado na matriz SWeeP"

    genes_puros = pd.Series(var_names).str.replace(r"\..*$", "", regex=True).str.strip().str.upper().values
    cam_g1 = os.path.join(P["etapa1"], "genes_pan_61541.txt")
    if os.path.exists(cam_g1):
        with open(cam_g1, encoding="utf-8") as f:
            g_etapa1 = np.array([l.strip().upper() for l in f if l.strip()])
        assert np.array_equal(genes_puros, g_etapa1), "Ordem dos genes diverge da Etapa 1!"

    donors = df_meta["Donor"].astype(str).values
    y = y_tsv - 1
    print(f"   -> Auditoria OK: {len(y)} celulas, {len(np.unique(donors))} doadores, {len(genes_puros)} genes.")
    return X_sweep, y, donors, var_names, genes_puros


def criar_split_90_10(y, donors, seed_split, P):
    cam = os.path.join(P["etapa5"], "split_final_90_10_early_stopping.npz")
    if os.path.exists(cam) and RETOMANDO:
        sp_ = np.load(cam)
        idx_tr, idx_va = sp_["idx_treino_final"], sp_["idx_val_es"]
        print(f"   [retomada] Split 90/10 carregado: treino={len(idx_tr)} | val_es={len(idx_va)}")
        return idx_tr, idx_va

    # Estratifica simultaneamente por (Doador, Classe) para que todos os 8 doadores e 8 classes
    # estejam perfeitamente representados tanto nos 90% de treino quanto nos 10% de checkpoint
    estrato = np.array([f"{d}_{c}" for d, c in zip(donors, y)])
    indices = np.arange(len(y))
    idx_tr, idx_va = train_test_split(
        indices, test_size=FRAC_HOLDOUT_ES, random_state=seed_split, stratify=estrato
    )
    assert len(np.intersect1d(idx_tr, idx_va)) == 0
    assert set(np.unique(y[idx_tr])) == set(range(8)) and set(np.unique(y[idx_va])) == set(range(8))
    assert set(np.unique(donors[idx_tr])) == set(np.unique(donors))
    assert set(np.unique(donors[idx_va])) == set(np.unique(donors))

    if _checar_novo(cam):
        np.savez(cam, idx_treino_final=idx_tr, idx_val_es=idx_va)
    print(f"   -> Split 90/10 (estratificado por Doador x Classe): "
          f"treino final={len(idx_tr)} (90%) | hold-out early stopping={len(idx_va)} (10%)")
    return idx_tr, idx_va


# ==============================================================================
# ARQUITETURA DA MLP E TREINAMENTO COM CHECKPOINT
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


def treinar_modelo_final(Xtr, ytr, Xva, yva, usar_pesos, seed, cfg):
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
        pesos = np.ones(nc, dtype=np.float32)
        loss_tr_fn = nn.CrossEntropyLoss()
    loss_va_fn = nn.CrossEntropyLoss()  # sem pesos para parada consistente

    opt = torch.optim.Adam(modelo.parameters(), lr=cfg["lr"])
    ld_tr = DataLoader(
        TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long)),
        batch_size=cfg["batch_treino"], shuffle=True
    )
    ld_va = DataLoader(
        TensorDataset(torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva, dtype=torch.long)),
        batch_size=cfg["batch_eval"], shuffle=False
    )

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

    return {
        "state_dict": estado,
        "y_pred_val_es": p_va.astype(np.int64),
        "pesos_classe_treino": pesos.tolist(),
        "F1_Macro_Val_ES": float(f1_score(yva, p_va, average="macro", labels=list(range(nc)), zero_division=0)),
        "Acc_Bal_Val_ES": float(balanced_accuracy_score(yva, p_va)),
        "Acc_Val_ES": float(accuracy_score(yva, p_va)),
        "Loss_Val_CE": float(melhor),
        "Epoca_Ideal": int(ep_ideal),
        "Epocas_Rodadas": int(len(hist_va)),
        "hist_treino": hist_tr,
        "hist_val": hist_va,
        "Tempo_s": float(time.time() - t0),
    }


# ==============================================================================
# CONSTRUCAO E CONGELAMENTO DOS TRANSFORMADORES POR BRACO
# ==============================================================================
def ajustar_e_salvar_transformadores(braco, X_sweep, idx_tr, idx_va, var_names, genes_puros, cfg, P):
    tipo = braco["tipo"]
    seed_split = cfg["seed_split"]
    art = P["artefatos_congelados"]

    if tipo == "sweep_puro":
        t0 = time.time()
        sc600 = StandardScaler()
        Xtr = sc600.fit_transform(X_sweep[idx_tr])
        Xva = sc600.transform(X_sweep[idx_va])
        cam_sc = os.path.join(art, "scaler_sweep600d.pkl")
        salvar_pkl(sc600, cam_sc)
        # Round-trip check do scaler
        sc_reload = joblib.load(cam_sc)
        assert np.allclose(Xva, sc_reload.transform(X_sweep[idx_va]), atol=1e-6), "Falha no round-trip scaler_sweep600d"
        return Xtr, Xva, {"scaler_pkl": "scaler_sweep600d.pkl", "Tempo_Prep_s": float(time.time() - t0)}

    if tipo == "sweep_pca":
        t0 = time.time()
        k = braco["dim"]
        pca = PCA(n_components=cfg["k_max_pca"], random_state=seed_split)
        Ztr = pca.fit_transform(X_sweep[idx_tr])
        Zva = pca.transform(X_sweep[idx_va])
        sc_k = StandardScaler()
        Xtr = sc_k.fit_transform(Ztr[:, :k])
        Xva = sc_k.transform(Zva[:, :k])

        cam_pca = os.path.join(art, "pca_sweep50d_full.pkl")
        cam_sc = os.path.join(art, f"scaler_sweep_pca{k}d.pkl")
        salvar_pkl(pca, cam_pca)
        salvar_pkl(sc_k, cam_sc)

        # Round-trip check do PCA + recorte k + scaler
        pca_rel = joblib.load(cam_pca)
        sc_rel = joblib.load(cam_sc)
        Xva_rel = sc_rel.transform(pca_rel.transform(X_sweep[idx_va])[:, :k])
        assert np.allclose(Xva, Xva_rel, atol=1e-6), "Falha no round-trip PCA+Scaler SWeeP"
        return Xtr, Xva, {
            "pca_pkl": "pca_sweep50d_full.pkl",
            "k_recorte": int(k),
            "var_explicada_k_%": float(np.sum(pca.explained_variance_ratio_[:k]) * 100),
            "scaler_pkl": f"scaler_sweep_pca{k}d.pkl",
            "Tempo_Prep_s": float(time.time() - t0),
        }

    if tipo == "hvg_log":
        t0 = time.time()
        print("   -> Carregando h5ad log-norm e ajustando 2.000 HVGs nos 90% de treino final...")
        adata = sc.read_h5ad(CAM_H5AD_LOG)
        ad_tr = adata[idx_tr, :].copy()
        sc.pp.highly_variable_genes(ad_tr, n_top_genes=N_HVG, flavor="seurat")
        mask = ad_tr.var["highly_variable"].values.astype(bool)
        del ad_tr
        assert int(mask.sum()) == N_HVG, f"Esperado {N_HVG} HVGs, obteve {int(mask.sum())}"

        cols = np.where(mask)[0].astype(np.int64)
        ids_orig = var_names[cols]
        ids_puros = genes_puros[cols]

        salvar_npy(mask, os.path.join(art, "hvg2000_mask_61541.npy"))
        salvar_npy(cols, os.path.join(art, "hvg2000_indices_colunas.npy"))
        salvar_txt(ids_orig, os.path.join(art, "hvg2000_var_names_originais.txt"))
        salvar_txt(ids_puros, os.path.join(art, "hvg2000_ensembl_ids_puros.txt"))

        Xall = adata.X

        def extrair_denso(idx):
            M = Xall[idx][:, cols]
            return (M.toarray() if sp.issparse(M) else np.asarray(M)).astype(np.float32)

        Htr = extrair_denso(idx_tr)
        Hva = extrair_denso(idx_va)
        del adata, Xall

        pca_hvg = PCA(n_components=50, random_state=seed_split)
        Ztr = pca_hvg.fit_transform(Htr)
        Zva = pca_hvg.transform(Hva)
        del Htr

        sc_hvg = StandardScaler()
        Xtr = sc_hvg.fit_transform(Ztr)
        Xva = sc_hvg.transform(Zva)

        cam_pca = os.path.join(art, "pca_hvg50.pkl")
        cam_sc = os.path.join(art, "scaler_hvg50.pkl")
        salvar_pkl(pca_hvg, cam_pca)
        salvar_pkl(sc_hvg, cam_sc)

        # Round-trip check do HVG PCA + Scaler
        pca_rel = joblib.load(cam_pca)
        sc_rel = joblib.load(cam_sc)
        Xva_rel = sc_rel.transform(pca_rel.transform(Hva))
        assert np.allclose(Xva, Xva_rel, atol=1e-5), "Falha no round-trip PCA+Scaler do Baseline HVG"
        del Hva
        return Xtr, Xva, {
            "hvg_mask_npy": "hvg2000_mask_61541.npy",
            "hvg_indices_npy": "hvg2000_indices_colunas.npy",
            "hvg_ensembl_txt": "hvg2000_ensembl_ids_puros.txt",
            "pca_pkl": "pca_hvg50.pkl",
            "var_explicada_50_%": float(np.sum(pca_hvg.explained_variance_ratio_) * 100),
            "scaler_pkl": "scaler_hvg50.pkl",
            "regra_genes_ausentes": "zero_fill_apos_log1p_1e4",
            "Tempo_Prep_s": float(time.time() - t0),
        }

    raise ValueError(f"Tipo desconhecido: {tipo}")


def verificar_roundtrip_mlp(cam_pt, input_dim, Xva, y_pred_esperado):
    m = ClassificadorCelular(input_dim, 8).to(DEVICE)
    estado_disco = torch.load(cam_pt, map_location=DEVICE, weights_only=True)
    m.load_state_dict(estado_disco)
    m.eval()
    ld = DataLoader(TensorDataset(torch.tensor(Xva, dtype=torch.float32)), batch_size=512, shuffle=False)
    with torch.no_grad():
        p_disco = np.concatenate([m(bx.to(DEVICE)).argmax(1).cpu().numpy() for (bx,) in ld])
    assert np.array_equal(p_disco, y_pred_esperado), f"ERRO CRITICO: predicao apos recarregar {cam_pt} difere!"


def plotar_curva_final(P, nome, meta):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, len(meta["hist_treino"]) + 1), meta["hist_treino"], label="Loss treino final (90%)", lw=2)
    ax.plot(range(1, len(meta["hist_val"]) + 1), meta["hist_val"], label="Loss hold-out ES (10%, CE s/ pesos)", lw=2)
    ax.axvline(meta["Epoca_Ideal"], color="red", ls="--", label=f"Checkpoint congelado (epoca {meta['Epoca_Ideal']})")
    ax.set_xlabel("Epocas")
    ax.set_ylabel("Cross-entropy")
    ax.set_title(f"Modelo Final Congelado (90/10) - {nome}")
    ax.legend()
    ax.grid(True, ls=":", alpha=0.6)
    salvar_fig(fig, os.path.join(P["figuras"], f"curva_loss_final_{nome}.png"))


def comparar_estabilidade_hvg_final(P):
    cam_final = os.path.join(P["artefatos_congelados"], "hvg2000_var_names_originais.txt")
    if not os.path.exists(cam_final):
        return None
    with open(cam_final, encoding="utf-8") as f:
        set_final = set(l.strip() for l in f if l.strip())

    linhas = []
    cam_e3 = os.path.join(P["etapa3"], "hvg2000_ensembl_ids_treino70.txt")
    if os.path.exists(cam_e3):
        with open(cam_e3, encoding="utf-8") as f:
            s3 = set(l.strip() for l in f if l.strip())
        inter = len(set_final & s3)
        uniao = len(set_final | s3)
        linhas.append({"Comparacao": "Final_90pct_vs_Etapa3_70pct", "Intersecao_Genes": inter,
                       "Jaccard": inter / max(uniao, 1)})

    pasta_hvg_e4 = os.path.join(P["etapa4"], "hvg_ids")
    if os.path.isdir(pasta_hvg_e4):
        for arq in sorted(os.listdir(pasta_hvg_e4)):
            if arq.startswith("hvg2000_fold") and arq.endswith(".txt"):
                with open(os.path.join(pasta_hvg_e4, arq), encoding="utf-8") as f:
                    sf = set(l.strip() for l in f if l.strip())
                inter = len(set_final & sf)
                uniao = len(set_final | sf)
                linhas.append({"Comparacao": f"Final_90pct_vs_{arq[:-4]}", "Intersecao_Genes": inter,
                               "Jaccard": inter / max(uniao, 1)})
    if not linhas:
        return None
    df_j = pd.DataFrame(linhas)
    salvar_df(df_j, os.path.join(P["etapa5"], "Tabela5b_Comparacao_HVG_Final_vs_Etapa3_e_LODO.csv"), index=False)
    return df_j


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    global RETOMANDO
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN, help="Pasta run_* da Etapa 2")
    ap.add_argument("--retomar", action="store_true", help="Continua uma execucao interrompida da Etapa 5")
    args = ap.parse_args()
    RETOMANDO = args.retomar

    decisao, cfg = carregar_run(args.run)
    P = preparar_pastas(args.run)
    bracos = definir_bracos(decisao)
    usar_pesos = decisao["pesos_escolhido"] == "Balanced"
    seeds = cfg["seeds_treino"]
    seed_principal = seeds[0]

    print("=" * 78)
    print("ETAPA 5 - TREINAMENTO E CONGELAMENTO DOS 3 MODELOS FINAIS (GSE267301 - 100%)")
    print("=" * 78)
    print(f"[INFO] Dispositivo: {DEVICE}")
    print(f"[INFO] Run: {P['run']}")
    print(f"[INFO] Pesos={decisao['pesos_escolhido']} | k={decisao['k_escolhido']} | seeds={seeds}")
    print(f"[INFO] Bracos a congelar: {[b['nome'] for b in bracos]}")

    criar_lock(P, decisao, bracos)

    print("\n[1/4] Carregando dados, auditando alinhamento e gerando split 90/10 de checkpoint...")
    X_sweep, y, donors, var_names, genes_puros = carregar_e_auditar(P)
    idx_tr, idx_va = criar_split_90_10(y, donors, cfg["seed_split"], P)
    ytr, yva = y[idx_tr], y[idx_va]

    # Salva referencia genomica imutavel dentro de artefatos_congelados
    salvar_txt(genes_puros, os.path.join(P["artefatos_congelados"], "genes_pan_61541.txt"))
    salvar_txt(var_names, os.path.join(P["artefatos_congelados"], "genes_pan_61541_var_names_originais.txt"))
    hash_genes_puros = sha256_lista(genes_puros)

    print("\n[2/4] Ajustando transformadores (nos 90%) e treinando modelos finais...")
    info_artefatos = {}
    linhas_tab = []

    for b in bracos:
        nome, dim = b["nome"], b["dim"]
        print(f"\n--- {nome} ({dim}D) ---")
        Xtr, Xva, info_prep = ajustar_e_salvar_transformadores(
            b, X_sweep, idx_tr, idx_va, var_names, genes_puros, cfg, P
        )
        info_artefatos[nome] = info_prep

        for s in seeds:
            cam_meta = os.path.join(P["meta"], f"meta_final_{nome}_seed{s}.json")
            cam_pt_seed = os.path.join(P["seeds"], f"mlp_{nome}_seed{s}.pt")

            if os.path.exists(cam_meta) and os.path.exists(cam_pt_seed) and RETOMANDO:
                with open(cam_meta, encoding="utf-8") as f:
                    m = json.load(f)
                print(f"   [retomada seed {s:>4}] epoca={m['Epoca_Ideal']:02d}/{m['Epocas_Rodadas']:02d} | "
                      f"F1-macro holdout ES={m['F1_Macro_Val_ES']:.4f}")
            else:
                r = treinar_modelo_final(Xtr, ytr, Xva, yva, usar_pesos, s, cfg)
                estado = r.pop("state_dict")
                yp_va = r.pop("y_pred_val_es")

                salvar_pt(estado, cam_pt_seed)
                verificar_roundtrip_mlp(cam_pt_seed, dim, Xva, yp_va)

                if s == seed_principal:
                    cam_pt_canon = os.path.join(P["artefatos_congelados"], f"mlp_{nome}.pt")
                    salvar_pt(estado, cam_pt_canon)
                    verificar_roundtrip_mlp(cam_pt_canon, dim, Xva, yp_va)
                    plotar_curva_final(P, nome, r)

                m = {
                    "Representacao": nome,
                    "Dimensao": dim,
                    "Pesos": decisao["pesos_escolhido"],
                    "Seed": s,
                    "N_Treino_Final": int(len(idx_tr)),
                    "N_Holdout_ES": int(len(idx_va)),
                    "Tempo_Prep_s": info_prep["Tempo_Prep_s"],
                    **r,
                }
                salvar_json(m, cam_meta)
                print(f"   [seed {s:>4}] epoca={m['Epoca_Ideal']:02d}/{m['Epocas_Rodadas']:02d} | "
                      f"Loss ES={m['Loss_Val_CE']:.4f} | F1-macro holdout ES={m['F1_Macro_Val_ES']:.4f} | "
                      f"Treino={m['Tempo_s']:.0f}s (Round-trip OK)")

            linhas_tab.append({
                "Representacao": m["Representacao"],
                "Dimensao": m["Dimensao"],
                "Pesos": m["Pesos"],
                "Seed": m["Seed"],
                "N_Treino_Final": m["N_Treino_Final"],
                "N_Holdout_ES": m["N_Holdout_ES"],
                "Epoca_Ideal": m["Epoca_Ideal"],
                "Epocas_Rodadas": m["Epocas_Rodadas"],
                "Loss_Holdout_CE": m["Loss_Val_CE"],
                "F1_Macro_Holdout_ES": m["F1_Macro_Val_ES"],
                "Acc_Bal_Holdout_ES": m["Acc_Bal_Val_ES"],
                "Acc_Holdout_ES": m["Acc_Val_ES"],
                "Tempo_Prep_s": m["Tempo_Prep_s"],
                "Tempo_Treino_s": m["Tempo_s"],
            })

        del Xtr, Xva

    print("\n[3/4] Consolidando tabelas de diagnostico e estabilidade dos HVGs finais...")
    df_tab = pd.DataFrame(linhas_tab)
    salvar_df(df_tab, os.path.join(P["etapa5"], "Tabela5_Treino_Final_POR_SEED.csv"), index=False)

    cols_res = ["Epoca_Ideal", "Loss_Holdout_CE", "F1_Macro_Holdout_ES", "Acc_Bal_Holdout_ES", "Tempo_Treino_s"]
    df_res = df_tab.groupby("Representacao", sort=False)[cols_res].agg(["mean", "std"])
    df_res.columns = [f"{a}_{b}" for a, b in df_res.columns]
    df_res = df_res.reset_index()
    salvar_df(df_res, os.path.join(P["etapa5"], "Tabela5a_Resumo_Modelos_Finais.csv"), index=False)

    df_jac = comparar_estabilidade_hvg_final(P)

    print("\n[4/4] Gerando MANIFESTO_INFERENCIA_EXTERNA.json...")
    manifesto = {
        "gerado_em": datetime.datetime.now().isoformat(timespec="seconds"),
        "coorte_referencia": "GSE267301 (Pan et al., 2024)",
        "n_celulas_total": int(len(y)),
        "n_celulas_treino_final_90pct": int(len(idx_tr)),
        "n_celulas_holdout_es_10pct": int(len(idx_va)),
        "n_doadores": int(len(np.unique(donors))),
        "n_genes_molde_pan": 61541,
        "sha256_genes_pan_61541_puros": hash_genes_puros,
        "seed_canonica_principal": int(seed_principal),
        "seeds_disponiveis": seeds,
        "esquema_pesos_congelado": decisao["pesos_escolhido"],
        "k_pca_sweep_congelado": int(decisao["k_escolhido"]),
        "dicionario_classes_base0_pytorch": {str(i): NOMES_CLASSES[i] for i in range(8)},
        "dicionario_classes_base1_pan": {str(i + 1): NOMES_CLASSES[i] for i in range(8)},
        "regras_pre_registradas_inferencia_externa": {
            "alinhamento_genomico": (
                "Toda matriz externa para projecao SWeeP deve estar alinhada posicionalmente "
                "aos 61.541 Ensembl IDs puros de genes_pan_61541.txt (mesmo SHA-256)."
            ),
            "tratamento_genes_ausentes_SWeeP": (
                "Zero estrutural (0) nas colunas dos genes nao medidos antes da multiplicacao "
                "pela base ortonormal W (61.541 x 600)."
            ),
            "tratamento_genes_ausentes_Baseline_HVG": (
                "1) Normalizar contagens brutas de cada celula externa (normalize_total=1e4 + log1p) "
                "sobre os genes medidos no estudo externo; "
                "2) Mapear os 2.000 Ensembl IDs de hvg2000_ensembl_ids_puros.txt na ordem exata; "
                "3) Preencher com zero estrutural (0.0) qualquer gene dos 2.000 HVGs ausente na coorte externa; "
                "4) Aplicar pca_hvg50.pkl seguido de scaler_hvg50.pkl e mlp_Baseline_HVG2000_PCA50.pt."
            ),
            "avaliacao_classes_ausentes": (
                "Em coortes externas onde determinadas classes biologicas nao existem (ex.: ausencia de "
                "Oligodendrocitos maduros em tecido fetal/xenogregos no GSE305140), reportar F1-macro "
                "restrito as classes presentes no ground truth e a taxa de falsos positivos nas classes ausentes."
            ),
        },
        "artefatos_por_braco": {
            "Baseline_HVG2000_PCA50": {
                **info_artefatos["Baseline_HVG2000_PCA50"],
                "mlp_canonico_pt": "mlp_Baseline_HVG2000_PCA50.pt",
                "arquitetura": "Linear(50->64)->ReLU->Linear(64->32)->ReLU->Linear(32->8)",
            },
            "SWeeP_600D_Puro": {
                **info_artefatos["SWeeP_600D_Puro"],
                "mlp_canonico_pt": "mlp_SWeeP_600D_Puro.pt",
                "arquitetura": "Linear(600->128)->ReLU->Linear(128->32)->ReLU->Linear(32->8)",
            },
            f"SWeeP_600D_PCA{decisao['k_escolhido']}": {
                **info_artefatos[f"SWeeP_600D_PCA{decisao['k_escolhido']}"],
                "mlp_canonico_pt": f"mlp_SWeeP_600D_PCA{decisao['k_escolhido']}.pt",
                "arquitetura": f"Linear({decisao['k_escolhido']}->64)->ReLU->Linear(64->32)->ReLU->Linear(32->8)",
            },
        },
        "ambiente": {
            "python": sys.version,
            "plataforma": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scanpy": versao("scanpy"),
            "scikit-learn": versao("scikit-learn"),
            "joblib": versao("joblib"),
        },
    }
    salvar_json(manifesto, os.path.join(P["etapa5"], "MANIFESTO_INFERENCIA_EXTERNA.json"))
    salvar_json(manifesto, os.path.join(P["artefatos_congelados"], "MANIFESTO_INFERENCIA_EXTERNA.json"))

    pd.set_option("display.width", 220)
    print("\n[RESUMO DOS MODELOS FINAIS CONGELADOS (90% treino / 10% hold-out ES)]")
    print(df_res.round(4).to_string(index=False))
    if df_jac is not None:
        print("\n[CONCORDANCIA DOS 2.000 HVGs FINAIS COM ETAPA 3 E FOLDS DO LODO]")
        print(df_jac.round(4).to_string(index=False))

    print("\n" + "=" * 78)
    print("ETAPA 5 CONCLUIDA COM SUCESSO!")
    print(f" -> Artefatos congelados prontos para inferencia externa em:\n    {P['artefatos_congelados']}")
    print("=" * 78)


if __name__ == "__main__":
    main()