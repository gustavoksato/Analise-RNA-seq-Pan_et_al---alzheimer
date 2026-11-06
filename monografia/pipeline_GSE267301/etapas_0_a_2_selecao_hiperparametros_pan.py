#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETAPAS 0 A 2 - GSE267301 (Pan et al.) - AUDITORIA E SELECAO DE HIPERPARAMETROS

Etapa 0: configuracao, pastas de saida (nova pasta por execucao, sem overwrite), regras de decisao.
Etapa 1: auditoria de alinhamento (h5ad x metadados x TSV SWeeP) e tabela Doador x Classe.
Etapa 2: selecao de hiperparametros usando SOMENTE Treino e Validacao.
         O Teste Cego NUNCA e construido nem carregado aqui: apenas os indices sao salvos.
         (a) esquema de pesos: Balanced vs Sem_Pesos (decisao unica para os 3 bracos)
         (b) numero de PCs k no SWeeP+PCA (regra 1-SE)

Uso:
    python etapas_0_a_2_selecao_hiperparametros_pan.py
    python etapas_0_a_2_selecao_hiperparametros_pan.py --retomar /caminho/da/run_AAAAMMDD_HHMMSS

Politica de escrita: nenhum arquivo existente e sobrescrito. Cada execucao cria uma pasta nova
(run_<timestamp>). Em --retomar, o que ja existe e mantido e os treinos ja registrados sao pulados.
"""

import os
import sys
import gc
import json
import time
import copy
import platform
import argparse
import datetime

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

# ==============================================================================
# ETAPA 0 - CAMINHOS E CONFIGURACAO
# ==============================================================================
BASE = "/media/gustavo-k-sato/D61CF48E1CF46B3F1/analisesPAN"

CAM_SWEEP_600 = "/media/gustavo-k-sato/D61CF48E1CF46B3F1/AIBIA/R_analises/SWeeP_600_Classes_Alzheimer_Oficial.tsv"
CAM_METADADOS = os.path.join(BASE, "Metadados_Pan_Limpos.csv")
CAM_H5AD_LOG = os.path.join(BASE, "Pan_58k_LogNorm_Para_BaselineHVG.h5ad")
CAM_MOLDE_ENSG = os.path.join(BASE, "features_ENSG_puro.tsv")  # usado so como checagem opcional

PASTA_RAIZ = os.path.join(BASE, "resultados_MLP_GSE267301_v2")

# Subpastas (as das Etapas 3 a 5 ja ficam definidas para os proximos scripts)
SUBPASTAS = {
    "etapa1": "etapa1_auditoria",
    "etapa2": "etapa2_selecao_hiperparametros",
    "etapa3": "etapa3_teste_cego",
    "etapa4": "etapa4_lodo",
    "etapa5": "etapa5_modelos_finais",
}

CFG = {
    "seed_split": 42,                 # seed do split 70/15/15 (fixa, igual ao script anterior)
    "seeds_treino": [42, 100, 2026],
    "frac_treino": 0.70,
    "frac_val": 0.15,
    "frac_teste": 0.15,
    "k_lista": [50, 45, 40, 35, 30, 25, 20, 15, 10],
    "k_max_pca": 50,
    "epocas_max": 80,
    "paciencia": 10,                  # para se a loss de validacao nao melhora por N epocas
    "batch_treino": 256,
    "batch_eval": 512,
    "lr": 1e-3,
    "n_classes": 8,
    "n_genes_esperado": 61541,
    "n_dim_sweep": 600,
    "criterio_selecao": "F1-macro de VALIDACAO no checkpoint de menor loss de validacao (CE sem pesos)",
    "regra_pesos": ("Media (entre as 2 representacoes SWeeP) do F1-macro de validacao por seed. "
                    "Sem_Pesos so vence se (media_Sem - media_Bal) > 1 desvio-padrao das seeds do Balanced; "
                    "caso contrario, Balanced (empate -> Balanced)."),
    "regra_k": ("1-SE: acha o k de maior F1-macro medio de validacao (3 seeds); "
                "escolhe o MENOR k com media >= media_do_melhor - desvio_do_melhor (ddof=1)."),
    "RODAR_ABLACOES": False,          # reservado para as Etapas 3/4 (flag desligada ate decisao do orientador)
}

NOMES_CLASSES = ["1: Astrocytes", "2: Oligodendrocytes", "3: Exc. Neurons", "4: Inh. Neurons",
                 "5: OPCs", "6: Microglia", "7: Endothelial", "8: VLMCs"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------
# Utilitarios de escrita SEM overwrite
# ------------------------------------------------------------------------------
RETOMANDO = False


def _checar_novo(caminho):
    """Retorna True se pode escrever; False se deve pular (retomada); levanta erro se sobrescreveria."""
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


def salvar_fig(fig, caminho):
    if _checar_novo(caminho):
        fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)


def preparar_pastas(retomar):
    global RETOMANDO
    if retomar:
        RETOMANDO = True
        run_dir = os.path.abspath(retomar)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"Pasta de retomada nao existe: {run_dir}")
    else:
        os.makedirs(PASTA_RAIZ, exist_ok=True)
        carimbo = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(PASTA_RAIZ, f"run_{carimbo}")
        os.makedirs(run_dir, exist_ok=False)  # falha se ja existir
    caminhos = {"run": run_dir}
    for chave, nome in SUBPASTAS.items():
        caminhos[chave] = os.path.join(run_dir, nome)
        os.makedirs(caminhos[chave], exist_ok=True)
    return caminhos


def fixar_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------------------
# MLP e treino (versao de SELECAO: nao recebe teste)
# ------------------------------------------------------------------------------
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


def treinar_validar(X_tr, y_tr, X_va, y_va, usar_pesos, seed):
    """Treina e devolve metricas de VALIDACAO no checkpoint de menor loss de validacao (CE sem pesos)."""
    fixar_seed(seed)
    t0 = time.time()
    modelo = ClassificadorCelular(X_tr.shape[1], CFG["n_classes"]).to(DEVICE)

    if usar_pesos:
        presentes = np.unique(y_tr)
        pesos = np.ones(CFG["n_classes"], dtype=np.float32)
        pesos[presentes] = compute_class_weight("balanced", classes=presentes, y=y_tr)
        loss_treino = nn.CrossEntropyLoss(weight=torch.tensor(pesos, dtype=torch.float32).to(DEVICE))
    else:
        loss_treino = nn.CrossEntropyLoss()
    loss_val_fn = nn.CrossEntropyLoss()  # SEM pesos, para escala comparavel entre os esquemas

    opt = torch.optim.Adam(modelo.parameters(), lr=CFG["lr"])
    ld_tr = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)),
                       batch_size=CFG["batch_treino"], shuffle=True)
    ld_va = DataLoader(TensorDataset(torch.tensor(X_va, dtype=torch.float32), torch.tensor(y_va, dtype=torch.long)),
                       batch_size=CFG["batch_eval"], shuffle=False)

    melhor, melhor_estado, epoca_ideal, sem_melhora = float("inf"), None, 0, 0
    epocas_rodadas = 0
    for ep in range(CFG["epocas_max"]):
        modelo.train()
        for bx, by in ld_tr:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            loss_treino(modelo(bx), by).backward()
            opt.step()
        modelo.eval()
        acum = 0.0
        with torch.no_grad():
            for bx, by in ld_va:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                acum += loss_val_fn(modelo(bx), by).item() * bx.size(0)
        lv = acum / len(ld_va.dataset)
        epocas_rodadas = ep + 1
        if lv < melhor:
            melhor, epoca_ideal, sem_melhora = lv, ep + 1, 0
            melhor_estado = {k: v.detach().cpu().clone() for k, v in copy.deepcopy(modelo.state_dict()).items()}
        else:
            sem_melhora += 1
            if sem_melhora >= CFG["paciencia"]:
                break

    modelo.load_state_dict({k: v.to(DEVICE) for k, v in melhor_estado.items()})
    modelo.eval()
    preds = []
    with torch.no_grad():
        for bx, _ in ld_va:
            preds.append(modelo(bx.to(DEVICE)).argmax(1).cpu().numpy())
    yp = np.concatenate(preds)

    return {
        "F1_Macro_Val": f1_score(y_va, yp, average="macro", labels=list(range(8)), zero_division=0),
        "Acc_Bal_Val": balanced_accuracy_score(y_va, yp),
        "Acc_Val": accuracy_score(y_va, yp),
        "F1_Endothelial_Val": f1_score(y_va, yp, average=None, labels=list(range(8)), zero_division=0)[6],
        "F1_VLMCs_Val": f1_score(y_va, yp, average=None, labels=list(range(8)), zero_division=0)[7],
        "Loss_Val_CE": melhor,
        "Epoca_Ideal": epoca_ideal,
        "Epocas_Rodadas": epocas_rodadas,
        "Tempo_s": time.time() - t0,
    }


# ==============================================================================
# ETAPA 1 - AUDITORIA
# ==============================================================================
def etapa1_auditoria(P):
    print("\n" + "=" * 78)
    print("ETAPA 1 - AUDITORIA ESTRITA DE ALINHAMENTO E SUPORTE")
    print("=" * 78)
    for nome, cam in [("SWeeP TSV", CAM_SWEEP_600), ("Metadados", CAM_METADADOS), ("h5ad LogNorm", CAM_H5AD_LOG)]:
        if not os.path.exists(cam):
            raise FileNotFoundError(f"{nome} nao encontrado: {cam}")

    print("[1.1] Carregando SWeeP 600D e metadados...")
    df_sw = pd.read_csv(CAM_SWEEP_600, sep="\t")
    y_tsv = df_sw["Cell_Type"].values.astype(int)
    X = df_sw.drop(columns=["Cell_Type"]).values.astype(np.float32)
    del df_sw
    df_meta = pd.read_csv(CAM_METADADOS)

    print("[1.2] Lendo apenas obs/var do h5ad (modo backed, sem carregar a matriz)...")
    ad_log = sc.read_h5ad(CAM_H5AD_LOG, backed="r")
    obs_index = ad_log.obs.index.values.astype(str)
    var_names = ad_log.var_names.values.astype(str)
    n_obs_h5, n_vars_h5 = ad_log.n_obs, ad_log.n_vars
    ad_log.file.close()
    del ad_log

    print("[1.3] Asserts de alinhamento...")
    for col in ["Barcode", "Donor", "Cell_Type_Num"]:
        assert col in df_meta.columns, f"Coluna ausente nos metadados: {col}"
    assert X.shape[1] == CFG["n_dim_sweep"], f"SWeeP com {X.shape[1]} dimensoes (esperado {CFG['n_dim_sweep']})"
    assert n_vars_h5 == CFG["n_genes_esperado"], f"h5ad com {n_vars_h5} genes (esperado {CFG['n_genes_esperado']})"
    assert X.shape[0] == len(y_tsv) == len(df_meta) == n_obs_h5, (
        f"Tamanhos divergentes: SWeeP={X.shape[0]}, TSV rotulos={len(y_tsv)}, "
        f"metadados={len(df_meta)}, h5ad={n_obs_h5}")
    assert np.array_equal(obs_index, df_meta["Barcode"].astype(str).values), \
        "obs_names do h5ad DIFEREM (conteudo ou ordem) da coluna Barcode dos metadados!"
    assert np.array_equal(y_tsv, df_meta["Cell_Type_Num"].values.astype(int)), \
        "Cell_Type do TSV SWeeP difere de Cell_Type_Num dos metadados (possivel desalinhamento de ordem)!"
    assert set(np.unique(y_tsv)) == set(range(1, 9)), f"Classes presentes: {np.unique(y_tsv)} (esperado 1..8)"
    assert not np.isnan(X).any(), "SWeeP contem NaN!"
    assert len(set(obs_index)) == len(obs_index), "Barcodes duplicados no h5ad!"
    print("   -> OK: h5ad == metadados (Barcode, ordem identica).")
    print("   -> OK: TSV SWeeP == metadados por ROTULO (o TSV nao contem barcodes; verificacao indireta).")

    # Checagem opcional da ordem de genes contra o molde (apenas informativa)
    genes_puros = pd.Series(var_names).str.replace(r"\..*$", "", regex=True).values
    if os.path.exists(CAM_MOLDE_ENSG):
        molde = pd.read_csv(CAM_MOLDE_ENSG, sep="\t", header=None)[0].astype(str).str.replace(r"\..*$", "", regex=True).values
        if len(molde) == len(genes_puros):
            print(f"   -> Genes vs molde features_ENSG_puro.tsv: {int(np.sum(molde == genes_puros))}/{len(genes_puros)} identicos na mesma posicao.")
        else:
            print(f"   -> [AVISO] molde tem {len(molde)} linhas (h5ad: {len(genes_puros)}).")

    print("[1.4] Salvando ordem dos genes e tabela Doador x Classe...")
    salvar_txt(var_names, os.path.join(P["etapa1"], "genes_pan_61541_var_names_originais.txt"))
    salvar_txt(genes_puros, os.path.join(P["etapa1"], "genes_pan_61541.txt"))

    ct = pd.crosstab(df_meta["Donor"].astype(str), df_meta["Cell_Type_Num"].astype(int))
    ct = ct.reindex(columns=range(1, 9), fill_value=0)
    ct.columns = NOMES_CLASSES
    ct["Total"] = ct.sum(axis=1)
    ct.loc["TOTAL"] = ct.sum(axis=0)
    salvar_df(ct, os.path.join(P["etapa1"], "Tabela0_Suporte_Doador_x_Classe.csv"))
    print("\n" + ct.to_string())

    corpo = ct.drop(index="TOTAL").drop(columns="Total")
    aviso = []
    for d in corpo.index:
        for c in corpo.columns:
            n = int(corpo.loc[d, c])
            if n < 20:
                aviso.append({"Doador": d, "Classe": c, "N_Celulas": n})
    df_aviso = pd.DataFrame(aviso, columns=["Doador", "Classe", "N_Celulas"])
    salvar_df(df_aviso, os.path.join(P["etapa1"], "Tabela0b_Celulas_Doador_Classe_menor_que_20.csv"), index=False)
    print(f"\n   -> {len(df_aviso)} combinacoes Doador x Classe com < 20 celulas "
          f"(F1 por doador dessas classes e instavel no LODO).")

    return X, y_tsv - 1, df_meta["Donor"].astype(str).values


# ==============================================================================
# ETAPA 2 - SELECAO NA VALIDACAO
# ==============================================================================
def criar_split(y, P):
    idx = np.arange(len(y))
    idx_tr, idx_tmp, y_tr, y_tmp = train_test_split(
        idx, y, test_size=CFG["frac_val"] + CFG["frac_teste"],
        random_state=CFG["seed_split"], stratify=y)
    idx_va, idx_te, _, _ = train_test_split(
        idx_tmp, y_tmp, test_size=CFG["frac_teste"] / (CFG["frac_val"] + CFG["frac_teste"]),
        random_state=CFG["seed_split"], stratify=y_tmp)
    cam = os.path.join(P["run"], "split_indices_70_15_15.npz")
    if _checar_novo(cam):
        np.savez(cam, idx_treino=idx_tr, idx_val=idx_va, idx_teste=idx_te)
    print(f"   -> Split: treino={len(idx_tr)} | validacao={len(idx_va)} | teste cego={len(idx_te)} "
          f"(teste: SO indices salvos, nenhum dado de teste e usado nesta etapa)")
    return idx_tr, idx_va


def carregar_cache(cam_csv):
    if os.path.exists(cam_csv):
        df = pd.read_csv(cam_csv)
        return {(r.Representacao, int(r.K), r.Pesos, int(r.Seed)): r._asdict() for r in df.itertuples(index=False)}
    return {}


def anexar_linha(cam_csv, linha):
    pd.DataFrame([linha]).to_csv(cam_csv, mode="a", header=not os.path.exists(cam_csv), index=False)


def etapa2_selecao(X, y, P):
    print("\n" + "=" * 78)
    print("ETAPA 2 - SELECAO DE HIPERPARAMETROS (TREINO/VALIDACAO; TESTE CEGO INTOCADO)")
    print("=" * 78)
    idx_tr, idx_va = criar_split(y, P)
    y_tr, y_va = y[idx_tr], y[idx_va]
    X_tr_raw, X_va_raw = X[idx_tr], X[idx_va]

    # Features SWeeP 600D (scaler ajustado so no treino)
    sc600 = StandardScaler()
    F600 = (sc600.fit_transform(X_tr_raw), sc600.transform(X_va_raw))

    # PCA de 50 componentes ajustado so no treino (recortar as k primeiras == refazer com k componentes)
    pca = PCA(n_components=CFG["k_max_pca"], random_state=CFG["seed_split"])
    Ztr = pca.fit_transform(X_tr_raw)
    Zva = pca.transform(X_va_raw)
    var_acum = np.cumsum(pca.explained_variance_ratio_) * 100

    def feats(repr_nome, k):
        if repr_nome == "SWeeP_600D_Puro":
            return F600
        sck = StandardScaler()
        return sck.fit_transform(Ztr[:, :k]), sck.transform(Zva[:, :k])

    cam_raw = os.path.join(P["etapa2"], "Tabela1_Selecao_Validacao_RAW.csv")  # append-only, uma linha por treino
    cache = carregar_cache(cam_raw)
    if cache:
        print(f"   [retomada] {len(cache)} treinos ja registrados serao reaproveitados.")

    def rodar(repr_nome, k, pesos, seed):
        chave = (repr_nome, int(k), pesos, int(seed))
        if chave in cache:
            return cache[chave]
        Xtr, Xva = feats(repr_nome, k)
        r = treinar_validar(Xtr, y_tr, Xva, y_va, usar_pesos=(pesos == "Balanced"), seed=seed)
        linha = {"Representacao": repr_nome, "K": int(k), "Pesos": pesos, "Seed": int(seed), **r}
        anexar_linha(cam_raw, linha)
        cache[chave] = linha
        print(f"   [{repr_nome} k={k:>3} {pesos:<9} seed={seed:>4}] "
              f"F1-macro val={r['F1_Macro_Val']:.4f} | epoca={r['Epoca_Ideal']:02d}/{r['Epocas_Rodadas']:02d} | {r['Tempo_s']:.0f}s")
        return linha

    # ---------------- 2a: decisao de pesos ----------------
    print("\n[2a] Esquema de pesos (SWeeP_600D_Puro e SWeeP+PCA k=50; 3 seeds)")
    reprs_pesos = [("SWeeP_600D_Puro", CFG["n_dim_sweep"]), ("SWeeP_600D_PCA", CFG["k_max_pca"])]
    medias_por_seed = {"Balanced": [], "Sem_Pesos": []}
    for pesos in ["Balanced", "Sem_Pesos"]:
        for seed in CFG["seeds_treino"]:
            vals = [rodar(rn, k, pesos, seed)["F1_Macro_Val"] for rn, k in reprs_pesos]
            medias_por_seed[pesos].append(float(np.mean(vals)))

    m_bal, s_bal = np.mean(medias_por_seed["Balanced"]), np.std(medias_por_seed["Balanced"], ddof=1)
    m_sem, s_sem = np.mean(medias_por_seed["Sem_Pesos"]), np.std(medias_por_seed["Sem_Pesos"], ddof=1)
    delta = m_sem - m_bal
    pesos_escolhido = "Sem_Pesos" if delta > s_bal else "Balanced"
    tab_pesos = pd.DataFrame([
        {"Pesos": "Balanced", "F1_Macro_Val_Media": m_bal, "DP_Seeds": s_bal, "Valores_por_seed": medias_por_seed["Balanced"]},
        {"Pesos": "Sem_Pesos", "F1_Macro_Val_Media": m_sem, "DP_Seeds": s_sem, "Valores_por_seed": medias_por_seed["Sem_Pesos"]},
    ])
    salvar_df(tab_pesos, os.path.join(P["etapa2"], "Tabela1a_Decisao_Pesos.csv"), index=False)
    print(f"\n   Balanced : {m_bal:.4f} +- {s_bal:.4f}")
    print(f"   Sem_Pesos: {m_sem:.4f} +- {s_sem:.4f}")
    print(f"   Delta (Sem - Bal) = {delta:+.4f} | limiar = 1 DP do Balanced = {s_bal:.4f}")
    print(f"   >>> DECISAO UNICA DE PESOS PARA OS 3 BRACOS: {pesos_escolhido}")

    # ---------------- 2b: curva de k (1-SE) ----------------
    print(f"\n[2b] Curva de PCs no SWeeP+PCA (pesos={pesos_escolhido}; 3 seeds)")
    linhas_k = []
    for k in CFG["k_lista"]:
        res = [rodar("SWeeP_600D_PCA", k, pesos_escolhido, s) for s in CFG["seeds_treino"]]
        f1s = np.array([r["F1_Macro_Val"] for r in res])
        linhas_k.append({
            "K": k,
            "Var_Explicada_Acum_%": float(var_acum[k - 1]),
            "F1_Macro_Val_Media": f1s.mean(),
            "F1_Macro_Val_DP": f1s.std(ddof=1),
            "Acc_Bal_Val_Media": float(np.mean([r["Acc_Bal_Val"] for r in res])),
            "F1_Endothelial_Val_Media": float(np.mean([r["F1_Endothelial_Val"] for r in res])),
            "F1_VLMCs_Val_Media": float(np.mean([r["F1_VLMCs_Val"] for r in res])),
            "Epoca_Ideal_Media": float(np.mean([r["Epoca_Ideal"] for r in res])),
        })
    df_k = pd.DataFrame(linhas_k)
    i_melhor = int(df_k["F1_Macro_Val_Media"].idxmax())
    k_melhor = int(df_k.loc[i_melhor, "K"])
    m_melhor = float(df_k.loc[i_melhor, "F1_Macro_Val_Media"])
    dp_melhor = float(df_k.loc[i_melhor, "F1_Macro_Val_DP"])
    limiar = m_melhor - dp_melhor
    elegiveis = df_k[df_k["F1_Macro_Val_Media"] >= limiar]
    k_escolhido = int(elegiveis["K"].min())
    df_k["Dentro_1SE"] = df_k["F1_Macro_Val_Media"] >= limiar
    df_k["Escolhido"] = df_k["K"] == k_escolhido
    salvar_df(df_k, os.path.join(P["etapa2"], "Tabela1b_Curva_k_Validacao.csv"), index=False)
    print("\n" + df_k.to_string(index=False))
    print(f"\n   Melhor k bruto = {k_melhor} (F1 val {m_melhor:.4f} +- {dp_melhor:.4f}); limiar 1-SE = {limiar:.4f}")
    print(f"   >>> k ESCOLHIDO (menor k dentro de 1 DP do melhor): {k_escolhido}")

    # ---------------- Figuras ----------------
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.errorbar(df_k["K"], df_k["F1_Macro_Val_Media"], yerr=df_k["F1_Macro_Val_DP"], marker="o", capsize=3)
    ax.axhline(limiar, ls="--", color="gray", label=f"limiar 1-SE = {limiar:.4f}")
    ax.axvline(k_escolhido, ls=":", color="red", label=f"k escolhido = {k_escolhido}")
    ax.set_xlabel("Numero de PCs (k)")
    ax.set_ylabel("F1-macro de validacao (media +- DP, 3 seeds)")
    ax.set_title(f"SWeeP + PCA: curva de dimensionalidade (pesos={pesos_escolhido})")
    ax.invert_xaxis()
    ax.legend()
    ax.grid(True, ls=":", alpha=0.5)
    salvar_fig(fig, os.path.join(P["etapa2"], "Fig_Curva_k_Validacao.png"))

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.bar(["Balanced", "Sem_Pesos"], [m_bal, m_sem], yerr=[s_bal, s_sem], capsize=4)
    ax.set_ylim(min(m_bal, m_sem) - 0.02, max(m_bal, m_sem) + 0.01)
    ax.set_ylabel("F1-macro de validacao")
    ax.set_title("Decisao de pesos")
    salvar_fig(fig, os.path.join(P["etapa2"], "Fig_Decisao_Pesos.png"))

    # ---------------- Decisao registrada ----------------
    decisao = {
        "gerado_em": datetime.datetime.now().isoformat(timespec="seconds"),
        "pesos_escolhido": pesos_escolhido,
        "k_escolhido": k_escolhido,
        "k_melhor_bruto": k_melhor,
        "limiar_1SE": limiar,
        "pesos_detalhe": {"Balanced_media": m_bal, "Balanced_dp": s_bal,
                          "Sem_Pesos_media": m_sem, "Sem_Pesos_dp": s_sem, "delta": delta},
        "regra_pesos": CFG["regra_pesos"],
        "regra_k": CFG["regra_k"],
        "criterio": CFG["criterio_selecao"],
        "teste_cego_tocado": False,
    }
    salvar_json(decisao, os.path.join(P["run"], "decisao_hiperparametros.json"))
    return decisao


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retomar", default=None, help="Pasta run_* existente para continuar (nao sobrescreve nada).")
    args = ap.parse_args()

    P = preparar_pastas(args.retomar)
    print(f"[INFO] Dispositivo: {DEVICE}")
    print(f"[INFO] Pasta desta execucao: {P['run']}")

    # ---- Etapa 0: registrar configuracao e ambiente ----
    print("\n" + "=" * 78)
    print("ETAPA 0 - CONFIGURACAO")
    print("=" * 78)
    config_completo = {
        "cfg": CFG,
        "caminhos_entrada": {"sweep_tsv": CAM_SWEEP_600, "metadados": CAM_METADADOS, "h5ad_lognorm": CAM_H5AD_LOG},
        "pastas_saida": P,
        "ambiente": {"python": sys.version, "plataforma": platform.platform(), "torch": torch.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "scanpy": sc.__version__,
                     "device": str(DEVICE)},
    }
    salvar_json(config_completo, os.path.join(P["run"], "config_etapas_0_a_2.json"))
    print(f"   Criterio: {CFG['criterio_selecao']}")
    print(f"   Regra pesos: {CFG['regra_pesos']}")
    print(f"   Regra k: {CFG['regra_k']}")
    print(f"   RODAR_ABLACOES = {CFG['RODAR_ABLACOES']} (nao usado nestas etapas)")

    X, y, doadores = etapa1_auditoria(P)
    decisao = etapa2_selecao(X, y, P)

    print("\n" + "=" * 78)
    print("ETAPAS 0 A 2 CONCLUIDAS")
    print(f" -> Pesos (unico, 3 bracos): {decisao['pesos_escolhido']}")
    print(f" -> k do SWeeP+PCA: {decisao['k_escolhido']}")
    print(f" -> Decisao salva em: {os.path.join(P['run'], 'decisao_hiperparametros.json')}")
    print(" -> Teste cego NAO foi tocado.")
    print("=" * 78)


if __name__ == "__main__":
    main()