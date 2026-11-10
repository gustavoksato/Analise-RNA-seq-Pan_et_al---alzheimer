# -*- coding: utf-8 -*-
"""Funcoes comuns as Etapas 6b (Pan -> Dharsini) e 7 (Dharsini -> Pan).

- Caminhos canonicos (W oficial, run travado, h5ad pos-QC do GSE263468, h5ad log-norm do Pan)
- Leitura em blocos de linhas de um h5ad CSR (sem carregar tudo na RAM)
- Normalizacao igual a do Pan: normalize_total(1e4) + log1p
- MLP identica a das Etapas 2-5
- Metricas por confusao 8x8, F1-macro restrito as classes 1-6 e bootstrap pareado por DOADOR
"""
import os, json, hashlib, datetime
import numpy as np
import pandas as pd
import h5py
import scipy.sparse as sp
import torch
from torch import nn

DISCO = "/media/gustavo-k-sato/D61CF48E1CF46B3F1"
BASE = f"{DISCO}/analisesPAN"
RUN_OFICIAL = f"{BASE}/resultados_MLP_GSE267301_v2/run_20261102_201556"
E5 = f"{RUN_OFICIAL}/etapa5_modelos_finais_v2"
ART = f"{E5}/artefatos_congelados"
GENES_PAN_ENSG = f"{E5}/etapa5c_chave_ensembl/genes_pan_61541_ensembl_sem_versao.txt"
W_BIN = f"{E5}/preflight_20261103_095621/w_45d6frin/W_convertida.bin"
W_RDS = f"{DISCO}/AIBIA/R_analises/resultados_SWeeP_GSE263468/orthbase_mproj_600d.rds"
W_SHA = "1eee02c5e7c26ca3b0132d32deb8441f099c050f5be6affca76f33aeb00569fa"
H5_DHARSINI = (f"{DISCO}/Analise-RNA-seq-Pan_et_al---alzheimer/Documents/export_GSE263468_h5ad_posQC/"
               "GSE263468_posQC_Geral_SemVazamento_counts.h5ad")
H5_PAN_LOG = f"{BASE}/Pan_58k_LogNorm_Para_BaselineHVG.h5ad"
META_PAN = f"{BASE}/Metadados_Pan_Limpos.csv"
TARGET_SUM = 1e4
NOMES = ["Astrocytes", "Oligodendrocytes", "Exc_Neurons", "Inh_Neurons", "OPCs", "Microglia",
         "Endothelial", "VLMCs"]
CLASSES_AVAL = list(range(6))           # classes 1-6 (indices 0-5); 7-8 so descritivas
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(m):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {m}", flush=True)


def sha256(c):
    h = hashlib.sha256()
    with open(c, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


def salvar_json_novo(o, c):
    if os.path.exists(c):
        raise FileExistsError(f"Recusando sobrescrever: {c}")
    with open(c, "w", encoding="utf-8") as f:
        json.dump(o, f, indent=2, ensure_ascii=False, default=str)


# ------------------------------------------------------------------ W oficial
def carregar_W_oficial():
    assert sha256(W_RDS) == W_SHA, "W oficial (.rds) com sha256 diferente do manifesto!"
    d0, d1 = np.fromfile(W_BIN, dtype="<i4", count=2)
    W = np.fromfile(W_BIN, dtype="<f8", offset=8).reshape((d0, d1), order="F")
    assert W.shape == (61541, 600) and np.abs(W.T @ W - np.eye(600)).max() < 1e-8
    return W.astype(np.float32)


# ------------------------------------------------------------------ leitura h5ad
class LeitorCSR:
    """Le blocos de linhas (indices ORDENADOS) de X de um h5ad CSR."""
    def __init__(self, caminho):
        self.f = h5py.File(caminho, "r")
        self.X = self.f["X"]
        assert self.X.attrs.get("encoding-type") == "csr_matrix"
        self.shape = tuple(int(v) for v in self.X.attrs["shape"])
        self.ip = self.X["indptr"][:]

    def linhas(self, rr):
        rr = np.asarray(rr)
        a0, b0 = int(self.ip[rr[0]]), int(self.ip[rr[-1] + 1])
        dat = self.X["data"][a0:b0]; col = self.X["indices"][a0:b0]
        D, C, P = [], [], [0]
        for r in rr:
            a, b = int(self.ip[r]) - a0, int(self.ip[r + 1]) - a0
            D.append(dat[a:b]); C.append(col[a:b]); P.append(P[-1] + (b - a))
        return sp.csr_matrix((np.concatenate(D).astype(np.float32), np.concatenate(C), np.array(P)),
                             shape=(len(rr), self.shape[1]))

    def blocos(self, linhas, tam=20000):
        linhas = np.sort(np.asarray(linhas))
        for i in range(0, len(linhas), tam):
            yield i, self.linhas(linhas[i:i + tam])

    def fechar(self):
        self.f.close()


def lognorm(Xc):
    s = np.asarray(Xc.sum(1)).ravel(); s[s == 0] = 1.0
    X = sp.diags((TARGET_SUM / s).astype(np.float32)) @ Xc
    X = X.tocsr(); X.data = np.log1p(X.data)
    return X


def binarizar(X):
    B = X.copy(); B.data = (B.data > 0).astype(np.float32); B.eliminate_zeros()
    return B


# ------------------------------------------------------------------ MLP
class ClassificadorCelular(nn.Module):
    def __init__(self, input_dim, num_classes=8):
        super().__init__()
        h1 = 128 if input_dim > 100 else 64
        self.camadas = nn.Sequential(nn.Linear(input_dim, h1), nn.ReLU(), nn.Linear(h1, 32), nn.ReLU(),
                                     nn.Linear(32, num_classes))

    def forward(self, x):
        return self.camadas(x)


def prever(modelo, Z, batch=4096):
    modelo.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(Z), batch):
            out.append(modelo(torch.tensor(Z[i:i + batch], dtype=torch.float32).to(DEVICE)).argmax(1).cpu().numpy())
    return np.concatenate(out).astype(np.int8)


# ------------------------------------------------------------------ metricas
def confusao(y, yp, n=8):
    return np.bincount(y.astype(np.int64) * n + yp.astype(np.int64), minlength=n * n).reshape(n, n)


def f1_de_confusao(M, classes=CLASSES_AVAL):
    """F1 por classe a partir da confusao 8x8 (linhas = verdade). Predicoes em 7/8 contam como erro."""
    f1 = []
    for c in classes:
        tp = M[c, c]; fn = M[c].sum() - tp; fp = M[:, c].sum() - tp
        # FP so de celulas cujas verdades estao no universo avaliado (classes 1-6)
        fp = M[classes, c].sum() - tp
        den = 2 * tp + fn + fp
        f1.append(2 * tp / den if den > 0 else np.nan)
    return np.array(f1, dtype=float)


def f1_macro(M, classes=CLASSES_AVAL):
    v = f1_de_confusao(M, classes)
    return float(np.nanmean(v))


def confusoes_por_doador(y, yp, doador):
    return {d: confusao(y[doador == d], yp[doador == d]) for d in np.unique(doador)}


def bootstrap_doadores(conf_por_braco, B=2000, seed=0):
    """conf_por_braco: {braco: {doador: M8x8}}. Reamostra DOADORES (pareado entre bracos).
    Devolve DataFrame com F1-macro por replica e braco."""
    bracos = list(conf_por_braco)
    doadores = sorted(conf_por_braco[bracos[0]])
    rng = np.random.default_rng(seed)
    pilha = {b: np.stack([conf_por_braco[b][d] for d in doadores]) for b in bracos}
    linhas = []
    for _ in range(B):
        idx = rng.integers(0, len(doadores), len(doadores))
        linhas.append({b: f1_macro(pilha[b][idx].sum(0)) for b in bracos})
    return pd.DataFrame(linhas)


def ic(v, a=0.05):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return float(np.quantile(v, a / 2)), float(np.quantile(v, 1 - a / 2))
