#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETAPA 2b - GSE267301 (Pan et al.) - QUAL DIMENSAO DO SWeeP PRESERVA MAIS INFORMACAO?

Pergunta: a dimensao 600 do SWeeP foi fixada a priori. Aqui medimos, SO em Treino/Validacao
(o Teste Cego continua intocado), como a informacao preservada varia com a dimensao d.

Fatos que justificam o desenho (verificados em 06/11/2026):
  - A W oficial (resultados_SWeeP_GSE263468/orthbase_mproj_600d.rds, sha 1eee02c5...) e uma base
    ortonormal GAUSSIANA ALEATORIA: dp das entradas = 1/sqrt(61541), curtose ~0, normas de linha
    uniformes e variancia da projecao sem ordenacao entre colunas. Logo, as d primeiras colunas
    formam um SWeeP valido de dimensao d (bases ANINHADAS -> curva suave).
  - ||x W|| tem correlacao 0,995 com sqrt(n_genes detectados): a norma do vetor e quase so
    profundidade. Por isso testamos tambem a variante com normalizacao L2 por celula.

Representacoes avaliadas
  SWeeP:   base "oficial" (W do pipeline, d <= 600) e bases "aleatoria_s<seed>" (gaussiana + QR,
           d ate CFG["d_max"]). Variantes: "bruto" (como hoje) e "L2" (linha normalizada).
  HVG:     2.000 HVGs (seurat, ajustados SO no treino, log-norm) -> PCA k -> scaler (igual a Etapa 3).

Metricas (todas em VALIDACAO)
  A. MLP da Etapa 2 (mesma arquitetura/treino; pesos Balanced; 3 seeds): F1-macro, acuracia
     balanceada, F1 por classe.
  B. Sonda linear (regressao logistica, class_weight balanced): F1-macro. Separa "informacao
     linearmente disponivel" da capacidade da MLP.
  C. Preservacao da geometria (sem rotulos) numa amostra da validacao: Spearman entre similaridades
     cosseno no espaco binario completo (61.541 genes) e no espaco projetado; sobreposicao dos k
     vizinhos mais proximos (kNN@15).
  D. Silhueta (cosseno) por tipo celular (maior = melhor) e por doador (menor = melhor).

Decisao PRE-REGISTRADA (gravada em config.json antes de qualquer treino)
  Para cada (base, variante): d* = MENOR d com F1-macro medio de validacao >= melhor - dp_do_melhor
  (regra 1-SE, igual a Etapa 2). Base primaria: "aleatoria_s1" (cobre a grade toda); a base
  "oficial" serve de controle (deve concordar em d <= 600). As bases s2/s3 medem a variabilidade
  devida a sorte da base.

Uso
  python etapa2b_curva_dimensao_sweep_pan.py              # execucao completa (~1-2 h em CPU)
  python etapa2b_curva_dimensao_sweep_pan.py --rapido     # teste de fumaca (minutos)
  python etapa2b_curva_dimensao_sweep_pan.py --retomar <pasta_da_execucao>

Politica: nada e sobrescrito; cada execucao cria pasta nova; a tabela RAW e append-only e permite retomar.
"""
import os, sys, gc, json, time, copy, hashlib, argparse, datetime, platform, warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import numpy as np
import pandas as pd
import h5py
import scipy.sparse as sp
from scipy.stats import spearmanr
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score, silhouette_score
from sklearn.utils.class_weight import compute_class_weight

# ==============================================================================
# CONFIGURACAO
# ==============================================================================
BASE = "/media/gustavo-k-sato/D61CF48E1CF46B3F1/analisesPAN"
RUN_OFICIAL = os.path.join(BASE, "resultados_MLP_GSE267301_v2", "run_20261102_201556")
CAM_SPLIT = os.path.join(RUN_OFICIAL, "split_indices_70_15_15.npz")
CAM_METADADOS = os.path.join(BASE, "Metadados_Pan_Limpos.csv")
CAM_H5AD_LOG = os.path.join(BASE, "Pan_58k_LogNorm_Para_BaselineHVG.h5ad")
# W oficial ja convertida para binario pelo preflight da Etapa 5 v2 (cabecalho int32 x2 + float64 em ordem F)
CAM_W_OFICIAL_BIN = os.path.join(RUN_OFICIAL, "etapa5_modelos_finais_v2", "preflight_20261103_095621",
                                 "w_45d6frin", "W_convertida.bin")
CAM_W_OFICIAL_RDS = "/media/gustavo-k-sato/D61CF48E1CF46B3F1/AIBIA/R_analises/resultados_SWeeP_GSE263468/orthbase_mproj_600d.rds"
SHA_W_OFICIAL_RDS = "1eee02c5e7c26ca3b0132d32deb8441f099c050f5be6affca76f33aeb00569fa"
PASTA_RAIZ = os.path.join(BASE, "resultados_MLP_GSE267301_v2")

CFG = {
    "dims": [10, 25, 50, 100, 200, 300, 400, 600, 800, 1000, 1500, 2000],
    "d_max": 2000,
    "bases_completas": ["oficial", "aleatoria_s1"],     # todas as variantes x 3 seeds de MLP
    "bases_variabilidade": ["aleatoria_s2", "aleatoria_s3"],  # so variante "bruto", seed 42
    "variantes": ["bruto", "L2"],
    "seeds_treino": [42, 100, 2026],
    "hvg_n": 2000, "hvg_k": [10, 25, 50, 100, 200],
    "pesos": "Balanced",
    "epocas_max": 80, "paciencia": 10, "batch_treino": 256, "batch_eval": 512, "lr": 1e-3,
    "n_classes": 8, "n_genes": 61541,
    "geom_amostra": 3000, "geom_knn": 15, "geom_pares": 300_000, "seed_amostra": 7,
    "sonda_max_iter": 300,
    "regra_decisao": ("1-SE por (base, variante): menor d com media(F1-macro val, 3 seeds) >= "
                      "melhor_media - dp_do_melhor (ddof=1). Base primaria: aleatoria_s1."),
    "teste_cego_tocado": False,
}
CFG_RAPIDO = {"dims": [25, 100, 600], "d_max": 600, "seeds_treino": [42], "epocas_max": 3, "paciencia": 2,
              "bases_variabilidade": [], "hvg_k": [50], "geom_amostra": 800, "geom_pares": 20_000,
              "sub_treino": 6000, "sonda_max_iter": 50}
NOMES = ["Astrocytes", "Oligodendrocytes", "Exc_Neurons", "Inh_Neurons", "OPCs", "Microglia",
         "Endothelial", "VLMCs"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RETOMANDO = False


# ==============================================================================
# UTILITARIOS (sem overwrite)
# ==============================================================================
def checar_novo(c):
    if os.path.exists(c):
        if RETOMANDO:
            return False
        raise FileExistsError(f"Recusando sobrescrever: {c}")
    return True


def salvar_json(o, c):
    if checar_novo(c):
        with open(c, "w", encoding="utf-8") as f:
            json.dump(o, f, indent=2, ensure_ascii=False, default=str)


def sha256(c, limite=None):
    h = hashlib.sha256()
    with open(c, "rb") as f:
        while True:
            b = f.read(16 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fixar_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def log(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


# ==============================================================================
# DADOS
# ==============================================================================
def carregar_dados(rapido):
    """Metadados + split oficial (treino/val) + matriz binaria esparsa SO de treino+val."""
    meta = pd.read_csv(CAM_METADADOS)
    y = meta["Cell_Type_Num"].values.astype(int) - 1
    doador = meta["Donor"].astype(str).values
    z = np.load(CAM_SPLIT)
    idx_tr, idx_va = np.sort(z["idx_treino"]), np.sort(z["idx_val"])
    assert len(np.intersect1d(idx_tr, idx_va)) == 0
    assert len(np.intersect1d(idx_tr, z["idx_teste"])) == 0 and len(np.intersect1d(idx_va, z["idx_teste"])) == 0
    if rapido:
        rng = np.random.default_rng(0)
        idx_tr = np.sort(rng.choice(idx_tr, CFG["sub_treino"], replace=False))
    from anndata.io import read_elem
    with h5py.File(CAM_H5AD_LOG, "r") as f:
        obs_idx = read_elem(f["obs"]).index.astype(str).values
        assert np.array_equal(obs_idx, meta["Barcode"].astype(str).values), "h5ad x metadados desalinhados"
        X = f["X"]
        assert X.attrs.get("encoding-type") == "csr_matrix"
        assert int(X.attrs["shape"][1]) == CFG["n_genes"]
        ip = X["indptr"][:]

        def ler_binaria(linhas):                        # linhas ORDENADAS; teste cego nunca passa aqui
            blocos = []
            for i in range(0, len(linhas), 4000):
                rr = linhas[i:i + 4000]
                a0, b0 = int(ip[rr[0]]), int(ip[rr[-1] + 1])   # faixa contigua que cobre o bloco
                dat = X["data"][a0:b0]; col = X["indices"][a0:b0]
                sel_c, sel_p = [], [0]
                for r in rr:
                    a, b = int(ip[r]) - a0, int(ip[r + 1]) - a0
                    c_ = col[a:b][dat[a:b] > 0]                 # binarizacao: x > 0
                    sel_c.append(c_); sel_p.append(sel_p[-1] + len(c_))
                cc = np.concatenate(sel_c).astype(np.int32)
                blocos.append(sp.csr_matrix((np.ones(len(cc), np.float32), cc, np.array(sel_p)),
                                            shape=(len(rr), CFG["n_genes"])))
            return sp.vstack(blocos).tocsr()

        B_tr, B_va = ler_binaria(idx_tr), ler_binaria(idx_va)
    log(f"Binaria: treino {B_tr.shape} nnz={B_tr.nnz:,} | val {B_va.shape} (teste cego nao lido)")
    return {"B_tr": B_tr, "B_va": B_va, "y_tr": y[idx_tr], "y_va": y[idx_va],
            "don_va": doador[idx_va], "idx_tr": idx_tr, "idx_va": idx_va}


def carregar_lognorm_hvg(idx_tr, idx_va):
    """Igual a Etapa 3: 2.000 HVGs (seurat) ajustados SO no treino sobre o log-norm."""
    import scanpy as sc
    ad = sc.read_h5ad(CAM_H5AD_LOG, backed="r")
    ad_tr = ad[idx_tr].to_memory()
    sc.pp.highly_variable_genes(ad_tr, n_top_genes=CFG["hvg_n"], flavor="seurat")
    mask = ad_tr.var["highly_variable"].values
    assert int(mask.sum()) == CFG["hvg_n"]
    Htr = ad_tr.X[:, mask]; Htr = Htr.toarray() if sp.issparse(Htr) else np.asarray(Htr)
    Hva = ad[idx_va].to_memory().X[:, mask]; Hva = Hva.toarray() if sp.issparse(Hva) else np.asarray(Hva)
    ad.file.close()
    return Htr.astype(np.float32), Hva.astype(np.float32)


# ==============================================================================
# BASES SWeeP
# ==============================================================================
def base_oficial():
    assert sha256(CAM_W_OFICIAL_RDS) == SHA_W_OFICIAL_RDS, "W oficial (.rds) com sha256 diferente do manifesto!"
    d0, d1 = np.fromfile(CAM_W_OFICIAL_BIN, dtype="<i4", count=2)
    W = np.fromfile(CAM_W_OFICIAL_BIN, dtype="<f8", offset=8).reshape((d0, d1), order="F")
    assert W.shape == (CFG["n_genes"], 600)
    assert np.abs(W.T @ W - np.eye(600)).max() < 1e-8
    return W.astype(np.float32)


def base_aleatoria(seed, d_max, cache_dir):
    """Gaussiana N(0,1) 61541 x d_max -> QR reduzida (colunas ortonormais), como orthBase do SWeeP."""
    cam = os.path.join(cache_dir, f"W_aleatoria_s{seed}_{d_max}.npy")
    if os.path.exists(cam):
        return np.load(cam)
    t0 = time.time()
    G = np.random.default_rng(seed).standard_normal((CFG["n_genes"], d_max))
    Q, R = np.linalg.qr(G, mode="reduced")
    Q *= np.sign(np.diag(R))[None, :]          # QR unica (sinais fixos)
    Q = Q.astype(np.float32); del G, R; gc.collect()
    np.save(cam, Q)
    log(f"Base aleatoria s{seed} ({d_max}D) gerada em {time.time()-t0:.0f}s")
    return Q


def projetar(B, W):
    out = np.empty((B.shape[0], W.shape[1]), np.float32)
    for i in range(0, B.shape[0], 5000):
        out[i:i + 5000] = B[i:i + 5000] @ W
    return out


# ==============================================================================
# MLP (identica a Etapa 2) E SONDA LINEAR
# ==============================================================================
class ClassificadorCelular(nn.Module):
    def __init__(self, input_dim, num_classes=8):
        super().__init__()
        h1 = 128 if input_dim > 100 else 64
        self.camadas = nn.Sequential(nn.Linear(input_dim, h1), nn.ReLU(), nn.Linear(h1, 32), nn.ReLU(),
                                     nn.Linear(32, num_classes))

    def forward(self, x):
        return self.camadas(x)


def treinar_validar(X_tr, y_tr, X_va, y_va, seed):
    fixar_seed(seed); t0 = time.time()
    m = ClassificadorCelular(X_tr.shape[1], CFG["n_classes"]).to(DEVICE)
    pres = np.unique(y_tr); pesos = np.ones(CFG["n_classes"], np.float32)
    pesos[pres] = compute_class_weight("balanced", classes=pres, y=y_tr)
    lt = nn.CrossEntropyLoss(weight=torch.tensor(pesos).to(DEVICE)); lv_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), lr=CFG["lr"])
    ld_tr = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.long)),
                       batch_size=CFG["batch_treino"], shuffle=True)
    ld_va = DataLoader(TensorDataset(torch.tensor(X_va), torch.tensor(y_va, dtype=torch.long)),
                       batch_size=CFG["batch_eval"], shuffle=False)
    melhor, estado, ep_id, sem = float("inf"), None, 0, 0
    for ep in range(CFG["epocas_max"]):
        m.train()
        for bx, by in ld_tr:
            opt.zero_grad(); lt(m(bx.to(DEVICE)), by.to(DEVICE)).backward(); opt.step()
        m.eval(); ac = 0.0
        with torch.no_grad():
            for bx, by in ld_va:
                ac += lv_fn(m(bx.to(DEVICE)), by.to(DEVICE)).item() * bx.size(0)
        lv = ac / len(ld_va.dataset)
        if lv < melhor:
            melhor, ep_id, sem = lv, ep + 1, 0
            estado = {k: v.detach().cpu().clone() for k, v in copy.deepcopy(m.state_dict()).items()}
        else:
            sem += 1
            if sem >= CFG["paciencia"]:
                break
    m.load_state_dict({k: v.to(DEVICE) for k, v in estado.items()}); m.eval()
    with torch.no_grad():
        yp = np.concatenate([m(bx.to(DEVICE)).argmax(1).cpu().numpy() for bx, _ in ld_va])
    f1c = f1_score(y_va, yp, average=None, labels=list(range(8)), zero_division=0)
    r = {"F1_Macro_Val": f1_score(y_va, yp, average="macro", labels=list(range(8)), zero_division=0),
         "Acc_Bal_Val": balanced_accuracy_score(y_va, yp), "Loss_Val_CE": melhor,
         "Epoca_Ideal": ep_id, "Tempo_s": time.time() - t0}
    r.update({f"F1_{n}": float(v) for n, v in zip(NOMES, f1c)})
    return r


def sonda_linear(X_tr, y_tr, X_va, y_va):
    t0 = time.time()
    lr = LogisticRegression(max_iter=CFG["sonda_max_iter"], class_weight="balanced", n_jobs=-1)
    lr.fit(X_tr, y_tr); yp = lr.predict(X_va)
    return {"F1_Macro_Val_Sonda": f1_score(y_va, yp, average="macro", labels=list(range(8)), zero_division=0),
            "Tempo_Sonda_s": time.time() - t0}


def preparar(Z_tr, Z_va, variante):
    if variante == "L2":
        Z_tr, Z_va = normalize(Z_tr), normalize(Z_va)
    s = StandardScaler().fit(Z_tr)
    return s.transform(Z_tr).astype(np.float32), s.transform(Z_va).astype(np.float32)


# ==============================================================================
# GEOMETRIA (sem rotulos) E SILHUETA
# ==============================================================================
class Geometria:
    def __init__(self, B_va, y_va, don_va):
        rng = np.random.default_rng(CFG["seed_amostra"])
        self.s = np.sort(rng.choice(B_va.shape[0], min(CFG["geom_amostra"], B_va.shape[0]), replace=False))
        Bn = normalize(B_va[self.s])                      # cosseno no espaco binario completo
        S0 = (Bn @ Bn.T).toarray()
        n = len(self.s); k = CFG["geom_knn"]
        iu = np.triu_indices(n, 1)
        sel = rng.choice(len(iu[0]), min(CFG["geom_pares"], len(iu[0])), replace=False)
        self.pares = (iu[0][sel], iu[1][sel])
        self.s0 = S0[self.pares]
        np.fill_diagonal(S0, -np.inf)
        self.knn0 = np.argsort(-S0, axis=1)[:, :k]
        self.y, self.don = y_va[self.s], don_va[self.s]
        self.ref = {"Sil_Tipo": silhouette_score(Bn, self.y, metric="cosine"),
                    "Sil_Doador": silhouette_score(Bn, self.don, metric="cosine")}

    def medir(self, Z_va):
        Zn = normalize(Z_va[self.s]); S = Zn @ Zn.T
        rho = spearmanr(self.s0, S[self.pares]).correlation
        np.fill_diagonal(S, -np.inf)
        knn = np.argsort(-S, axis=1)[:, :CFG["geom_knn"]]
        sobre = np.mean([len(np.intersect1d(a, b)) / CFG["geom_knn"] for a, b in zip(self.knn0, knn)])
        return {"Geom_Spearman_Cos": rho, "Geom_kNN15": sobre,
                "Sil_Tipo": silhouette_score(Zn, self.y, metric="cosine"),
                "Sil_Doador": silhouette_score(Zn, self.don, metric="cosine")}


# ==============================================================================
# EXECUCAO
# ==============================================================================
def main():
    global RETOMANDO
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--retomar", default=None)
    ap.add_argument("--dims", nargs="+", type=int, help="sobrescreve a grade de dimensoes")
    ap.add_argument("--bases", nargs="+", help="ex.: aleatoria_s1 (bases completas)")
    ap.add_argument("--variantes", nargs="+")
    ap.add_argument("--sem_hvg", action="store_true")
    ap.add_argument("--rotulo", default="", help="sufixo do nome da pasta (ex.: extensao)")
    a = ap.parse_args()
    if a.rapido:
        CFG.update(CFG_RAPIDO)
    if a.dims:
        CFG["dims"] = a.dims; CFG["d_max"] = max(a.dims)
    if a.bases:
        CFG["bases_completas"] = a.bases; CFG["bases_variabilidade"] = []
    if a.variantes:
        CFG["variantes"] = a.variantes
    if a.sem_hvg:
        CFG["hvg_k"] = []
    if a.retomar:
        RETOMANDO = True; run = os.path.abspath(a.retomar)
    else:
        nome = ("etapa2b_RAPIDO_" if a.rapido else "etapa2b_dimensao_sweep_") + (a.rotulo + "_" if a.rotulo else "") + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run = os.path.join(PASTA_RAIZ, nome); os.makedirs(run, exist_ok=False)
    cache = os.path.join(run, "cache_bases"); os.makedirs(cache, exist_ok=True)
    cam_raw = os.path.join(run, "Tabela2b_RAW_MLP.csv"); cam_geo = os.path.join(run, "Tabela2b_Geometria.csv")
    cam_son = os.path.join(run, "Tabela2b_RAW_Sonda.csv")
    log(f"Pasta de saida: {run}")

    # ---------- pre-registro ----------
    salvar_json({"cfg": CFG, "script_sha256": sha256(os.path.abspath(__file__)),
                 "entradas": {"split": CAM_SPLIT, "split_sha256": sha256(CAM_SPLIT),
                              "metadados": CAM_METADADOS, "h5ad": CAM_H5AD_LOG,
                              "W_oficial_rds": CAM_W_OFICIAL_RDS, "W_oficial_sha256": SHA_W_OFICIAL_RDS},
                 "python": sys.version, "plataforma": platform.platform(), "torch": torch.__version__,
                 "inicio": datetime.datetime.now().isoformat()}, os.path.join(run, "config_preregistro.json"))

    feitos = set()
    if os.path.exists(cam_raw):
        for r in pd.read_csv(cam_raw).itertuples(index=False):
            feitos.add((r.Representacao, r.Base, int(r.D), r.Variante, int(r.Seed)))
    if os.path.exists(cam_son):
        for r in pd.read_csv(cam_son).itertuples(index=False):
            feitos.add((r.Representacao, r.Base, int(r.D), r.Variante, -1))
    geo_feitos = set()
    if os.path.exists(cam_geo):
        geo_feitos = {(r.Base, int(r.D)) for r in pd.read_csv(cam_geo).itertuples(index=False)}

    def anexar(cam, linha):
        pd.DataFrame([linha]).to_csv(cam, mode="a", header=not os.path.exists(cam), index=False)

    D = carregar_dados(a.rapido)
    geo = Geometria(D["B_va"], D["y_va"], D["don_va"])
    log(f"Referencia espaco binario completo: silhueta tipo={geo.ref['Sil_Tipo']:.3f} doador={geo.ref['Sil_Doador']:.3f}")
    salvar_json(geo.ref, os.path.join(run, "referencia_espaco_binario_completo.json"))

    plano = [(b, CFG["variantes"], CFG["seeds_treino"]) for b in CFG["bases_completas"]]
    plano += [(b, ["bruto"], [42]) for b in CFG["bases_variabilidade"]]

    for base, variantes, seeds in plano:
        if base == "oficial":
            W = base_oficial()
        else:
            W = base_aleatoria(int(base.split("_s")[1]), CFG["d_max"], cache)
        dims = [d for d in CFG["dims"] if d <= W.shape[1]]
        Ztr_all = projetar(D["B_tr"], W); Zva_all = projetar(D["B_va"], W)
        log(f"Base {base}: projecao {Ztr_all.shape[1]}D pronta; dims = {dims}")
        for d in dims:
            Ztr, Zva = Ztr_all[:, :d], Zva_all[:, :d]
            if (base, d) not in geo_feitos:
                g = geo.medir(Zva); anexar(cam_geo, {"Base": base, "D": d, **g}); geo_feitos.add((base, d))
                log(f"  [geom] {base} d={d}: Spearman={g['Geom_Spearman_Cos']:.3f} kNN15={g['Geom_kNN15']:.3f} "
                    f"silTipo={g['Sil_Tipo']:.3f} silDoador={g['Sil_Doador']:.3f}")
            for var in variantes:
                Xtr, Xva = preparar(Ztr, Zva, var)
                if ("SWeeP", base, d, var, -1) not in feitos:
                    s_ = sonda_linear(Xtr, D["y_tr"], Xva, D["y_va"])
                    anexar(cam_son, {"Representacao": "SWeeP", "Base": base, "D": d, "Variante": var, **s_})
                    feitos.add(("SWeeP", base, d, var, -1))
                    log(f"  [sonda] {base} d={d} {var}: F1={s_['F1_Macro_Val_Sonda']:.4f}")
                for seed in seeds:
                    if ("SWeeP", base, d, var, seed) in feitos:
                        continue
                    r = treinar_validar(Xtr, D["y_tr"], Xva, D["y_va"], seed)
                    anexar(cam_raw, {"Representacao": "SWeeP", "Base": base, "D": d, "Variante": var, "Seed": seed, **r})
                    feitos.add(("SWeeP", base, d, var, seed))
                    log(f"  [MLP] {base} d={d} {var} seed={seed}: F1={r['F1_Macro_Val']:.4f} "
                        f"End={r['F1_Endothelial']:.3f} VLMC={r['F1_VLMCs']:.3f} ({r['Tempo_s']:.0f}s)")
        del W, Ztr_all, Zva_all; gc.collect()

    # ---------- referencia HVG -> PCA k ----------
    if any(("HVG", "hvg2000", k, "bruto", s) not in feitos for k in CFG["hvg_k"] for s in CFG["seeds_treino"]):
        Htr, Hva = carregar_lognorm_hvg(D["idx_tr"], D["idx_va"])
        pca = PCA(n_components=max(CFG["hvg_k"]), random_state=42).fit(Htr)
        Ptr, Pva = pca.transform(Htr).astype(np.float32), pca.transform(Hva).astype(np.float32)
        for k in CFG["hvg_k"]:
            Xtr, Xva = preparar(Ptr[:, :k], Pva[:, :k], "bruto")
            if ("HVG", "hvg2000", k, "bruto", -1) not in feitos:
                s_ = sonda_linear(Xtr, D["y_tr"], Xva, D["y_va"])
                anexar(cam_son, {"Representacao": "HVG", "Base": "hvg2000", "D": k, "Variante": "bruto", **s_})
            for seed in CFG["seeds_treino"]:
                if ("HVG", "hvg2000", k, "bruto", seed) in feitos:
                    continue
                r = treinar_validar(Xtr, D["y_tr"], Xva, D["y_va"], seed)
                anexar(cam_raw, {"Representacao": "HVG", "Base": "hvg2000", "D": k, "Variante": "bruto", "Seed": seed, **r})
                log(f"  [MLP] HVG PCA k={k} seed={seed}: F1={r['F1_Macro_Val']:.4f}")

    resumir(run, cam_raw, cam_geo, cam_son)


# ==============================================================================
# RESUMO, DECISAO 1-SE E FIGURAS
# ==============================================================================
def resumir(run, cam_raw, cam_geo, cam_son):
    mlp = pd.read_csv(cam_raw); geo = pd.read_csv(cam_geo); sonda = pd.read_csv(cam_son)
    chave = ["Representacao", "Base", "Variante", "D"]
    res = mlp.groupby(chave).agg(F1_media=("F1_Macro_Val", "mean"), F1_dp=("F1_Macro_Val", "std"),
                                 n_seeds=("Seed", "nunique"), End_media=("F1_Endothelial", "mean"),
                                 VLMC_media=("F1_VLMCs", "mean"), Tempo_medio_s=("Tempo_s", "mean")).reset_index()
    res = res.merge(sonda[chave + ["F1_Macro_Val_Sonda"]], on=chave, how="left")
    res = res.merge(geo.rename(columns={"Base": "Base"}), on=["Base", "D"], how="left")
    res.to_csv(os.path.join(run, "Tabela2b_Resumo.csv"), index=False)

    dec = {}
    for (rep, base, var), g in res.groupby(["Representacao", "Base", "Variante"]):
        g = g[g.n_seeds >= 2].sort_values("D")
        if g.empty:
            continue
        top = g.loc[g.F1_media.idxmax()]
        lim = top.F1_media - (top.F1_dp if np.isfinite(top.F1_dp) else 0)
        esc = g[g.F1_media >= lim].iloc[0]
        dec[f"{rep}|{base}|{var}"] = {"d_melhor_bruto": int(top.D), "F1_melhor": float(top.F1_media),
                                      "limiar_1SE": float(lim), "d_escolhido_1SE": int(esc.D),
                                      "F1_escolhido": float(esc.F1_media)}
    dec["variabilidade_entre_bases_seed42"] = (
        mlp[mlp.Base.str.startswith("aleatoria") & (mlp.Variante == "bruto") & (mlp.Seed == 42)]
        .groupby("D").F1_Macro_Val.agg(["mean", "std", "count"]).round(4).reset_index().to_dict("records"))
    with open(os.path.join(run, f"decisao_dimensao_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"), "w") as f:
        json.dump(dec, f, indent=2)
    log("Decisao 1-SE: " + json.dumps({k: v.get("d_escolhido_1SE") for k, v in dec.items() if isinstance(v, dict)}))

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for (rep, base, var), g in res.groupby(["Representacao", "Base", "Variante"]):
        g = g.sort_values("D"); lab = f"{base} {var}" if rep == "SWeeP" else "HVG2000→PCA k"
        ax[0].errorbar(g.D, g.F1_media, yerr=g.F1_dp.fillna(0), marker="o", ms=3, capsize=2, label=lab)
        ax[1].plot(g.D, g.F1_Macro_Val_Sonda, marker="o", ms=3, label=lab)
    for base, g in geo.groupby("Base"):
        g = g.sort_values("D")
        ax[2].plot(g.D, g.Geom_kNN15, marker="o", ms=3, label=f"{base} kNN@15")
        ax[2].plot(g.D, g.Geom_Spearman_Cos, ls="--", marker=".", label=f"{base} Spearman cos")
    for a_, t in zip(ax, ["MLP: F1-macro validação (média ± dp, 3 seeds)", "Sonda linear: F1-macro validação",
                          "Preservação da geometria (vs. binário 61.541 genes)"]):
        a_.set_xscale("log"); a_.set_xlabel("dimensão (d ou k)"); a_.set_title(t, fontsize=10); a_.grid(alpha=.3)
        a_.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(run, "Fig_Curva_Dimensao_SWeeP.png"), dpi=150); plt.close(fig)
    log(f"Resumo, decisao e figura salvos em {run}")


if __name__ == "__main__":
    main()
