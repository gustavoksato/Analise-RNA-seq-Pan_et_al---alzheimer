#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETAPA 9 - PCA DESCRITIVO: SWeeP2000 vs HVG2000->PCA50, Pan + Dharsini (espaco ajustado no Pan)

Objetivo: observar a distribuicao das celulas nos dois espacos de representacao, no referencial do
modelo (ajustado no Pan), com o Dharsini projetado nele.

  HVG2000->PCA50 : transformador CONGELADO da Etapa 5 v2 (colunas HVG, media e componentes do PCA50).
                   Pan = h5ad log-norm; Dharsini = contagens -> normalize_total(1e4) + log1p.
  SWeeP2000      : (x>0) @ W_aleatoria_s1_2000 (base gaussiana+QR da Etapa 2b; ainda nao ha W-2000 oficial),
                   PCA(10) ajustado SO nas celulas de treino do Pan (split 70/15/15 oficial) e aplicado a todos.
  Referencia     : embeddings dos autores do Dharsini (X_pca_harmony, X_umap; corrigidos p/ quimica + regiao).

Amostragem: Pan inteiro (58.326); Dharsini estratificado ate CAP nucleos por doador x classe x Assay x SORT
            (Endothelial/VLMC entram todos). Pontos embaralhados antes de plotar (sem vies de ordem).
Cores: tipo celular, dataset, Assay x SORT (Dharsini) e n. de genes detectados (eixo de profundidade).

Uso: python etapa9_pca_sweep2000_hvg_pan_dharsini.py [--cap 150]
"""
import os, sys, json, argparse, datetime
import numpy as np
import pandas as pd
import h5py
from anndata.io import read_elem
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comum_coortes import *  # noqa

W2000 = f"{BASE}/resultados_MLP_GSE267301_v2/etapa2b_dimensao_sweep_20261106_113938/cache_bases/W_aleatoria_s1_2000.npy"
SPLIT_PAN = f"{RUN_OFICIAL}/split_indices_70_15_15.npz"
SAIDA_RAIZ = f"{BASE}/resultados_MLP_GSE267301_v2"

# paleta categorica de referencia (8 slots validados, ordem fixa) e rampa sequencial azul
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = LinearSegmentedColormap.from_list("azul", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
COR_TIPO = dict(zip(NOMES, CAT))                       # classe -> cor (fixa em todas as figuras)
ESTRATOS = ["10X-Chr-V3 | All Nuclei", "10X-Chr-V3 | NeuN", "10X-Chr-V2 | All Nuclei", "10X-Chr-V2 | NeuN",
            "DropSeq | All Nuclei", "DropSeq | NeuN"]
COR_EST = dict(zip(ESTRATOS, CAT[:6]))
COR_DS = {"Pan": CAT[0], "Dharsini": CAT[1]}
TINTA = "#3d3d3a"; MUDO = "#8a897f"


def amostra_dharsini(obs, cap, seed=11):
    y = obs["Cell_Type_Num"].values.astype(int) - 1
    est = (obs["Donor"] + "|" + obs["Assay"] + "|" + obs["SORT"] + "|" + y.astype(str)).values
    rng = np.random.default_rng(seed); sel = []
    for _, ix in pd.Series(np.arange(len(obs))).groupby(est).groups.items():
        ix = np.asarray(ix)
        lim = len(ix) if y[ix[0]] >= 6 else cap        # Endo/VLMC: todos
        sel.append(rng.choice(ix, lim, replace=False) if len(ix) > lim else ix)
    return np.sort(np.concatenate(sel))


def ler_reps(h5, linhas, A_hvg, W, ja_log, tam):
    L = LeitorCSR(h5); H, S, ng = [], [], []
    for _, Xc in L.blocos(linhas, tam=tam):
        Lg = Xc if ja_log else lognorm(Xc)
        Hc = Lg[:, A_hvg["hvg_cols"]].toarray().astype(np.float32)
        H.append((Hc - A_hvg["pca_mean"]) @ A_hvg["pca_components"].T)
        B = binarizar(Xc); S.append(np.asarray(B @ W, dtype=np.float32)); ng.append(np.diff(B.indptr))
    L.fechar()
    return np.vstack(H).astype(np.float32), np.vstack(S), np.concatenate(ng)


def eixo(ax, titulo):
    ax.set_title(titulo, fontsize=9, color=TINTA, loc="left")
    ax.tick_params(labelsize=7, colors=MUDO, length=2)
    for s in ax.spines.values():
        s.set_color("#d9d8d2"); s.set_linewidth(0.6)


def scatter(ax, x, y, cores, ordem):
    ax.scatter(x[ordem], y[ordem], c=np.asarray(cores, dtype=object)[ordem] if not np.isscalar(cores[0]) or isinstance(cores[0], str) else cores[ordem],
               s=1.2, alpha=0.45, linewidths=0, rasterized=True)


def legenda(ax, mapa, titulo):
    h = [Line2D([0], [0], marker="o", ls="", ms=5, mfc=c, mec="none", label=k) for k, c in mapa.items()]
    ax.legend(handles=h, title=titulo, fontsize=6.5, title_fontsize=7, frameon=False, loc="best", markerscale=1)


def painel(Z, meta, nome_rep, var_exp, pcs, out, lim):
    i, j = pcs
    x, y = Z[:, i], Z[:, j]
    rng = np.random.default_rng(0)
    pan = (meta.dataset == "Pan").values; dh = ~pan
    fig, axs = plt.subplots(2, 3, figsize=(15, 9.4), facecolor="white")
    lab = lambda k: f"PC{k + 1} ({var_exp[k] * 100:.1f}% var. no Pan)"

    def sub(ax, m, cores, titulo):
        idx = np.where(m)[0]; idx = idx[rng.permutation(len(idx))]
        ax.scatter(x[idx], y[idx], c=[cores[k] for k in idx] if isinstance(cores, list) else cores[idx],
                   s=1.2, alpha=0.45, linewidths=0, rasterized=True)
        eixo(ax, titulo); ax.set_xlim(lim[i]); ax.set_ylim(lim[j])
        ax.set_xlabel(lab(i), fontsize=7.5, color=MUDO); ax.set_ylabel(lab(j), fontsize=7.5, color=MUDO)

    ctipo = [COR_TIPO[t] for t in meta.tipo]
    sub(axs[0, 0], pan, ctipo, f"Pan ({pan.sum():,} núcleos) — tipo celular")
    legenda(axs[0, 0], COR_TIPO, "Tipo celular")
    sub(axs[0, 1], dh, ctipo, f"Dharsini ({dh.sum():,}, amostra estratificada) — tipo celular")
    legenda(axs[0, 1], COR_TIPO, "Tipo celular")
    cds = [COR_DS[d] for d in meta.dataset]
    sub(axs[0, 2], np.ones(len(meta), bool), cds, "Pan + Dharsini — dataset")
    legenda(axs[0, 2], COR_DS, "Dataset")
    cest = [COR_EST.get(e, MUDO) for e in meta.estrato]
    sub(axs[1, 0], dh, cest, "Dharsini — química × sorting")
    legenda(axs[1, 0], COR_EST, "Química | sorting")
    ng = np.log10(meta.n_genes.values.clip(1))
    vmin, vmax = np.quantile(ng, [0.01, 0.99])
    for ax, m, t in [(axs[1, 1], dh, "Dharsini — nº de genes detectados"), (axs[1, 2], pan, "Pan — nº de genes detectados")]:
        idx = np.where(m)[0]; idx = idx[rng.permutation(len(idx))]
        sc_ = ax.scatter(x[idx], y[idx], c=ng[idx], cmap=SEQ, vmin=vmin, vmax=vmax, s=1.2, alpha=0.55,
                         linewidths=0, rasterized=True)
        eixo(ax, t); ax.set_xlim(lim[i]); ax.set_ylim(lim[j])
        ax.set_xlabel(lab(i), fontsize=7.5, color=MUDO); ax.set_ylabel(lab(j), fontsize=7.5, color=MUDO)
    cb = fig.colorbar(sc_, ax=axs[1, 1:], shrink=0.8, pad=0.01)
    cb.set_label("log10(genes detectados)", fontsize=7.5, color=TINTA); cb.ax.tick_params(labelsize=7, colors=MUDO)
    fig.suptitle(f"{nome_rep} — PC{i + 1} × PC{j + 1} (PCA ajustado no Pan; Dharsini projetado)",
                 fontsize=11, color=TINTA, x=0.01, ha="left")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)


def limites(Z, k=4):
    return [tuple(np.quantile(Z[:, c], [0.002, 0.998]) + np.array([-1, 1]) * 0.05 * np.ptp(np.quantile(Z[:, c], [0.002, 0.998])))
            for c in range(k)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cap", type=int, default=150); a = ap.parse_args()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"{SAIDA_RAIZ}/etapa9_pca_sweep2000_hvg_{ts}"; os.makedirs(out, exist_ok=False)
    log(f"Saida: {out}")
    with np.load(f"{ART}/Baseline_HVG2000_PCA50/transformador.npz", allow_pickle=False) as z:
        A = {k: z[k] for k in z.files}
    W = np.load(W2000)
    assert W.shape == (61541, 2000) and np.abs(W[:, :50].T @ W[:, :50] - np.eye(50)).max() < 1e-4

    # ---------------- Pan (todos) ----------------
    meta_p = pd.read_csv(META_PAN)
    with h5py.File(H5_PAN_LOG, "r") as f:
        assert np.array_equal(read_elem(f["obs"]).index.astype(str).values, meta_p["Barcode"].astype(str).values)
    Hp, Sp, ngp = ler_reps(H5_PAN_LOG, np.arange(len(meta_p)), A, W, ja_log=True, tam=8000)
    log(f"Pan: {len(meta_p):,} núcleos projetados")

    # ---------------- Dharsini (amostra) ----------------
    with h5py.File(H5_DHARSINI, "r") as f:
        obs = read_elem(f["obs"])
        for c in obs.columns:
            if hasattr(obs[c], "cat"):
                obs[c] = obs[c].astype(str)
        idx = amostra_dharsini(obs, a.cap)
        emb = {k: f["obsm"][k][idx] for k in ("X_pca_harmony", "X_umap")}
    Hd, Sd, ngd = ler_reps(H5_DHARSINI, idx, A, W, ja_log=False, tam=1500)
    od = obs.iloc[idx]
    log(f"Dharsini: {len(idx):,} núcleos (amostra estratificada, cap={a.cap})")

    # ---------------- PCA do SWeeP2000 ajustado no treino do Pan ----------------
    tr = np.sort(np.load(SPLIT_PAN)["idx_treino"])
    pca = PCA(n_components=10, random_state=42).fit(Sp[tr])
    Zs = pca.transform(np.vstack([Sp, Sd])).astype(np.float32)
    Zh = np.vstack([Hp, Hd])[:, :10]
    var_s = pca.explained_variance_ratio_
    var_h = A["pca_var_ratio"][:10]

    meta = pd.DataFrame({
        "dataset": ["Pan"] * len(meta_p) + ["Dharsini"] * len(idx),
        "tipo": [NOMES[t - 1] for t in meta_p["Cell_Type_Num"].astype(int)] + [NOMES[t - 1] for t in od["Cell_Type_Num"].astype(int)],
        "doador": list(meta_p["Donor"].astype(str)) + list(od["Donor"]),
        "estrato": ["Pan (10X V3.1 | glia)"] * len(meta_p) + list(od["Assay"] + " | " + od["SORT"]),
        "regiao": [""] * len(meta_p) + list(od["Brain.Region"]),
        "ADNC": [""] * len(meta_p) + list(od["Disease.Group"]),
        "n_genes": np.concatenate([ngp, ngd])})

    for nome, Z, v in [("SWeeP2000 (base aleatória s1)", Zs, var_s), ("HVG2000 → PCA50 (congelado, Etapa 5)", Zh, var_h)]:
        tag = "SWeeP2000" if nome.startswith("SWeeP") else "HVG2000_PCA50"
        lim = limites(Z)
        for pcs in [(0, 1), (1, 2)]:
            painel(Z, meta, nome, v, pcs, f"{out}/Fig9_{tag}_PC{pcs[0] + 1}xPC{pcs[1] + 1}.png", lim)

    # ---------------- variancia explicada (Pan) ----------------
    fig, ax = plt.subplots(figsize=(6.4, 3.4), facecolor="white")
    k = np.arange(1, 11)
    ax.plot(k, var_h * 100, marker="o", ms=4, lw=2, color=CAT[0], label="HVG2000 → PCA50")
    ax.plot(k, var_s * 100, marker="o", ms=4, lw=2, color=CAT[1], label="SWeeP2000 → PCA")
    ax.text(10.15, var_h[-1] * 100, "HVG", fontsize=7.5, color=TINTA, va="center")
    ax.text(10.15, var_s[-1] * 100, "SWeeP", fontsize=7.5, color=TINTA, va="center")
    eixo(ax, "Variância explicada por PC (ajuste no Pan)")
    ax.set_xlabel("componente", fontsize=8, color=MUDO); ax.set_ylabel("% da variância", fontsize=8, color=MUDO)
    ax.grid(axis="y", color="#ecebe6", lw=0.6); ax.legend(fontsize=7.5, frameon=False)
    fig.savefig(f"{out}/Fig9_variancia_explicada.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # ---------------- embeddings dos autores (Dharsini) ----------------
    fig, axs = plt.subplots(2, 2, figsize=(11, 9.4), facecolor="white")
    md = meta[meta.dataset == "Dharsini"].reset_index(drop=True)
    rng = np.random.default_rng(1); o = rng.permutation(len(md))
    for r, (k_, nomek) in enumerate([("X_pca_harmony", "PCA Harmony (PC1 × PC2)"), ("X_umap", "UMAP")]):
        E = emb[k_]
        for c_, (col, mapa, tit) in enumerate([("tipo", COR_TIPO, "tipo celular"), ("estrato", COR_EST, "química × sorting")]):
            ax = axs[r, c_]
            ax.scatter(E[o, 0], E[o, 1], c=[mapa.get(t, MUDO) for t in md[col].values[o]], s=1.2, alpha=0.45,
                       linewidths=0, rasterized=True)
            eixo(ax, f"Autores — {nomek} — {tit}"); legenda(ax, mapa, tit.capitalize())
    fig.suptitle("Dharsini — embeddings dos autores (integrados por química + região), mesma amostra",
                 fontsize=11, color=TINTA, x=0.01, ha="left")
    fig.savefig(f"{out}/Fig9_autores_Dharsini_harmony_umap.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # ---------------- tabelas ----------------
    cols = {f"SWeeP2000_PC{c + 1}": Zs[:, c] for c in range(10)}
    cols.update({f"HVG_PC{c + 1}": Zh[:, c] for c in range(10)})
    pd.concat([meta, pd.DataFrame(cols)], axis=1).to_csv(f"{out}/T9_coordenadas_PC1a10.csv.gz", index=False)
    pd.DataFrame({"PC": k, "HVG2000_PCA50_var_pct": var_h * 100, "SWeeP2000_PCA_var_pct": var_s * 100}) \
        .to_csv(f"{out}/T9_variancia_explicada.csv", index=False)
    json.dump({"cap_dharsini": a.cap, "n_pan": int(len(meta_p)), "n_dharsini": int(len(idx)),
               "W_sweep": W2000, "pca_sweep_ajuste": "treino Pan (split 70/15/15 oficial)",
               "hvg": "transformador congelado Etapa 5 v2", "inicio": ts}, open(f"{out}/config.json", "w"), indent=2)
    log("Figuras e tabelas salvas.")
    log("Variância explicada (%, PCs 1-5): HVG " + str(np.round(var_h[:5] * 100, 1)) + " | SWeeP " + str(np.round(var_s[:5] * 100, 1)))


if __name__ == "__main__":
    main()
