#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETAPA 7c - FIGURAS DO TREINO CRUZADO (re-pontuacao das predicoes salvas da Etapa 7; nada e retreinado)

Fig 1: treino Pan  -> inferencia Dharsini  (so estratos All Nuclei; NeuN+ fora da metrica principal)
Fig 2: treino Dharsini -> inferencia Pan   (Pan inteiro e sem ALSP; F1 por doador)

Metrica: F1-macro nas classes 1-6 ELEGIVEIS (>= 50 nucleos e >= 3 doadores no escopo).
Ponto  : F1 da confusao somada das 3 seeds (42, 100, 2026).
IC 95% : bootstrap de DOADORES (B = 2000) sobre a confusao somada das seeds.
"""
import os, sys, json, datetime
import numpy as np
import pandas as pd
import h5py
from anndata.io import read_elem
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comum_coortes import *  # noqa

R = f"{BASE}/resultados_MLP_GSE267301_v2"
E7a = f"{R}/etapa7_pan_para_dharsini_20261106_131808"
E7b = f"{R}/etapa7_dharsini_para_pan_20261106_132919"
REPS = {"HVG2000_PCA50": "HVG2000 → PCA50", "SWeeP_600D_oficial_bruto": "SWeeP 600", "SWeeP_2000D_aleat_s1_bruto": "SWeeP 2000"}
COR = {"HVG2000_PCA50": "#2a78d6", "SWeeP_600D_oficial_bruto": "#eb6834", "SWeeP_2000D_aleat_s1_bruto": "#1baf7a"}
MARC = {"HVG2000_PCA50": "o", "SWeeP_600D_oficial_bruto": "s", "SWeeP_2000D_aleat_s1_bruto": "D"}
SEEDS = (42, 100, 2026)
TINTA, MUDO, GRADE = "#3d3d3a", "#8a897f", "#ecebe6"
NOME_CL = {0: "Astrócitos", 1: "Oligodendrócitos", 2: "Exc.", 3: "Inib.", 4: "OPC", 5: "Microglia"}


def elegiveis(y, d, m):
    return [c for c in range(6) if (m & (y == c)).sum() >= 50 and len(np.unique(d[m & (y == c)])) >= 3]


def estimar(y, d, preds, m, B=2000, seed=0):
    """F1-macro (classes elegiveis) da confusao somada das seeds + IC por bootstrap de doadores."""
    m = m & (y < 6); cl = elegiveis(y, d, m); doad = np.unique(d[m]); out = {}
    for r, ps in preds.items():
        pil = np.stack([sum(confusao(y[m & (d == x)], p[m & (d == x)]) for p in ps) for x in doad])
        rng = np.random.default_rng(seed)
        bt = [f1_macro(pil[rng.integers(0, len(doad), len(doad))].sum(0), cl) for _ in range(B)]
        f1c = f1_de_confusao(pil.sum(0), cl)
        out[r] = {"F1": f1_macro(pil.sum(0), cl), "lo": ic(bt)[0], "hi": ic(bt)[1],
                  **{NOME_CL[c]: v for c, v in zip(cl, f1c)}, "n_doadores": len(doad), "n_nucleos": int(m.sum())}
    return out


def estilo(ax, titulo):
    ax.set_title(titulo, fontsize=10, color=TINTA, loc="left", pad=8)
    ax.tick_params(labelsize=8, colors=MUDO, length=0)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d9d8d2")
    ax.grid(axis="x", color=GRADE, lw=0.7); ax.set_axisbelow(True)


def dots(ax, linhas, res, xlim, rotulo_x):
    """linhas: [(rotulo, chave)], res[chave][rep] -> dict. Um ponto por representacao, deslocado verticalmente."""
    desl = {r: o for r, o in zip(REPS, (-0.22, 0, 0.22))}
    for i, (rot, k) in enumerate(linhas):
        for r in REPS:
            v = res[k][r]; yy = i + desl[r]
            ax.errorbar(v["F1"], yy, xerr=[[v["F1"] - v["lo"]], [v["hi"] - v["F1"]]], fmt="none",
                        ecolor=COR[r], elinewidth=2, capsize=0, alpha=0.9)
            ax.plot(v["F1"], yy, MARC[r], ms=8, mfc=COR[r], mec="white", mew=1.5)
            ax.text(min(v["hi"], xlim[1]) + (xlim[1] - xlim[0]) * 0.012, yy, f"{v['F1']:.3f}", va="center",
                    fontsize=7.5, color=TINTA)
    ax.set_yticks(range(len(linhas))); ax.set_yticklabels([l[0] for l in linhas], fontsize=8.5, color=TINTA)
    ax.set_ylim(len(linhas) - 0.5, -0.5); ax.set_xlim(*xlim)
    ax.set_xlabel(rotulo_x, fontsize=8.5, color=MUDO)


def legenda(fig):
    h = [Line2D([0], [0], marker=MARC[r], ls="", ms=8, mfc=COR[r], mec="white", label=REPS[r]) for r in REPS]
    fig.legend(handles=h, loc="upper right", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.99, 0.995))


def main():
    out = f"{R}/etapa7c_figuras_treino_cruzado_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    os.makedirs(out, exist_ok=False)
    with h5py.File(H5_DHARSINI, "r") as f:
        obs = read_elem(f["obs"])
    for c in obs.columns:
        if hasattr(obs[c], "cat"):
            obs[c] = obs[c].astype(str)
    y_all = obs.Cell_Type_Num.values.astype(int) - 1; d_all = obs.Donor.values
    # indices de teste da Etapa 7 (Pan -> Dharsini), reconstruidos de forma deterministica
    homo = obs.Homologo_10XV3_AllNuclei.values.astype(bool)
    est = (obs.Donor + "|" + obs.Assay + "|" + obs.SORT).values
    rng = np.random.default_rng(3); sel = []
    g = pd.Series(est[~homo], index=np.where(~homo)[0])
    for _, ix in g.groupby(g).groups.items():
        ix = np.asarray(ix); sel.append(rng.choice(ix, 1000, replace=False) if len(ix) > 1000 else ix)
    i_te = np.sort(np.concatenate([np.where(homo)[0], np.sort(np.concatenate(sel))]))
    assert len(i_te) == json.load(open(f"{E7a}/config.json"))["n_teste"]
    y, d = y_all[i_te], d_all[i_te]; s = (obs.Assay + " | " + obs.SORT).values[i_te]
    P = {r: [np.load(f"{E7a}/predicoes/ypred_{r}_seed{x}.npy") for x in SEEDS] for r in REPS}
    esc1 = [("10X v3 · All Nuclei\n(Homólogo, 16 doadores)", s == "10X-Chr-V3 | All Nuclei"),
            ("10X v2 · All Nuclei\n(7 doadores)", s == "10X-Chr-V2 | All Nuclei"),
            ("DropSeq · All Nuclei\n(29 doadores)", s == "DropSeq | All Nuclei")]
    res1 = {rot: estimar(y, d, P, m) for rot, m in esc1}

    # ---------- Figura 1 ----------
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.05, 1], "wspace": 0.42}, facecolor="white")
    dots(a, [(r, r) for r, _ in esc1], res1, (0.90, 1.005), "F1-macro (6 classes) · ponto = 3 seeds · barra = IC 95% por doador")
    estilo(a, "A · F1-macro por estrato (só All Nuclei)")
    k_ds = esc1[2][0]
    linhas_b = [(NOME_CL[c], c) for c in range(6)]
    resb = {c: {r: {"F1": res1[k_ds][r][NOME_CL[c]], "lo": res1[k_ds][r][NOME_CL[c]], "hi": res1[k_ds][r][NOME_CL[c]]}
                for r in REPS} for c in range(6)}
    dots(b, linhas_b, resb, (0.80, 1.005), "F1 por classe (3 seeds somadas)")
    estilo(b, "B · DropSeq · All Nuclei, por classe")
    legenda(fig)
    fig.suptitle("Treino no Pan (GSE267301) → inferência no Dharsini (GSE263468)", fontsize=12, color=TINTA,
                 x=0.01, ha="left", y=1.02)
    fig.text(0.01, -0.04, "Estratos NeuN+ fora da métrica principal (quase sem glia). Classes com ≥ 50 núcleos e ≥ 3 doadores. "
             "SWeeP 2000 = base aleatória s1 (W-2000 oficial pendente). Eixo x não começa em zero.",
             fontsize=7.5, color=MUDO)
    fig.savefig(f"{out}/Fig7c_1_treino_Pan_inferencia_Dharsini.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # ---------- Figura 2 ----------
    meta = pd.read_csv(META_PAN); yp = meta.Cell_Type_Num.values - 1; dp = meta.Donor.astype(str).values
    Q = {r: [np.load(f"{E7b}/predicoes/ypred_{r}_seed{x}.npy") for x in SEEDS] for r in REPS}
    esc2 = [("Pan, todos\n(8 doadores)", np.ones(len(meta), bool)), ("Pan sem ALSP\n(7 doadores)", dp != "ALSP")]
    res2 = {rot: estimar(yp, dp, Q, m) for rot, m in esc2}
    doad = sorted(np.unique(dp), key=lambda x: (x == "ALSP", x))
    res_d = {}
    for x in doad:
        m = (dp == x) & (yp < 6)
        res_d[x] = {r: {"F1": f1_macro(sum(confusao(yp[m], p[m]) for p in ps)), "lo": np.nan, "hi": np.nan} for r, ps in Q.items()}
        for r in REPS:
            res_d[x][r]["lo"] = res_d[x][r]["hi"] = res_d[x][r]["F1"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.30}, facecolor="white")
    dots(a, [(r, r) for r, _ in esc2], res2, (0.90, 1.005), "F1-macro (6 classes) · ponto = 3 seeds · barra = IC 95% por doador")
    estilo(a, "A · F1-macro no Pan")
    dots(b, [(f"doador {x}", x) for x in doad], res_d, (0.80, 1.005), "F1-macro por doador (3 seeds somadas)")
    estilo(b, "B · Por doador do Pan")
    legenda(fig)
    fig.suptitle("Treino no Dharsini (GSE263468; 32 doadores, validação por doador) → inferência no Pan (GSE267301)",
                 fontsize=12, color=TINTA, x=0.01, ha="left", y=1.02)
    fig.text(0.01, -0.04, "Teste no Pan: uma química (10X v3.1), sem NeuN+. Só 8 doadores: ICs por bootstrap são aproximados. "
             "SWeeP 2000 > HVG em todos os 8 doadores. Eixo x não começa em zero.", fontsize=7.5, color=MUDO)
    fig.savefig(f"{out}/Fig7c_2_treino_Dharsini_inferencia_Pan.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    linhas = []
    for nome, rr in [("Pan->Dharsini", res1), ("Dharsini->Pan", res2), ("Dharsini->Pan por doador", res_d)]:
        for esc_, v in rr.items():
            for r, dd in v.items():
                linhas.append({"Direcao": nome, "Escopo": esc_.replace("\n", " "), "Representacao": REPS[r], **dd})
    pd.DataFrame(linhas).to_csv(f"{out}/T7c_valores_das_figuras.csv", index=False)
    log(f"Figuras salvas em {out}")


if __name__ == "__main__":
    main()
