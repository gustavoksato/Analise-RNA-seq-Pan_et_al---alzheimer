#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETAPA 7 - TREINO CRUZADO NAS DUAS DIRECOES, MESMO PROTOCOLO
   --direcao dharsini_para_pan : treina no GSE263468 (Dharsini), testa no GSE267301 (Pan)
   --direcao pan_para_dharsini : treina no Pan, testa no Dharsini (Homologo inteiro + Geral amostrado)

Por que um protocolo proprio (e nao so a Etapa 6b): os modelos congelados da Etapa 5 tem 8 classes e
d=600 fixo. Para comparar as DIRECOES de forma justa, aqui os dois sentidos usam exatamente o mesmo:
  - classes 1-6 (Endothelial/VLMC tem ~20 celulas no Dharsini -> fora do treino e da metrica)
  - mesma MLP, pesos Balanced, early stopping na loss de validacao (CE sem pesos), 3 seeds
  - mesmos bracos: HVG2000->PCA50 (HVGs seurat ajustados SO no treino) e SWeeP_<d>D (bruto e L2)
  - SWeeP: d <= 600 usa a W OFICIAL; d > 600 usa uma base aleatoria gaussiana+QR (--base_npy),
    validada na Etapa 2b (a sorte da base muda o F1 em ~0,003 para d >= 300)
Validacao (early stopping)
  - Dharsini: doadores separados (~20%, estratificados por ADNC, >= 20 celulas por classe)
  - Pan: o split 70/15/15 oficial (treino e validacao; o teste cego do Pan NAO e usado no treino)
Os 6 doadores do Pan ja foram removidos do Dharsini: nao ha doador em comum entre treino e teste.

Uso:
  python etapa7_reverso_dharsini_para_pan.py --direcao dharsini_para_pan --dims 600 2000 \
         --base_npy .../cache_bases/W_aleatoria_s1_2000.npy
"""
import os, sys, json, argparse, datetime, time, copy
import numpy as np
import pandas as pd
import h5py
import scipy.sparse as sp
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from anndata.io import read_elem
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comum_coortes import *  # noqa

SAIDA_RAIZ = f"{BASE}/resultados_MLP_GSE267301_v2"
CAM_SPLIT_PAN = f"{RUN_OFICIAL}/split_indices_70_15_15.npz"
CFG = {"cap_treino": 600, "cap_val": 300, "frac_val_doadores": 0.2, "min_val_por_classe": 20,
       "cap_teste_geral": 1000, "seeds": [42, 100, 2026], "epocas_max": 80, "paciencia": 10,
       "batch": 256, "lr": 1e-3, "n_hvg": 2000, "k_pca_hvg": 50}


def fixar_seed(s):
    np.random.seed(s); torch.manual_seed(s)


def treinar(Xtr, ytr, Xva, yva, seed, nc=6):
    fixar_seed(seed); t0 = time.time()
    m = ClassificadorCelular(Xtr.shape[1], nc).to(DEVICE)
    pres = np.unique(ytr); w = np.ones(nc, np.float32); w[pres] = compute_class_weight("balanced", classes=pres, y=ytr)
    lt = nn.CrossEntropyLoss(weight=torch.tensor(w).to(DEVICE)); lv = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), lr=CFG["lr"])
    dtr = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.long)), batch_size=CFG["batch"], shuffle=True)
    dva = DataLoader(TensorDataset(torch.tensor(Xva), torch.tensor(yva, dtype=torch.long)), batch_size=4096)
    best, est, ep_b, sem = np.inf, None, 0, 0
    for ep in range(CFG["epocas_max"]):
        m.train()
        for bx, by in dtr:
            opt.zero_grad(); lt(m(bx.to(DEVICE)), by.to(DEVICE)).backward(); opt.step()
        m.eval(); acc = 0.0
        with torch.no_grad():
            for bx, by in dva:
                acc += lv(m(bx.to(DEVICE)), by.to(DEVICE)).item() * len(by)
        l_ = acc / len(yva)
        if l_ < best:
            best, ep_b, sem = l_, ep + 1, 0
            est = {k: v.detach().cpu().clone() for k, v in copy.deepcopy(m.state_dict()).items()}
        else:
            sem += 1
            if sem >= CFG["paciencia"]:
                break
    m.load_state_dict(est); m.eval()
    return m, {"Epoca_ideal": ep_b, "Loss_val": best, "Tempo_s": time.time() - t0}


def amostrar(grupo, mask, cap, seed):
    """Ate `cap` celulas por grupo (rotulo de estrato) dentro de mask."""
    rng = np.random.default_rng(seed); sel = []
    g = pd.Series(grupo[mask], index=np.where(mask)[0])
    for _, ix in g.groupby(g).groups.items():
        ix = np.asarray(ix)
        sel.append(rng.choice(ix, cap, replace=False) if len(ix) > cap else ix)
    return np.sort(np.concatenate(sel))


def ler(caminho, linhas, W):
    """(log-norm CSR, SWeeP D-dim) das linhas. O h5ad do Pan ja e log-norm; o do Dharsini e contagem."""
    L = LeitorCSR(caminho); Lg, Sw = [], []
    for _, Xc in L.blocos(linhas):
        Lg.append(Xc if caminho == H5_PAN_LOG else lognorm(Xc))
        Sw.append(np.asarray(binarizar(Xc) @ W, dtype=np.float32))
    L.fechar()
    return sp.vstack(Lg).tocsr(), np.vstack(Sw)


# ------------------------------------------------------------------ coortes
def coorte_dharsini():
    with h5py.File(H5_DHARSINI, "r") as f:
        obs = read_elem(f["obs"])
    for c in obs.columns:
        if hasattr(obs[c], "cat"):
            obs[c] = obs[c].astype(str)
    y = obs["Cell_Type_Num"].values.astype(int) - 1
    return obs, y, obs["Donor"].astype(str).values


def coorte_pan():
    meta = pd.read_csv(META_PAN)
    with h5py.File(H5_PAN_LOG, "r") as f:
        assert np.array_equal(read_elem(f["obs"]).index.astype(str).values, meta["Barcode"].astype(str).values)
    return meta, meta["Cell_Type_Num"].values.astype(int) - 1, meta["Donor"].astype(str).values


def split_treino_dharsini(obs, y, don, ok=None, val_por_grupo=None):
    """Validacao = doadores separados, estratificados por ADNC. `ok` = mascara das celulas elegiveis
    (quimica, sorting, doadores excluidos). Candidatos a validacao: doadores com All Nuclei (tem glia)."""
    ok = np.ones(len(y), bool) if ok is None else ok
    sub = obs[ok]
    grupos = sub.groupby("Donor")["Disease.Group"].first().astype(str)
    com_all = set(sub.loc[sub["SORT"] == "All Nuclei", "Donor"])
    for s in range(42, 242):
        rng = np.random.default_rng(s); val = []
        for _, ds in grupos.groupby(grupos):
            pool = [x for x in ds.index.values if x in com_all] or list(ds.index.values)
            k = val_por_grupo if val_por_grupo else max(1, int(round(CFG["frac_val_doadores"] * len(ds))))
            val += list(rng.choice(pool, min(k, len(pool)), replace=False))
        mv = np.isin(don, val) & ok & (y < 6)
        if (np.bincount(y[mv], minlength=6)[:6] >= CFG["min_val_por_classe"]).all():
            break
    else:
        raise RuntimeError("sem particao valida de doadores")
    est = np.char.add(np.char.add(don.astype(str), "|"), y.astype(str))     # doador x classe (sortings somados)
    i_tr = amostrar(est, ~np.isin(don, val) & ok & (y < 6), CFG["cap_treino"], 1)
    i_va = amostrar(est, np.isin(don, val) & ok & (y < 6), CFG["cap_val"], 2)
    return i_tr, i_va, {"doadores_val": sorted(val), "seed_particao": s,
                        "doadores_treino": sorted(set(don[i_tr])), "n_doadores_treino": len(set(don[i_tr]))}


def split_treino_pan(y):
    z = np.load(CAM_SPLIT_PAN)
    i_tr = np.sort(z["idx_treino"]); i_va = np.sort(z["idx_val"])
    return i_tr[y[i_tr] < 6], i_va[y[i_va] < 6], {"split": CAM_SPLIT_PAN, "teste_cego_pan_usado": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direcao", required=True, choices=["dharsini_para_pan", "pan_para_dharsini"])
    ap.add_argument("--dims", nargs="+", type=int, default=[600])
    ap.add_argument("--base_npy", default=None, help="base aleatoria (61541 x D) para d > 600")
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--treino_assay", default=None, help="so esta quimica no treino Dharsini (ex.: 10X-Chr-V3)")
    ap.add_argument("--neun_so_neuronios", action="store_true", help="do NeuN+ entram so Exc/Inh (glia residual fora)")
    ap.add_argument("--excluir_doadores", nargs="*", default=[], help="doadores do Dharsini fora do treino/validacao")
    ap.add_argument("--val_por_grupo", type=int, default=None, help="n de doadores de validacao por grupo ADNC")
    ap.add_argument("--sem_l2", action="store_true")
    ap.add_argument("--rotulo", default="")
    a = ap.parse_args()
    if a.rapido:
        CFG.update({"cap_treino": 60, "cap_val": 40, "cap_teste_geral": 30, "seeds": [42], "epocas_max": 3,
                    "paciencia": 2, "min_val_por_classe": 5})
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"{SAIDA_RAIZ}/etapa7_{a.direcao}_{a.rotulo + '_' if a.rotulo else ''}{'RAPIDO_' if a.rapido else ''}{ts}"
    os.makedirs(f"{out}/predicoes", exist_ok=False)
    log(f"Saida: {out}")

    W_of = carregar_W_oficial()
    W_al = np.load(a.base_npy) if a.base_npy else None
    if max(a.dims) > 600:
        assert W_al is not None and W_al.shape[1] >= max(a.dims), "d > 600 exige --base_npy com colunas suficientes"
        assert np.abs(W_al[:, :50].T @ W_al[:, :50] - np.eye(50)).max() < 1e-4

    obs_d, y_d, don_d = coorte_dharsini()
    meta_p, y_p, don_p = coorte_pan()
    if a.direcao == "dharsini_para_pan":
        h5_tr, h5_te = H5_DHARSINI, H5_PAN_LOG
        ok = np.ones(len(y_d), bool)
        if a.treino_assay:
            ok &= obs_d["Assay"].values == a.treino_assay
        if a.neun_so_neuronios:
            ok &= ~((obs_d["SORT"].values == "NeuN") & ~np.isin(y_d, [2, 3]))
        if a.excluir_doadores:
            ok &= ~np.isin(don_d, a.excluir_doadores)
        i_tr, i_va, info_split = split_treino_dharsini(obs_d, y_d, don_d, ok, a.val_por_grupo)
        info_split.update({"treino_assay": a.treino_assay, "neun_so_neuronios": a.neun_so_neuronios,
                           "excluir_doadores": a.excluir_doadores,
                           "treino_por_classe_x_sort": pd.crosstab(pd.Series([NOMES[c] for c in y_d[i_tr]], name="classe"),
                                                                   pd.Series(obs_d["SORT"].values[i_tr], name="sort")).to_dict()})
        assert not set(don_d[i_tr]) & set(don_d[i_va]), "doador em treino e validacao"
        y_tr_all, don_tr_all = y_d, don_d
        i_te = np.arange(len(meta_p)); y_te_all, don_te_all = y_p, don_p
        escopos_te = {"Pan_todos": np.ones(len(i_te), bool), "Pan_sem_ALSP": don_p != "ALSP"}
    else:
        h5_tr, h5_te = H5_PAN_LOG, H5_DHARSINI
        i_tr, i_va, info_split = split_treino_pan(y_p)
        y_tr_all, don_tr_all = y_p, don_p
        homo = obs_d["Homologo_10XV3_AllNuclei"].values.astype(bool)
        est = (obs_d["Donor"] + "|" + obs_d["Assay"] + "|" + obs_d["SORT"]).values
        i_geral = amostrar(est, ~homo, CFG["cap_teste_geral"], 3)
        i_te = np.sort(np.concatenate([np.where(homo)[0], i_geral]))
        y_te_all, don_te_all = y_d, don_d
        o = obs_d.iloc[i_te]
        escopos_te = {"Homologo": o["Homologo_10XV3_AllNuclei"].values.astype(bool),
                      "Geral_amostrado": np.ones(len(i_te), bool),
                      "Geral_sem111": o["Donor"].values != "111"}
        for v in sorted((o["Assay"] + " | " + o["SORT"]).unique()):
            escopos_te[f"Assay|SORT={v}"] = (o["Assay"] + " | " + o["SORT"]).values == v
        for v in sorted(o["Disease.Group"].unique()):
            escopos_te[f"ADNC={v}"] = o["Disease.Group"].values == v
    salvar_json_novo({"cfg": CFG, "direcao": a.direcao, "dims": a.dims, "base_npy": a.base_npy, **info_split,
                      "n_treino": len(i_tr), "n_val": len(i_va), "n_teste": len(i_te), "W_oficial_sha256": W_SHA,
                      "inicio": ts, "regra_escolha_melhor": "maior F1-macro medio (3 seeds) na VALIDACAO interna; o teste nao entra"},
                     f"{out}/config.json")
    np.savez(f"{out}/indices_treino_val_teste.npz", i_tr=i_tr, i_va=i_va, i_te=i_te)
    log(f"{a.direcao}: treino {len(i_tr):,} | val {len(i_va):,} | teste {len(i_te):,} | {info_split.get('doadores_val')}")

    Wfull = W_of if max(a.dims) <= 600 else W_al
    def proj(h5, idx):
        Lg, S = ler(h5, idx, Wfull)
        S6 = S if Wfull is W_of else None
        return Lg, S
    Lg_tr, S_tr = proj(h5_tr, i_tr); Lg_va, S_va = proj(h5_tr, i_va); Lg_te, S_te = proj(h5_te, i_te)
    # se Wfull for a base aleatoria, d <= 600 ainda deve usar a W OFICIAL -> projeta tambem com ela
    if Wfull is not W_of and any(d <= 600 for d in a.dims):
        def so_oficial(h5, idx):
            return ler(h5, idx, W_of)[1]
        O_tr, O_va, O_te = so_oficial(h5_tr, i_tr), so_oficial(h5_tr, i_va), so_oficial(h5_te, i_te)
    else:
        O_tr, O_va, O_te = S_tr, S_va, S_te
    ytr, yva, yte = y_tr_all[i_tr], y_tr_all[i_va], y_te_all[i_te]
    dte = don_te_all[i_te]
    log("Leitura e projecao concluidas")

    import anndata as ad, scanpy as sc
    a_tr = ad.AnnData(Lg_tr); sc.pp.highly_variable_genes(a_tr, n_top_genes=CFG["n_hvg"], flavor="seurat")
    cols = np.where(a_tr.var["highly_variable"].values)[0]; del a_tr
    np.save(f"{out}/hvg_cols_treino.npy", cols)
    H = lambda X: X[:, cols].toarray().astype(np.float32)
    pca = PCA(n_components=CFG["k_pca_hvg"], random_state=42).fit(H(Lg_tr))
    reps = {"HVG2000_PCA50": (pca.transform(H(Lg_tr)), pca.transform(H(Lg_va)), pca.transform(H(Lg_te)))}
    for d in a.dims:
        src = (O_tr, O_va, O_te) if d <= 600 else (S_tr, S_va, S_te)
        tag = "oficial" if d <= 600 else "aleat_s1"
        reps[f"SWeeP_{d}D_{tag}_bruto"] = tuple(Z[:, :d] for Z in src)
        if not a.sem_l2:
            reps[f"SWeeP_{d}D_{tag}_L2"] = tuple(normalize(Z[:, :d]) for Z in src)
    del Lg_tr, Lg_va

    linhas, porc, pdoa, conf_boot, todas = [], [], [], {}, {}
    for nome, (Ztr, Zva, Zte) in reps.items():
        sc_ = StandardScaler().fit(Ztr)
        Ztr, Zva, Zte = (sc_.transform(Z).astype(np.float32) for Z in (Ztr, Zva, Zte))
        for s in CFG["seeds"]:
            m, info = treinar(Ztr, ytr, Zva, yva, s)
            p_va, p = prever(m, Zva), prever(m, Zte)
            np.save(f"{out}/predicoes/ypred_{nome}_seed{s}.npy", p)
            todas.setdefault(nome, []).append(p)
            base = {"Representacao": nome, "Seed": s, "F1_macro_val_treino": f1_macro(confusao(yva, p_va)), **info}
            for esc, me in escopos_te.items():
                av = me & (yte < 6)
                if av.sum() < 50:
                    continue
                linhas.append({**base, "Escopo_teste": esc, "N": int(av.sum()),
                               "N_doadores": int(len(np.unique(dte[av]))),
                               "F1_macro_teste": f1_macro(confusao(yte[av], p[av]))})
            principal = list(escopos_te)[0]
            log(f"  {nome} seed={s}: val={base['F1_macro_val_treino']:.4f} | teste {principal}="
                f"{[l for l in linhas if l['Representacao']==nome and l['Seed']==s and l['Escopo_teste']==principal][0]['F1_macro_teste']:.4f}")
            if s == 42:
                av = yte < 6
                porc += [{"Representacao": nome, "Classe": c, "F1": v}
                         for c, v in zip(NOMES[:6], f1_de_confusao(confusao(yte[av], p[av])))]
                conf_boot[nome] = {esc: confusoes_por_doador(yte[me & av], p[me & av], dte[me & av])
                                   for esc, me in escopos_te.items() if (me & av).sum() >= 50}
                for d in np.unique(dte[av]):
                    md = av & (dte == d)
                    pdoa.append({"Representacao": nome, "Doador": d, "N": int(md.sum()),
                                 "F1_macro": f1_macro(confusao(yte[md], p[md]))})
    tab = pd.DataFrame(linhas); tab.to_csv(f"{out}/T7_F1_por_escopo.csv", index=False)
    pd.DataFrame(porc).to_csv(f"{out}/T7_F1_por_classe_seed42.csv", index=False)
    pd.DataFrame(pdoa).to_csv(f"{out}/T7_F1_por_doador_seed42.csv", index=False)

    # IC por bootstrap de doadores de TESTE (pareado entre representacoes), seed 42
    ics = []
    nomes = list(conf_boot)
    for esc in escopos_te:
        if not all(esc in conf_boot[n] for n in nomes):
            continue
        bt = bootstrap_doadores({n: conf_boot[n][esc] for n in nomes}, B=1000 if not a.rapido else 100)
        for n in nomes:
            lo, hi = ic(bt[n]); ics.append({"Escopo_teste": esc, "Comparacao": n,
                                            "F1": f1_macro(sum(conf_boot[n][esc].values())), "IC95_inf": lo, "IC95_sup": hi})
            if n != "HVG2000_PCA50":
                dl = bt["HVG2000_PCA50"] - bt[n]; lo, hi = ic(dl)
                ics.append({"Escopo_teste": esc, "Comparacao": f"HVG - {n}",
                            "F1": f1_macro(sum(conf_boot["HVG2000_PCA50"][esc].values())) - f1_macro(sum(conf_boot[n][esc].values())),
                            "IC95_inf": lo, "IC95_sup": hi, "Frac_boot_HVG_melhor": float((dl > 0).mean())})
    pd.DataFrame(ics).to_csv(f"{out}/T7_IC_bootstrap_doadores_seed42.csv", index=False)
    res = tab.groupby(["Escopo_teste", "Representacao"])["F1_macro_teste"].agg(["mean", "std"]).reset_index()
    res.to_csv(f"{out}/T7_resumo.csv", index=False)

    # F1 por classe x doador de teste (confusao somada das seeds)
    cxd = []
    for nome, ps in todas.items():
        for dd in np.unique(dte):
            md = (dte == dd) & (yte < 6)
            M = sum(confusao(yte[md], p[md]) for p in ps)
            sup = M.sum(1)
            for c in range(6):
                f = f1_de_confusao(M, [c])[0] if sup[c] > 0 else np.nan
                cxd.append({"Representacao": nome, "Doador": dd, "Classe": NOMES[c], "N_nucleos": int(sup[c]), "F1": f})
            cxd.append({"Representacao": nome, "Doador": dd, "Classe": "F1-macro", "N_nucleos": int(sup.sum()),
                        "F1": f1_macro(M)})
    pd.DataFrame(cxd).to_csv(f"{out}/T7_F1_classe_x_doador.csv", index=False)

    # escolha do melhor modelo SO pela validacao interna
    val = tab.drop_duplicates(["Representacao", "Seed"]).groupby("Representacao")["F1_macro_val_treino"].agg(["mean", "std"])
    melhor = val["mean"].idxmax()
    with open(f"{out}/escolha_melhor_modelo_pela_validacao.json", "w") as fh:
        json.dump({"criterio": "maior F1-macro medio de validacao interna (3 seeds)", "melhor": melhor,
                   "validacao": val.round(5).reset_index().to_dict("records")}, fh, indent=2)
    log(f"Melhor pela validacao interna: {melhor}\n" + val.round(4).to_string())
    log("Resumo F1-macro teste (media das seeds):\n" +
        res.pivot(index="Representacao", columns="Escopo_teste", values="mean").round(4).to_string())


if __name__ == "__main__":
    main()
