#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETAPA 6b - INFERENCIA EXTERNA: modelos CONGELADOS do Pan (Etapa 5 v2) -> GSE263468 (Dharsini)

Nenhum treino, nenhum ajuste: transformadores (.npz) e MLPs (.pt) sao lidos do run travado.
Entrada: h5ad pos-QC do Dharsini (contagens brutas; 372.908 nucleos; 40 doadores; sem os 6 do Pan).
  HVG   : normalize_total(1e4) + log1p -> colunas HVG do Pan -> PCA50 -> scaler -> MLP
  SWeeP : (x>0) @ W oficial (sha 1eee02c5) -> [PCA40] -> scaler -> MLP

PRE-REGISTRO (gravado antes de qualquer predicao; ver preregistro.json)
  - Escopo primario: pacote Homologo (10X-V3 + All Nuclei); secundario: Geral (todas as quimicas).
  - Metrica primaria: F1-macro nas classes 1-6; classes 7-8 (Endothelial/VLMC) so descritivas
    (recall), pois tem ~20 celulas.
  - Modelo primario: seed 42 (canonica); seeds 100/2026 = sensibilidade.
  - IC 95%: bootstrap de DOADORES (B=2000), pareado entre bracos; contraste HVG - SWeeP.
  - Sensibilidade: sem o doador 111 (possivel mesma pessoa que o 112 do Pan).
  - Estratos: Assay x SORT, regiao, ADNC (Disease.Group), Braak, doador.

Uso: python etapa6b_inferencia_pan_para_dharsini.py [--rapido]
"""
import os, sys, json, argparse, datetime, time
import numpy as np
import pandas as pd
import torch
from anndata.io import read_elem
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comum_coortes import *  # noqa

BRACOS = {"Baseline_HVG2000_PCA50": "hvg", "SWeeP_600D_Puro": "sweep", "SWeeP_600D_PCA40": "sweep_pca"}
SEEDS = [42, 100, 2026]
SAIDA_RAIZ = f"{RUN_OFICIAL}/etapa6_gse263468"


def carregar_braco(nome, seed):
    with np.load(f"{ART}/{nome}/transformador.npz", allow_pickle=False) as z:
        A = {k: z[k] for k in z.files}
    cam = f"{ART}/{nome}/mlp_canonico.pt" if seed == 42 else f"{E5}/seeds/mlp_{nome}_seed{seed}.pt"
    est = torch.load(cam, map_location=DEVICE, weights_only=True)
    m = ClassificadorCelular(est["camadas.0.weight"].shape[1], 8).to(DEVICE); m.load_state_dict(est); m.eval()
    return A, m, cam


def transformar(A, tipo, entrada):
    if tipo == "hvg":
        Z = entrada[:, A["hvg_cols"]].toarray().astype(np.float32)
        Z = (Z - A["pca_mean"]) @ A["pca_components"].T
    elif tipo == "sweep_pca":
        Z = (entrada - A["pca_mean"]) @ A["pca_components"].T
    else:
        Z = entrada
    return ((Z - A["scaler_mean"]) / A["scaler_scale"]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rapido", action="store_true"); a = ap.parse_args()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"{SAIDA_RAIZ}/etapa6b_inferencia_{'RAPIDO_' if a.rapido else ''}{ts}"
    os.makedirs(f"{out}/predicoes", exist_ok=False); os.makedirs(f"{out}/tabelas")
    log(f"Saida: {out}")

    # ---------------- checagens de entrada ----------------
    with h5py.File(H5_DHARSINI, "r") as f:
        obs = read_elem(f["obs"]); var = read_elem(f["var"])
    genes_ext = var["ENSEMBLE GENE"].astype(str).str.split(".").str[0].str.upper().values
    genes_pan = pd.read_csv(GENES_PAN_ENSG, header=None)[0].astype(str).str.upper().values
    assert np.array_equal(genes_ext, genes_pan), "Ordem de genes do Dharsini difere da do Pan!"
    log("Genes: ordem identica aos 61.541 do Pan (Ensembl sem versao).")
    W = carregar_W_oficial()
    n = len(obs)
    linhas = np.arange(n)
    if a.rapido:
        linhas = np.sort(np.random.default_rng(0).choice(n, 6000, replace=False))

    pre = {"escopo_primario": "Homologo_10XV3_AllNuclei", "escopo_secundario": "Geral_SemVazamento",
           "classes_avaliadas": "1-6 (Astro, Oligo, Exc, Inh, OPC, Micro)", "classes_descritivas": "7-8",
           "modelo_primario": "seed 42", "sensibilidade": ["seeds 100/2026", "sem doador 111"],
           "ic": "bootstrap de doadores B=2000 pareado", "contraste": "HVG - SWeeP_600D_Puro e HVG - SWeeP_600D_PCA40",
           "entrada": H5_DHARSINI, "entrada_sha256": "nao calculado (2 GB); ver provenance",
           "W": W_RDS, "W_sha256": W_SHA, "artefatos": ART, "rapido": a.rapido, "inicio": ts}
    salvar_json_novo(pre, f"{out}/preregistro.json")

    modelos = {(b, s): carregar_braco(b, s) for b in BRACOS for s in SEEDS}
    preds = {k: np.empty(len(linhas), np.int8) for k in modelos}
    L = LeitorCSR(H5_DHARSINI); t0 = time.time()
    for i, Xc in L.blocos(linhas):
        Lg = lognorm(Xc); Sw = np.asarray(binarizar(Xc) @ W, dtype=np.float32)
        for (b, s), (A, m, _) in modelos.items():
            tipo = BRACOS[b]
            preds[(b, s)][i:i + Xc.shape[0]] = prever(m, transformar(A, tipo, Lg if tipo == "hvg" else Sw))
        log(f"  {i + Xc.shape[0]:,}/{len(linhas):,} nucleos ({time.time() - t0:.0f}s)")
    L.fechar()
    for (b, s), p in preds.items():
        np.save(f"{out}/predicoes/ypred_{b}_seed{s}.npy", p)

    cel = obs.iloc[linhas][["Donor", "Assay", "SORT", "Brain.Region", "Disease.Group", "Braak.stage",
                            "NP.Diagonis", "Major_celltypes", "author_annotation", "Cell_Type_Num",
                            "Homologo_10XV3_AllNuclei", "qc_n_genes"]].copy()
    for c in cel.columns:
        if hasattr(cel[c], "cat"):
            cel[c] = cel[c].astype(str)
    cel.to_csv(f"{out}/celulas.csv")
    avaliar(out, cel, preds, B=200 if a.rapido else 2000)


def avaliar(out, cel, preds, B=2000):
    y = cel["Cell_Type_Num"].values.astype(int) - 1
    don = cel["Donor"].astype(str).values
    escopos = {"Homologo": cel["Homologo_10XV3_AllNuclei"].values.astype(bool),
               "Geral": np.ones(len(cel), bool)}
    escopos["Homologo_sem111"] = escopos["Homologo"] & (don != "111")
    escopos["Geral_sem111"] = don != "111"
    for col in ["Assay", "SORT", "Brain.Region", "Disease.Group", "Braak.stage"]:
        for v in sorted(cel[col].astype(str).unique()):
            escopos[f"{col}={v}"] = cel[col].astype(str).values == v
    for v in sorted((cel["Assay"] + " | " + cel["SORT"]).unique()):
        escopos[f"Assay|SORT={v}"] = (cel["Assay"] + " | " + cel["SORT"]).values == v

    linhas, por_classe, desc = [], [], []
    for nome_e, m in escopos.items():
        av = m & (y < 6)
        if av.sum() < 50:
            continue
        for (b, s), p in preds.items():
            M = confusao(y[m], p[m])
            f1c = f1_de_confusao(M)
            linhas.append({"Escopo": nome_e, "Braco": b, "Seed": s, "N_celulas": int(av.sum()),
                           "N_doadores": int(len(np.unique(don[av]))), "F1_macro_1a6": float(np.nanmean(f1c)),
                           "Acuracia_1a6": float(np.trace(M[:6, :6]) / max(M[:6].sum(), 1)),
                           "Frac_pred_7ou8_em_1a6": float(M[:6, 6:].sum() / max(M[:6].sum(), 1))})
            if nome_e in ("Homologo", "Geral"):
                for c, v in zip(NOMES[:6], f1c):
                    por_classe.append({"Escopo": nome_e, "Braco": b, "Seed": s, "Classe": c, "F1": v})
                for c in (6, 7):
                    mc = m & (y == c)
                    desc.append({"Escopo": nome_e, "Braco": b, "Seed": s, "Classe": NOMES[c], "N": int(mc.sum()),
                                 "Recall": float((p[mc] == c).mean()) if mc.sum() else np.nan,
                                 "Pred_nesta_classe_total": int((p[m] == c).sum())})
    tab = pd.DataFrame(linhas); tab.to_csv(f"{out}/tabelas/T6b_F1_por_escopo.csv", index=False)
    pd.DataFrame(por_classe).to_csv(f"{out}/tabelas/T6b_F1_por_classe.csv", index=False)
    pd.DataFrame(desc).to_csv(f"{out}/tabelas/T6b_classes7e8_descritivo.csv", index=False)

    # por doador (seed 42)
    pd_l = []
    for d in np.unique(don):
        md = (don == d)
        for b in BRACOS:
            M = confusao(y[md], preds[(b, 42)][md])
            pd_l.append({"Doador": d, "Braco": b, "N": int((md & (y < 6)).sum()), "F1_macro_1a6": f1_macro(M),
                         "Assays": "|".join(sorted(cel.loc[md, "Assay"].unique())),
                         "ADNC": cel.loc[md, "Disease.Group"].iloc[0], "Braak": cel.loc[md, "Braak.stage"].iloc[0]})
    pd.DataFrame(pd_l).to_csv(f"{out}/tabelas/T6b_F1_por_doador_seed42.csv", index=False)

    # bootstrap pareado por doador (seed 42)
    res_ic = []
    for nome_e in ["Homologo", "Geral", "Homologo_sem111", "Geral_sem111"]:
        m = escopos[nome_e]
        conf = {b: confusoes_por_doador(y[m], preds[(b, 42)][m], don[m]) for b in BRACOS}
        bt = bootstrap_doadores(conf, B=B)
        for b in BRACOS:
            pt = f1_macro(sum(conf[b].values()))
            lo, hi = ic(bt[b])
            res_ic.append({"Escopo": nome_e, "Comparacao": b, "F1_ou_Delta": pt, "IC95_inf": lo, "IC95_sup": hi})
        for b in ["SWeeP_600D_Puro", "SWeeP_600D_PCA40"]:
            dlt = bt["Baseline_HVG2000_PCA50"] - bt[b]
            pt = f1_macro(sum(conf["Baseline_HVG2000_PCA50"].values())) - f1_macro(sum(conf[b].values()))
            lo, hi = ic(dlt)
            res_ic.append({"Escopo": nome_e, "Comparacao": f"HVG - {b}", "F1_ou_Delta": pt, "IC95_inf": lo,
                           "IC95_sup": hi, "Frac_boot_HVG_melhor": float((dlt > 0).mean())})
    pd.DataFrame(res_ic).to_csv(f"{out}/tabelas/T6b_IC_bootstrap_doadores.csv", index=False)
    with open(f"{out}/INFERENCIA_CONCLUIDA.json", "w") as f:
        json.dump({"fim": datetime.datetime.now().isoformat(), "n_celulas": len(cel)}, f)
    t = tab[(tab.Seed == 42) & tab.Escopo.isin(["Homologo", "Geral"])]
    log("F1-macro (1-6), seed 42:\n" + t.pivot(index="Escopo", columns="Braco", values="F1_macro_1a6").round(4).to_string())
    log("IC bootstrap por doador:\n" + pd.DataFrame(res_ic).round(4).to_string())


if __name__ == "__main__":
    main()
