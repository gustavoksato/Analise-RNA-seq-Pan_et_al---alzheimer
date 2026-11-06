#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETAPA 6a - DIAGNOSTICO (READ-ONLY, CEGO) DA COORTE EXTERNA GSE263468 ANTES DA INFERENCIA

NAO carrega modelos, NAO faz predicoes, NAO olha desempenho. So verifica se o material exportado pelo script de
preparacao (matriz .mtx binarizada + Rotulos + Metadados) e confiavel e se o desenho da avaliacao e defensavel.

Verificacoes
  A. Genes: IDs Ensembl da coorte (coluna 'ENSEMBLE GENE') unicos, mesmo CONJUNTO do Pan e permutacao necessaria
     contra o ARQUIVO DE ALINHAMENTO CANONICO da Etapa 5c (genes_pan_61541_ensembl_sem_versao.txt).
  B. Matriz exportada x counts do h5ad (prova de ordem): amostra de celulas da .mtx comparada com
     (counts>0) do h5ad reordenado ao canonico. Se a mtx estiver na ordem ORIGINAL do h5ad, o script diz.
  C. Rotulos/Metadados: mesmas linhas da mtx; mapa de classes recomputado e comparado com o exportado.
  D. Suporte por classe (total, por Assay x SORT e por doador): classes com pouco suporte inviabilizam F1 por classe.
  E. Independencia: doadores em comum com o Pan (por ID) e busca de possiveis MESMAS pessoas com IDs diferentes
     (sexo+idade), alem de amostras (Sample) em comum quando existirem.
  F. Pacote 1 (Homologo) contido no Pacote 2 (Geral): nao sao avaliacoes independentes.
  Congela o mapa de classes (JSON) e, opcionalmente, o SHA-256 dos arquivos exportados.

Uso:
    python etapa6a_diagnostico_gse263468.py --externo-h5ad GSE263468_processed_data.h5ad \
        --export-dir export_GSE263468_SWeeP
    (opcional) --hash  --script-preparo dharsini_total.py  --max-linhas-mtx 30000  --n-amostra 300
"""

import os
import re
import sys
import json
import hashlib
import argparse
import datetime

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem

BASE = "/media/gustavo-k-sato/D61CF48E1CF46B3F1/analisesPAN"
DEFAULT_RUN = os.path.join(BASE, "resultados_MLP_GSE267301_v2", "run_20261102_201556")
CAM_PAN_META = os.path.join(BASE, "Metadados_Pan_Limpos.csv")
PREFIXOS = ["GSE263468_Homologo_10XV3_AllNuclei", "GSE263468_Geral_SemVazamento"]
RE_ENS = re.compile(r"^(ENS[A-Z]{0,6}G\d{11})(?:\.\d+)?$")
LIM_CLASSE, LIM_DOADOR_CLASSE = 30, 20
SEED = 2026
NOMES = {1: "Astrocytes", 2: "Oligodendrocytes", 3: "Exc. Neurons", 4: "Inh. Neurons",
         5: "OPCs", 6: "Microglia", 7: "Endothelial", 8: "VLMCs"}

RESULT = {}          # veredito de cada verificacao


def norm_id(s):
    s = str(s).strip().upper()
    m = RE_ENS.match(s)
    return m.group(1) if m else s


def sha256_arq(c, bloco=1 << 24):
    h = hashlib.sha256()
    with open(c, "rb") as f:
        for b in iter(lambda: f.read(bloco), b""):
            h.update(b)
    return h.hexdigest()


def mapear_8_classes(obs_df):
    """COPIA FIEL da regra do script de preparacao (para recomputar e comparar)."""
    major = obs_df["Major_celltypes"].astype(str)
    auth = obs_df["author_annotation"].astype(str)
    mapa = {"Astrocytes": 1, "Oligodendrocytes": 2, "Excitatory": 3, "Inhibitory": 4, "OPC": 5, "Microglia": 6}
    r = major.map(mapa)
    vasc = major == "Vascular"
    r = np.where(vasc & (auth == "Endothelial"), 7, r)
    r = np.where(vasc & (auth == "VLMC"), 8, r)
    return pd.Series(r, index=obs_df.index, dtype="float64")


def veredito(nome, ok, detalhe=""):
    RESULT[nome] = bool(ok)
    print(f"   >>> {nome}: {'OK' if ok else 'FALHA/ATENCAO'} {detalhe}")


# ------------------------------------------------------------------------------ A
def analisar_genes(var, col, canonico):
    if col not in var.columns:
        cands = [c for c in var.columns if var[c].astype(str).map(lambda x: bool(RE_ENS.match(x.strip().upper()))).mean() > 0.99]
        raise KeyError(f"Coluna '{col}' ausente no var. Candidatas 100% Ensembl: {cands}")
    ext = np.array([norm_id(x) for x in var[col].astype(str).values])
    info = {"n_genes_ext": int(len(ext)), "unicos": bool(len(set(ext)) == len(ext)),
            "n_canonico": int(len(canonico)), "mesmo_conjunto": bool(set(ext) == set(canonico)),
            "n_comuns": int(len(set(ext) & set(canonico)))}
    perm = None
    if info["unicos"] and info["mesmo_conjunto"] and len(ext) == len(canonico):
        pos = {g: i for i, g in enumerate(ext)}
        perm = np.array([pos[g] for g in canonico], dtype=np.int64)       # perm[k] = coluna externa do gene canonico k
        info["ja_na_ordem_canonica"] = bool(np.array_equal(perm, np.arange(len(perm))))
        info["n_posicoes_diferentes_da_ordem_externa"] = int(np.sum(perm != np.arange(len(perm))))
    return info, perm


# ------------------------------------------------------------------------------ B
def esperado_linha(grupo, i, inv):
    p0, p1 = int(grupo["indptr"][i]), int(grupo["indptr"][i + 1])
    idx = grupo["indices"][p0:p1]
    dat = grupo["data"][p0:p1]
    return set(inv[idx[dat > 0]].tolist())


def checar_mtx(cam_mtx, meta, obs, grupo, perm, n_amostra, max_linhas, rng):
    with open(cam_mtx) as f:
        cab1, cab2 = f.readline(), f.readline()
    n_lin, n_col, nnz = (int(x) for x in cab2.split())
    out = {"cabecalho": cab1.strip(), "n_linhas_mtx": n_lin, "n_colunas_mtx": n_col, "nnz_cabecalho": nnz,
           "n_linhas_metadados": int(len(meta))}
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    n_cand = min(max_linhas, n_lin)
    amostra0 = np.sort(rng.choice(n_cand, size=min(n_amostra, n_cand), replace=False))
    idx_obs = obs.index.get_indexer(meta.index[amostra0])
    if (idx_obs < 0).any():
        raise RuntimeError("Barcodes do Metadados ausentes no h5ad externo.")
    esp_canon = {int(r): esperado_linha(grupo, int(i), inv) for r, i in zip(amostra0, idx_obs)}
    esp_orig = {int(r): set(grupo["indices"][int(grupo["indptr"][i]):int(grupo["indptr"][i + 1])]
                            [grupo["data"][int(grupo["indptr"][i]):int(grupo["indptr"][i + 1])] > 0].tolist())
                for r, i in zip(amostra0, idx_obs)}
    alvo = set((amostra0 + 1).tolist())
    obtido = {r: [] for r in alvo}
    nao_binario = 0
    for ch in pd.read_csv(cam_mtx, sep=" ", header=None, skiprows=2, names=["r", "c", "v"],
                          dtype={"r": np.int64, "c": np.int64, "v": np.int64}, chunksize=5_000_000):
        m = np.isin(ch["r"].values, list(alvo))
        if m.any():
            sub = ch[m]
            nao_binario += int((sub["v"] != 1).sum())
            for r, g in sub.groupby("r"):
                obtido[int(r)].append(g["c"].values - 1)
        if ch["r"].values[-1] > amostra0.max() + 1:
            break
    iguais_canon = iguais_orig = vazias = 0
    for r0 in amostra0:
        got = np.concatenate(obtido[int(r0) + 1]) if obtido[int(r0) + 1] else np.array([], dtype=np.int64)
        got = set(got.tolist())
        vazias += int(len(got) == 0)
        iguais_canon += int(got == esp_canon[int(r0)])
        iguais_orig += int(got == esp_orig[int(r0)])
    n = len(amostra0)
    out.update({"n_linhas_amostradas": int(n), "linhas_iguais_ordem_CANONICA": iguais_canon,
                "linhas_iguais_ordem_ORIGINAL_h5ad": iguais_orig, "linhas_vazias_na_mtx": vazias,
                "valores_nao_binarios_na_amostra": nao_binario})
    return out


# ------------------------------------------------------------------------------ E
def tabela_doador(df, col_doador):
    cols = [c for c in df.columns if re.search(r"sex|gender|^age|diag|disease|braak|case|region|sample|assay|sort", str(c), re.I)
            and c != col_doador]
    linhas = []
    for d, g in df.groupby(col_doador):
        linha = {"Doador": str(d), "N_Celulas": len(g)}
        for c in cols:
            linha[c] = ";".join(sorted(map(str, g[c].dropna().unique()))[:6])
        linhas.append(linha)
    return pd.DataFrame(linhas), cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--externo-h5ad", required=True)
    ap.add_argument("--export-dir", required=True, help="Pasta export_GSE263468_SWeeP (Matriz_/Rotulos_/Metadados_)")
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--alinhamento", default=None, help="genes_pan_61541_ensembl_sem_versao.txt (Etapa 5c)")
    ap.add_argument("--pan-metadados", default=CAM_PAN_META)
    ap.add_argument("--col-genes-ext", default="ENSEMBLE GENE")
    ap.add_argument("--prefixos", nargs="*", default=PREFIXOS)
    ap.add_argument("--n-amostra", type=int, default=300)
    ap.add_argument("--max-linhas-mtx", type=int, default=30000)
    ap.add_argument("--hash", action="store_true", help="SHA-256 dos arquivos exportados (lento)")
    ap.add_argument("--script-preparo", default=None, help="Script de preparacao (para registrar o hash)")
    args = ap.parse_args()

    run = os.path.abspath(args.run)
    cam_al = args.alinhamento or os.path.join(run, "etapa5_modelos_finais_v2", "etapa5c_chave_ensembl",
                                              "genes_pan_61541_ensembl_sem_versao.txt")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    saida = os.path.join(run, "etapa6_gse263468", f"etapa6a_diagnostico_{ts}")
    os.makedirs(saida, exist_ok=False)
    rng = np.random.default_rng(SEED)
    pd.set_option("display.width", 220, "display.max_columns", 30)

    print("=" * 78)
    print("ETAPA 6a - DIAGNOSTICO CEGO DA COORTE EXTERNA GSE263468")
    print("=" * 78)
    with open(cam_al, encoding="utf-8") as f:
        canonico = [l.strip() for l in f if l.strip()]
    assert len(canonico) == 61541 and len(set(canonico)) == 61541, "Alinhamento canonico invalido"

    f_ext = h5py.File(args.externo_h5ad, "r")
    obs, var = read_elem(f_ext["obs"]), read_elem(f_ext["var"])
    grupo = f_ext["layers"]["counts"]
    print(f"[0] h5ad externo: {len(obs):,} celulas x {len(var):,} genes | colunas obs: {list(obs.columns)[:25]}")
    assert obs.index.is_unique, "Barcodes duplicados no h5ad externo"

    # ------------------------------------------------------------ A
    print("\n[A] Genes x alinhamento canonico (Etapa 5c)")
    info_g, perm = analisar_genes(var, args.col_genes_ext, canonico)
    for k, v in info_g.items():
        print(f"   {k}: {v}")
    veredito("A_genes_mesmo_conjunto_e_unicos", info_g["unicos"] and info_g["mesmo_conjunto"])
    if perm is None:
        raise SystemExit("Sem permutacao valida: nao ha como alinhar esta coorte ao Pan. Veja o relatorio.")

    # ------------------------------------------------------------ B/C/D por pacote
    resultados_pacote, tabs = {}, {}
    barcodes = {}
    for pref in args.prefixos:
        cam_mtx = os.path.join(args.export_dir, f"Matriz_{pref}_Binarizada.mtx")
        cam_rot = os.path.join(args.export_dir, f"Rotulos_{pref}.csv")
        cam_met = os.path.join(args.export_dir, f"Metadados_{pref}.csv")
        if not (os.path.exists(cam_rot) and os.path.exists(cam_met)):
            print(f"\n[pacote {pref}] arquivos ausentes, pulado.")
            continue
        print(f"\n================ PACOTE: {pref} ================")
        meta = pd.read_csv(cam_met, index_col=0)
        rot = pd.read_csv(cam_rot)
        barcodes[pref] = set(meta.index)
        print(f"[C] Rotulos/Metadados: {len(rot):,} / {len(meta):,} linhas")
        ok_n = len(rot) == len(meta)
        ok_lab = bool(ok_n and np.array_equal(rot["Cell_Type_Num"].values, meta["Cell_Type_Num"].values))
        veredito(f"C_rotulos_igual_metadados[{pref}]", ok_lab)
        ix = obs.index.get_indexer(meta.index)
        ok_bar = bool((ix >= 0).all())
        veredito(f"C_barcodes_no_h5ad[{pref}]", ok_bar)
        if ok_bar:
            recomputado = mapear_8_classes(obs.iloc[ix]).values
            ok_map = bool(np.array_equal(recomputado, meta["Cell_Type_Num"].values.astype(float)))
            veredito(f"C_mapa_de_classes_reprodutivel[{pref}]", ok_map)
            ct = pd.crosstab(obs.iloc[ix]["Major_celltypes"].astype(str).values,
                             obs.iloc[ix]["author_annotation"].astype(str).values)
            ct.to_csv(os.path.join(saida, f"C_crosstab_Major_x_author_{pref}.csv"))

        if os.path.exists(cam_mtx) and ok_bar:
            print(f"[B] Prova de ordem da .mtx (amostra de {args.n_amostra} celulas das primeiras "
                  f"{args.max_linhas_mtx:,} linhas)")
            rb = checar_mtx(cam_mtx, meta, obs, grupo, perm, args.n_amostra, args.max_linhas_mtx, rng)
            for k, v in rb.items():
                print(f"   {k}: {v}")
            ok_dim = rb["n_linhas_mtx"] == len(meta) and rb["n_colunas_mtx"] == 61541
            ok_ord = rb["linhas_iguais_ordem_CANONICA"] == rb["n_linhas_amostradas"] and rb["valores_nao_binarios_na_amostra"] == 0
            veredito(f"B_dimensoes_mtx[{pref}]", ok_dim)
            veredito(f"B_mtx_na_ordem_CANONICA[{pref}]", ok_ord,
                     "" if ok_ord else ("| a mtx esta na ordem ORIGINAL do h5ad (nao permutada)!"
                                        if rb["linhas_iguais_ordem_ORIGINAL_h5ad"] == rb["n_linhas_amostradas"]
                                        else "| nem canonica nem original: investigar"))
            resultados_pacote[pref] = rb

        # ---- D: suporte
        meta["Cell_Type_Num"] = meta["Cell_Type_Num"].astype(int)
        tot = meta["Cell_Type_Num"].value_counts().reindex(range(1, 9), fill_value=0)
        dc = pd.crosstab(meta["Donor"].astype(str), meta["Cell_Type_Num"]).reindex(columns=range(1, 9), fill_value=0)
        dc.columns = [NOMES[c] for c in dc.columns]
        dc.to_csv(os.path.join(saida, f"D_doador_x_classe_{pref}.csv"))
        tab = pd.DataFrame({"Classe": [NOMES[c] for c in range(1, 9)], "N": tot.values,
                            "Doadores_com_>=%d" % LIM_DOADOR_CLASSE: [(dc[NOMES[c]] >= LIM_DOADOR_CLASSE).sum() for c in range(1, 9)],
                            "Abaixo_do_limiar_(%d)" % LIM_CLASSE: tot.values < LIM_CLASSE})
        print("[D] Suporte por classe (n celulas; n doadores com >=%d celulas):" % LIM_DOADOR_CLASSE)
        print(tab.to_string(index=False))
        tab.to_csv(os.path.join(saida, f"D_suporte_por_classe_{pref}.csv"), index=False)
        for c in ["Assay", "SORT"]:
            if c in meta.columns:
                pd.crosstab(meta[c].astype(str), meta["Cell_Type_Num"].map(NOMES)).to_csv(
                    os.path.join(saida, f"D_{c}_x_classe_{pref}.csv"))
        if {"Assay", "SORT"} <= set(meta.columns):
            print("   Assay x SORT (n celulas):")
            print(meta.groupby(["Assay", "SORT"]).size().to_string())
        por_doador = dc.sum(axis=1).to_numpy()
        print(f"   Doadores: {meta['Donor'].astype(str).nunique()} | celulas por doador (min/mediana/max): "
              f"{int(por_doador.min())}/{int(np.median(por_doador))}/{int(por_doador.max())}")
        tabs[pref] = tab

    if len(barcodes) == 2:
        a, b = (barcodes[p] for p in args.prefixos[:2])
        print(f"\n[F] Pacote 1 contido no Pacote 2: {a <= b} ({len(a & b):,} de {len(a):,} barcodes)")
        veredito("F_pacote1_subconjunto_do_pacote2", a <= b, "(logo, NAO sao avaliacoes independentes)")

    # ------------------------------------------------------------ E
    print("\n[E] Independencia de doadores")
    pan = pd.read_csv(args.pan_metadados)
    obs["Donor_str"] = obs["Donor"].astype(str).str.strip()
    pan_d = sorted(pan["Donor"].astype(str).str.strip().unique())
    ext_d = sorted(obs["Donor_str"].unique())
    comum = sorted(set(pan_d) & set(ext_d))
    print(f"   Pan ({len(pan_d)}): {pan_d}\n   Externo ({len(ext_d)} doadores)\n   Em comum por ID ({len(comum)}): {comum}")
    print(f"   Pan SEM par por ID no externo: {sorted(set(pan_d) - set(ext_d))}")
    t_pan, c_pan = tabela_doador(pan.assign(Donor=pan["Donor"].astype(str)), "Donor")
    t_ext, c_ext = tabela_doador(obs, "Donor_str")
    t_pan.to_csv(os.path.join(saida, "E_doadores_Pan.csv"), index=False)
    t_ext.to_csv(os.path.join(saida, "E_doadores_externo.csv"), index=False)
    print("   Atributos dos doadores em comum (Pan | Externo):")
    cmp_ = t_pan[t_pan.Doador.isin(comum)].merge(t_ext[t_ext.Doador.isin(comum)], on="Doador", suffixes=("_Pan", "_Ext"))
    print(cmp_.to_string(index=False))
    cmp_.to_csv(os.path.join(saida, "E_doadores_em_comum_Pan_vs_Ext.csv"), index=False)

    def achar(cols, pad):
        r = [c for c in cols if re.search(pad, str(c), re.I)]
        return r[0] if r else None
    s_p, a_p = achar(c_pan, r"^sex|gender"), achar(c_pan, r"^age")
    s_e, a_e = achar(c_ext, r"^sex|gender"), achar(c_ext, r"^age")
    suspeitos = []
    if all([s_p, a_p, s_e, a_e]):
        for _, re_ in t_ext[~t_ext.Doador.isin(comum)].iterrows():
            for _, rp in t_pan.iterrows():
                if re_[s_e] == rp[s_p] and re_[a_e] == rp[a_p]:
                    suspeitos.append({"Doador_externo": re_["Doador"], "Doador_Pan": rp["Doador"],
                                      "Sexo": rp[s_p], "Idade": rp[a_p]})
        pd.DataFrame(suspeitos).to_csv(os.path.join(saida, "E_possiveis_mesmas_pessoas_ID_diferente.csv"), index=False)
        print(f"   Possiveis mesmas pessoas com ID diferente (sexo+idade iguais): {len(suspeitos)}")
        if suspeitos:
            print(pd.DataFrame(suspeitos).to_string(index=False))
        veredito("E_sem_suspeitos_sexo_idade", len(suspeitos) == 0, "(sexo+idade e criterio fraco: confirmar por diagnostico/regiao)")
    else:
        print(f"   [AVISO] colunas sexo/idade nao encontradas (Pan: {s_p},{a_p} | Ext: {s_e},{a_e}); busca por IDs diferentes NAO feita.")
        veredito("E_busca_sexo_idade_realizada", False)
    sa_p = achar(pan.columns, r"^sample"); sa_e = achar(obs.columns, r"^sample")
    if sa_p and sa_e:
        comuns_s = sorted(set(pan[sa_p].astype(str)) & set(obs[sa_e].astype(str)))
        print(f"   Amostras (Sample) em comum: {len(comuns_s)} {comuns_s[:12]}")
        pd.Series(comuns_s, name="Sample").to_csv(os.path.join(saida, "E_samples_em_comum.csv"), index=False)

    # ------------------------------------------------------------ congelamento
    mapa = {"regra": "Major_celltypes -> {Astrocytes:1, Oligodendrocytes:2, Excitatory:3, Inhibitory:4, OPC:5, "
                     "Microglia:6}; Vascular + author_annotation=='Endothelial' -> 7; Vascular + 'VLMC' -> 8; "
                     "demais (inclui Pericytes e outros Vascular) -> excluidos.",
            "nomes_classes_modelo": NOMES}
    prov = {"gerado_em": datetime.datetime.now().isoformat(timespec="seconds"), "externo_h5ad": args.externo_h5ad,
            "alinhamento_canonico": cam_al, "mapa_de_classes": mapa, "genes": info_g,
            "mtx_checagem": resultados_pacote, "vereditos": RESULT}
    if args.script_preparo and os.path.exists(args.script_preparo):
        prov["sha256_script_preparo"] = sha256_arq(args.script_preparo)
    if args.hash:
        prov["sha256_arquivos"] = {os.path.basename(p): sha256_arq(p) for p in
                                   [os.path.join(args.export_dir, x) for x in sorted(os.listdir(args.export_dir))]
                                   if os.path.isfile(p)}
    with open(os.path.join(saida, "provenance_e_vereditos.json"), "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2, ensure_ascii=False, default=str)
    f_ext.close()

    print("\n" + "=" * 78)
    print("RESUMO DOS VEREDITOS")
    for k, v in RESULT.items():
        print(f"   [{'OK ' if v else 'ATENCAO'}] {k}")
    print(f"Saida: {saida}")
    print("=" * 78)


if __name__ == "__main__":
    main()