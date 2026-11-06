#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETAPA 5c - CHAVE CANONICA ENSEMBL PARA INFERENCIA EXTERNA (apenas ADICIONA arquivos; nao retreina nada)

Motivo: o P1 da Etapa 5 v2 mostrou que os var_names do h5ad do Pan sao MISTOS (gene_name/simbolo em ~66% dos
genes, ID Ensembl com versao nos demais). Usar var_names como chave de mapeamento entre coortes limita a cobertura
a ~34% quando a coorte externa vem em Ensembl. Se o h5ad tem uma coluna so com Ensembl (ex.: 'index1'), ela e a
chave canonica correta: esta etapa a DETECTA, VALIDA e congela.

O que faz
  1. Le so o .var do h5ad e tabula cada coluna: fracao em formato Ensembl, unicidade (sem versao), igualdade com
     var_names e com o features_ENSG_puro.tsv (adjudica quem e quem: o molde e Ensembl ou gene_name?).
  2. Escolhe a coluna Ensembl (auto, ou --coluna-ensembl) e VALIDA: 61.541 valores, todos ENSG, unicos sem versao,
     linhas na mesma ordem dos artefatos congelados (var_names identicos aos da Etapa 5 v2).
  3. Gera o ARQUIVO DE ALINHAMENTO canonico (ordem exata das colunas do Pan / linhas da W):
        genes_pan_61541_ensembl_sem_versao.txt   <- usar no R/Python para conferir/reindexar coortes
     alem de com_versao, gene_name e os 2.000 HVGs congelados em Ensembl e gene_name (via indices, sem retreinar).
  4. Simula coortes externas com o MODELO HVG CONGELADO (colunas permutadas): Ensembl com versao, Ensembl sem
     versao + 5% de genes ausentes, IDs gene_name com chave Ensembl (limitacao esperada) e a chave antiga
     (var_names) com IDs Ensembl (mostra o problema corrigido).
  5. Emite o MANIFESTO v2.1 (herda o status da ordem W do v2) e libera o SWeeP para inferencia externa SO se
     ordem W = CONFIRMADA e os testes de chave passarem.

Politica de escrita: pasta nova etapa5c_chave_ensembl (se existir, recusa). Nada existente e alterado.

Uso:
    python etapa5c_chave_ensembl_pan.py
    python etapa5c_chave_ensembl_pan.py --run /caminho/run_AAAAMMDD_HHMMSS --coluna-ensembl index1
Requer etapa5_modelos_finais_v2_pan.py na MESMA pasta (reutiliza suas funcoes, ja testadas).
"""

import os
import sys
import json
import hashlib
import argparse
import datetime
import importlib.util

import numpy as np
import pandas as pd
import scanpy as sc

_AQUI = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("e5v2", os.path.join(_AQUI, "etapa5_modelos_finais_v2_pan.py"))
e5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e5)

SEED = 2026
HVG = "Baseline_HVG2000_PCA50"


def sha_lista(xs):
    h = hashlib.sha256()
    for s in xs:
        h.update((str(s) + "\n").encode("utf-8"))
    return h.hexdigest()


def sha_arq(cam):
    return e5.sha256_arquivo(cam)


def novo(caminho):
    if os.path.exists(caminho):
        raise FileExistsError(f"Recusando sobrescrever: {caminho}")
    return caminho


def escrever_txt(linhas, caminho):
    with open(novo(caminho), "w", encoding="utf-8") as f:
        f.write("\n".join(map(str, linhas)) + "\n")


# ------------------------------------------------------------------------------
def analisar_colunas(names, var, molde_key):
    fontes = {"<var_names/index>": np.asarray(names).astype(str)}
    for c in var.columns:
        s = var[c]
        if not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)):
            fontes[str(c)] = s.astype(str).values
    nk = np.array([e5.normalizar_id(x) for x in fontes["<var_names/index>"]])
    linhas, chaves = [], {}
    for nome, vals in fontes.items():
        key = np.array([e5.normalizar_id(x) for x in vals])
        ens = np.array([bool(e5.RE_ENSEMBL.match(str(x).strip().upper())) for x in vals])
        chaves[nome] = (vals, key)
        linhas.append({
            "Coluna": nome, "N": len(vals), "Frac_formato_Ensembl": float(ens.mean()),
            "N_unicos_chave": int(len(set(key))), "Chave_unica": bool(len(set(key)) == len(key)),
            "Frac_igual_var_names": float((key == nk).mean()),
            "Frac_igual_molde_posicional": (float((key == molde_key).mean()) if molde_key is not None
                                            and len(molde_key) == len(key) else None),
            "Exemplos": "; ".join(map(str, vals[:3]))})
    return pd.DataFrame(linhas), chaves


def escolher_coluna(df, chaves, pedida):
    if pedida:
        assert pedida in chaves, f"Coluna '{pedida}' nao existe. Colunas: {list(chaves)}"
        r = df[df.Coluna == pedida].iloc[0]
        assert r.Frac_formato_Ensembl == 1.0 and r.Chave_unica, f"Coluna '{pedida}' nao e Ensembl 100% unica"
        return pedida
    cand = df[(df.Frac_formato_Ensembl == 1.0) & (df.Chave_unica)]
    if len(cand) == 0:
        raise RuntimeError("NENHUMA coluna do .var e 100% Ensembl e unica. Veja C1_colunas_var.csv. Se o Ensembl "
                           "esta em outro arquivo (ex.: features.tsv.gz do GEO), a chave canonica deve vir dele.")
    nomes = list(cand.Coluna)
    distintas = {tuple(chaves[n][1]) for n in nomes}
    if len(distintas) > 1:
        raise RuntimeError(f"Varias colunas Ensembl com conteudo DIFERENTE {nomes}; escolha com --coluna-ensembl")
    pref = [n for n in nomes if any(t in n.lower() for t in ("index", "ens", "gene_id"))]
    return (pref or nomes)[0]


def simular(pasta_hvg, cam_pt, ids_com_versao, ids_gene_name, chave_ens, chave_antiga, X_ref, idx_cell, rng):
    """Usa o MODELO HVG CONGELADO. X_ref: (n, 61541) log-norm das celulas idx_cell, ordem do Pan."""
    import scipy.sparse as sp
    A, modelo = e5.carregar_pipeline_disco(pasta_hvg, cam_pt)
    p_ref = e5.prever_modelo(modelo, e5.aplicar_npz(A, "hvg_log", X_ref))
    perm = rng.permutation(X_ref.shape[1])
    Xp = sp.csc_matrix(X_ref)[:, perm].tocsr()
    linhas = []

    def reg(teste, esperado, cobertura, concord=None, detalhe=""):
        linhas.append({"Teste": teste, "Esperado": esperado, "Cobertura_Pan": cobertura,
                       "Concordancia_vs_referencia": concord, "Detalhe": detalhe})

    # T1: Ensembl COM versao, colunas permutadas -> deve reproduzir exatamente
    Xb, pres, res = e5.reindexar_para_pan(Xp, np.asarray(ids_com_versao)[perm], chave_ens, e5.normalizar_id)
    c = float((e5.prever_modelo(modelo, e5.aplicar_npz(A, "hvg_log", Xb)) == p_ref).mean())
    ok1 = res["cobertura_pan"] == 1.0 and c >= 0.9995
    reg("T1_externa_Ensembl_com_versao_permutada", "cobertura 1,0 e predicao identica", res["cobertura_pan"], c)

    # T2: Ensembl SEM versao + 5% de genes ausentes
    sem_v = np.array([e5.normalizar_id(x) for x in ids_com_versao])
    manter = np.sort(rng.choice(len(perm), size=int(0.95 * len(perm)), replace=False))
    _, pres2, res2 = e5.reindexar_para_pan(Xp[:, manter], sem_v[perm][manter], chave_ens, e5.normalizar_id)
    cov_h = e5.cobertura_hvg(pres2, A["hvg_cols"])
    ok2 = abs(res2["cobertura_pan"] - 0.95) < 0.005
    reg("T2_externa_Ensembl_sem_versao_5pct_ausentes", "cobertura ~0,95", res2["cobertura_pan"], None,
        f"cobertura_HVG={cov_h['cobertura_hvg']:.4f}")

    # T3: externa so com gene_name e chave Ensembl -> LIMITACAO esperada
    _, _, res3 = e5.reindexar_para_pan(Xp, np.asarray(ids_gene_name)[perm], chave_ens, e5.normalizar_id)
    reg("T3_externa_gene_name_com_chave_Ensembl", "LIMITACAO: cobertura baixa (exige tabela simbolo->Ensembl)",
        res3["cobertura_pan"])

    # T4: chave ANTIGA (var_names) com externa Ensembl -> problema que esta etapa corrige
    _, _, res4 = e5.reindexar_para_pan(Xp, np.asarray(ids_com_versao)[perm], chave_antiga, e5.normalizar_id)
    reg("T4_externa_Ensembl_com_chave_ANTIGA_var_names", "PROBLEMA corrigido: cobertura baixa", res4["cobertura_pan"])
    return pd.DataFrame(linhas), bool(ok1 and ok2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=e5.DEFAULT_RUN)
    ap.add_argument("--coluna-ensembl", default=None, help="Nome da coluna do .var com Ensembl (auto se omitido)")
    ap.add_argument("--molde", default=e5.CAM_MOLDE_ENSG)
    args = ap.parse_args()

    run = os.path.abspath(args.run)
    e5dir = os.path.join(run, "etapa5_modelos_finais_v2")
    art = os.path.join(e5dir, "artefatos_congelados")
    cam_man_v2 = os.path.join(e5dir, "MANIFESTO_INFERENCIA_EXTERNA_V2.json")
    for cam in [cam_man_v2, os.path.join(art, "genes_pan_61541_ids_originais.txt"),
                os.path.join(art, HVG, "transformador.npz"), os.path.join(art, HVG, "mlp_canonico.pt"),
                os.path.join(e5dir, "split_final_90_10_early_stopping.npz")]:
        if not os.path.exists(cam):
            raise FileNotFoundError(f"Etapa 5 v2 incompleta, falta: {cam}")
    saida = os.path.join(e5dir, "etapa5c_chave_ensembl")
    if os.path.exists(saida):
        raise FileExistsError(f"A Etapa 5c ja foi executada (pasta existe): {saida}")

    print("=" * 78)
    print("ETAPA 5c - CHAVE CANONICA ENSEMBL")
    print("=" * 78)
    ad = sc.read_h5ad(e5.CAM_H5AD_LOG, backed="r")
    var, names, n_vars = ad.var.copy(), np.asarray(ad.var_names.astype(str)), ad.n_vars
    with open(os.path.join(art, "genes_pan_61541_ids_originais.txt"), encoding="utf-8") as f:
        orig_v2 = np.array([l.strip() for l in f if l.strip()])
    assert n_vars == 61541 and np.array_equal(orig_v2, names), \
        "var_names do h5ad != lista congelada na Etapa 5 v2 (ordem diferente!): os indices dos HVGs nao valem."
    print(f"[1] .var com {var.shape[1]} colunas: {list(var.columns)}")

    molde_key = None
    if os.path.exists(args.molde):
        molde = pd.read_csv(args.molde, sep="\t", header=None)[0].astype(str).values
        molde_key = np.array([e5.normalizar_id(x) for x in molde])
    df, chaves = analisar_colunas(names, var, molde_key)
    pd.set_option("display.width", 220, "display.max_colwidth", 60)
    print(df.to_string(index=False))

    col = escolher_coluna(df, chaves, args.coluna_ensembl)       # se falhar, nada e escrito em disco
    os.makedirs(saida, exist_ok=False)
    df.to_csv(novo(os.path.join(saida, "C1_colunas_var.csv")), index=False)
    vals, key = chaves[col]
    print(f"\n[2] Coluna Ensembl escolhida: '{col}' (61.541 IDs Ensembl, unicos sem versao)")
    assert len(key) == 61541 and len(set(key)) == 61541
    if molde_key is not None:
        r = df[df.Coluna == col].iloc[0]
        melhor = df.sort_values("Frac_igual_molde_posicional", ascending=False).iloc[0]
        print(f"    molde == '{col}' em {r.Frac_igual_molde_posicional:.4f} das posicoes; melhor correspondencia do "
              f"molde: '{melhor.Coluna}' ({melhor.Frac_igual_molde_posicional:.4f})")

    # ---------------- arquivos de alinhamento
    escrever_txt(key, os.path.join(saida, "genes_pan_61541_ensembl_sem_versao.txt"))
    escrever_txt(vals, os.path.join(saida, "genes_pan_61541_ensembl_com_versao.txt"))
    escrever_txt(names, os.path.join(saida, "genes_pan_61541_gene_name.txt"))
    cols_hvg = np.load(os.path.join(art, HVG, "hvg2000_indices_colunas.npy"))
    with open(os.path.join(art, HVG, "hvg2000_chaves_mapeamento.txt"), encoding="utf-8") as f:
        chaves_v2 = [l.strip() for l in f if l.strip()]
    assert chaves_v2 == [e5.normalizar_id(x) for x in names[cols_hvg]], "HVGs da v2 inconsistentes com os indices"
    assert len(set(key[cols_hvg])) == 2000
    escrever_txt(key[cols_hvg], os.path.join(saida, "hvg2000_ensembl_sem_versao.txt"))
    escrever_txt(names[cols_hvg], os.path.join(saida, "hvg2000_gene_name.txt"))
    print(f"[3] Arquivos de alinhamento escritos em {saida}")

    # ---------------- simulacoes com o modelo HVG congelado
    print("[4] Simulacoes de coorte externa com o modelo HVG congelado")
    sp_ = np.load(os.path.join(e5dir, "split_final_90_10_early_stopping.npz"))
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(sp_["idx_val_es"], size=1500, replace=False))
    X_ref = ad[idx, :].to_memory().X
    ad.file.close()
    dsim, ok_sim = simular(os.path.join(art, HVG), os.path.join(art, HVG, "mlp_canonico.pt"), vals, names, key,
                           np.array([e5.normalizar_id(x) for x in names]), X_ref, idx, rng)
    print(dsim.to_string(index=False))
    dsim.to_csv(novo(os.path.join(saida, "C3_simulacao_externa.csv")), index=False)

    # ---------------- manifesto v2.1
    with open(cam_man_v2, encoding="utf-8") as f:
        m2 = json.load(f)
    w_ok = m2["verificacao_ordem_W"]["status"] == "CONFIRMADA"
    liberado = bool(w_ok and ok_sim)
    m = {
        "versao_manifesto": "v2.1",
        "gerado_em": datetime.datetime.now().isoformat(timespec="seconds"),
        "deriva_de": {"arquivo": "MANIFESTO_INFERENCIA_EXTERNA_V2.json", "sha256": sha_arq(cam_man_v2)},
        "correcoes_em_relacao_ao_v2": [
            "Chave de mapeamento entre coortes passa a ser o ID ENSEMBL sem versao (coluna '%s' do .var), nao os "
            "var_names (mistura de gene_name e Ensembl)." % col,
            "O flag de liberacao do SWeeP agora exige ordem W CONFIRMADA E testes de chave Ensembl aprovados.",
            "Arquivos 'hvg2000_ensembl_ids_*.txt' das Etapas 3, 4 e 5 v1 contem sobretudo gene_name (nome enganoso).",
        ],
        "genes": {"chave_canonica": f"Ensembl sem versao (coluna '{col}')",
                  "arquivo_alinhamento": "etapa5c_chave_ensembl/genes_pan_61541_ensembl_sem_versao.txt",
                  "sha256_alinhamento": sha_lista(key), "n_genes": 61541,
                  "sha256_ensembl_com_versao": sha_lista(vals), "sha256_gene_name": sha_lista(names),
                  "hvg2000_ensembl": "etapa5c_chave_ensembl/hvg2000_ensembl_sem_versao.txt",
                  "sha256_hvg2000_ensembl": sha_lista(key[cols_hvg])},
        "verificacao_ordem_W": m2["verificacao_ordem_W"],
        "verificacao_chave_ensembl": {"testes": dsim.to_dict(orient="records"), "aprovada": ok_sim},
        "sweep_liberado_para_inferencia_externa": liberado,
        "limitacao_conhecida": ("Coortes externas SO com gene_name (sem Ensembl) NAO sao mapeaveis por esta chave "
                                "(ver T3): exigem tabela simbolo->Ensembl versionada, com genes sem correspondencia "
                                "reportados."),
        "normalizacao_h5ad_evidencia": m2.get("normalizacao_h5ad_evidencia"),
        "regras_inferencia_externa_A_REVISAR_antes_de_usar": m2["regras_inferencia_externa_A_REVISAR_antes_de_usar"],
        "artefatos_por_braco": m2["artefatos_por_braco"],
        "sha256": m2["sha256"],
    }
    cam_m = novo(os.path.join(saida, "MANIFESTO_INFERENCIA_EXTERNA_V2_1.json"))
    with open(cam_m, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False, default=str)
    with open(novo(os.path.join(saida, "MANIFESTO_V2_1.sha256")), "w") as f:
        f.write(sha_arq(cam_m) + "  MANIFESTO_INFERENCIA_EXTERNA_V2_1.json\n")
    with open(novo(os.path.join(saida, "ALINHAMENTO_LEIA_ME.txt")), "w", encoding="utf-8") as f:
        f.write("ARQUIVO DE ALINHAMENTO CANONICO\n"
                "genes_pan_61541_ensembl_sem_versao.txt: 1 ID Ensembl (sem versao) por linha, na ORDEM EXATA das colunas "
                "do Pan e das linhas da W (orthbase_mproj_600d).\n"
                "Contrato: toda matriz a projetar/classificar deve (a) ser reindexada a esta lista a partir de IDs "
                "Ensembl da propria coorte e (b) trazer, ao lado, a lista de IDs na ordem de suas colunas, que o "
                "script confere contra este arquivo antes de multiplicar pela W.\n")

    print("\n" + "=" * 78)
    print(f"ETAPA 5c CONCLUIDA | chave: {col} | testes de chave: {'OK' if ok_sim else 'FALHARAM'} | "
          f"ordem W: {m2['verificacao_ordem_W']['status']}")
    print(f"SWeeP liberado para inferencia externa: {liberado}")
    print(f"Saida: {saida}")
    print("=" * 78)
    if not ok_sim:
        sys.exit(2)


if __name__ == "__main__":
    main()