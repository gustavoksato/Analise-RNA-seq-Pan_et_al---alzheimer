"""
Exploração dos metadados de adata.obs para contar doadores (e não só células)
controle vs AD. Uso no Python interativo:

    exec(open("explorar_metadados.py").read())
    tab = explorar(adata.obs)

Todas as contagens relevantes para a classificação de AD devem ser por DOADOR.
"""
import pandas as pd

DONOR = "Donor"
GRUPO = "Case_No_Case"            # rótulo caso/controle
CANDIDATAS = ["Case_No_Case", "NP.Diagnosis", "Braak.Group", "Braak.stage",
              "Amyloid", "APOE", "Sex", "Brain.Region", "Gray.vs.White", "SORT"]
ANOTACAO = "manual_annotation"


def _sec(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def explorar(obs):
    obs = obs.copy()
    obs.columns = obs.columns.str.strip()   # ' Braak.Group' tem espaço no início
    for c in obs.select_dtypes("category").columns:
        obs[c] = obs[c].astype(object)

    _sec("1. Visão geral")
    print(f"Células: {len(obs):,} | Doadores: {obs[DONOR].nunique()}")
    if "Unique.identifier" in obs:
        print(f"Amostras (Unique.identifier): {obs['Unique.identifier'].nunique()}")

    _sec("2. Células e doadores por categoria")
    for c in [c for c in CANDIDATAS if c in obs]:
        t = pd.DataFrame({
            "celulas": obs[c].value_counts(dropna=False),
            "doadores": obs.groupby(c, dropna=False)[DONOR].nunique(),
        })
        print(f"\n-- {c}\n{t}")

    _sec("3. Atributos constantes dentro do doador? (n de doadores com >1 valor)")
    cols = [c for c in CANDIDATAS + ["Age", "PMI.hr.", "Race", "RIN", "Brain.weight"]
            if c in obs]
    inconsist = {c: int((obs.groupby(DONOR)[c].nunique(dropna=False) > 1).sum())
                 for c in cols}
    print(pd.Series(inconsist, name="doadores_inconsistentes"))
    print("(0 = atributo de doador; >0 = varia por amostra/região ou há erro de rótulo)")

    _sec("4. Tabela por doador")
    donor_cols = [c for c in cols if inconsist[c] == 0]
    tab = obs.groupby(DONOR).agg({**{c: "first" for c in donor_cols}})
    tab["n_celulas"] = obs.groupby(DONOR).size()
    if "Brain.Region" in obs:
        tab["n_regioes"] = obs.groupby(DONOR)["Brain.Region"].nunique()
    print(tab.to_string())

    _sec("5. Cruzamentos no nível do doador")
    if GRUPO in tab:
        for c in ["NP.Diagnosis", "Braak.Group", "Braak.stage", "Amyloid", "APOE", "Sex"]:
            if c in tab:
                print(f"\n-- {GRUPO} x {c}\n{pd.crosstab(tab[GRUPO], tab[c], dropna=False)}")
        for c in ["Age", "PMI.hr."]:
            if c in tab:
                print(f"\n-- {c} por grupo\n"
                      f"{pd.to_numeric(tab[c], errors='coerce').groupby(tab[GRUPO]).describe()}")
        print(f"\n-- células por doador, por grupo\n{tab.groupby(GRUPO)['n_celulas'].describe()}")

    if "Brain.Region" in obs:
        _sec("6. Doador x região (n de células)")
        print(pd.crosstab(obs[DONOR], obs["Brain.Region"]).to_string())

    if ANOTACAO in obs and GRUPO in obs:
        _sec("7. Tipos celulares x grupo (células) e doadores com >=50 células por tipo")
        print(pd.crosstab(obs[ANOTACAO], obs[GRUPO], margins=True))
        por_doador = pd.crosstab(obs[DONOR], obs[ANOTACAO])
        print("\nDoadores com >=50 células em cada tipo:")
        print((por_doador >= 50).sum().rename("doadores"))

    return tab
