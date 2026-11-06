# Monografia — Classificação de tipos celulares (snRNA-seq, Alzheimer) com SWeeP + MLP

Pipeline oficial: **GSE267301 (Pan et al.)** como coorte de desenvolvimento → inferência cega em coortes externas
(GSE263468 / Dharsini; GSE305140 / Lattke).

Três braços comparados em todas as etapas:

| Braço | Representação |
|---|---|
| `Baseline_HVG2000_PCA50` | 2.000 HVGs (log-norm) → PCA50 → scaler |
| `SWeeP_600D_Puro` | SWeeP 600D → scaler |
| `SWeeP_600D_PCA40` | SWeeP 600D → PCA → 40 PCs (k escolhido pela regra 1-SE na Etapa 2) → scaler |

## Ordem de execução (`pipeline_GSE267301/`)

| # | Script | O que faz | Saída (dentro do run) |
|---|---|---|---|
| 0–2 | `etapas_0_a_2_selecao_hiperparametros_pan.py` | Auditoria de alinhamento; escolha de pesos (Balanced) e k (40) só com treino/validação | `etapa1_auditoria/`, `etapa2_selecao_hiperparametros/`, `decisao_hiperparametros.json`, `split_indices_70_15_15.npz` |
| 3 | `etapa3_teste_cego_3bracos_pan.py` | Teste cego 70/15/15, 3 braços × 3 seeds | `etapa3_teste_cego/` (Tabelas 2, 2a–2d) |
| 4 | `etapa4_lodo_3bracos_pan.py` | Leave-One-Donor-Out (8 doadores), bootstrap por doador | `etapa4_lodo/` (Tabelas 4, 4a–4i) |
| 5 | `etapa5_modelos_finais_pan.py` | Modelos finais v1 (90/10) — mantido porque a v2 compara com ele | `etapa5_modelos_finais/` |
| 5 v2 | `etapa5_modelos_finais_v2_pan.py` | Preflight (ordem W, IDs), modelos finais congelados + manifesto | `etapa5_modelos_finais_v2/` (Tabelas 5, 5a–5d) |
| 5c | `etapa5c_chave_ensembl_pan.py` | Chave Ensembl canônica para coortes externas. **Precisa estar na mesma pasta que a 5 v2.** | `etapa5_modelos_finais_v2/etapa5c_chave_ensembl/` |
| 6a | `etapa6a_diagnostico_gse263468.py` | Diagnóstico read-only da coorte GSE263468 (sem predição) | `etapa6_gse263468/` |

Run oficial: `/media/gustavo-k-sato/D61CF48E1CF46B3F1/analisesPAN/resultados_MLP_GSE267301_v2/run_20261102_201556`
(Etapas 3, 4, 5 e 5 v2 travadas com `*_LOCK.json` — não reexecutar do zero nesse run.)

## Entradas (caminhos absolutos dentro dos scripts)

- `/media/.../analisesPAN/Pan_58k_LogNorm_Para_BaselineHVG.h5ad` — 58.326 células, log-norm
- `/media/.../analisesPAN/Metadados_Pan_Limpos.csv`, `Rotulos_Pan_Limpos.csv`
- `/media/.../AIBIA/R_analises/SWeeP_600_Classes_Alzheimer_Oficial.tsv` — projeção SWeeP 600D
- `/media/.../AIBIA/R_analises/orthbase_mproj_600d.rds` — matriz W do SWeeP

## Coortes externas (scripts na raiz de `Documents/`)

Ficam na raiz porque usam caminhos **relativos** a `Documents/` — rode a partir dela.

- `dharsini_metadados.py` → `metadados_GSE263468*.csv`
- `dharsini_total.py` → `export_GSE263468_SWeeP/` (lê `~/Downloads/GSE263468_processed_data.h5ad`, 28,7 GB)
- `extracaodharsini.py`, `dharshiniQC.py` — versões anteriores da preparação
- `lattke.py`, `downexploracao.py` — GSE305140 / dataset Down (`dataset_down_syndrome/`)

## Backup

`/media/gustavo-k-sato/D61CF48E1CF46B3F1/BACKUP_MONOGRAFIA_2026-11-06/` — run completo + entradas + W + bundle git
(conferir com `SHA256SUMS.txt`). Cópia no Google Drive.

## Arquivado

Experimentos de ago/2026 (ruído Bernoulli, Matisse, MLP antigo) em `../_arquivo_antigo/`.
Lista de tudo que foi movido (origem → destino): `../_arquivo_antigo/MOVIMENTACOES_2026-11-06.tsv`.
