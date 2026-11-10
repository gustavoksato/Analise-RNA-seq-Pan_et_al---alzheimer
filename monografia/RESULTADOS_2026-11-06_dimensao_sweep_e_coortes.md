# Resultados de 06/11/2026: dimensão do SWeeP e transferência entre Pan e Dharsini

Todas as métricas são F1-macro nas classes 1 a 6 (Astro, Oligo, Exc, Inh, OPC, Micro). Os ICs de 95% vêm de bootstrap por **doador**, pareado entre os braços.
Raiz dos resultados: `/media/gustavo-k-sato/D61CF48E1CF46B3F1/analisesPAN/resultados_MLP_GSE267301_v2/`

## 1. Qual dimensão do SWeeP preserva mais informação? (Etapa 2b; só treino e validação do Pan, teste cego intocado)

- A W oficial (sha 1eee02c5…) é uma **base ortonormal gaussiana aleatória**, então as primeiras d colunas formam um SWeeP válido de dimensão d.
- Comparamos também bases aleatórias novas, com 3 seeds, e a variante L2, que normaliza cada célula. Essa variante existe porque ‖xW‖ tem correlação de 0,995 com √(genes detectados).

| d | 50 | 100 | 300 | 600 | 1.000 | 2.000 | 3.000 | 4.000 | HVG→PCA50 |
|---|---|---|---|---|---|---|---|---|---|
| MLP, F1 de validação (bruto) | 0,64 | 0,82 | 0,94 | 0,970 | 0,982 | **0,989** | 0,987 | 0,990 | 0,990 |
| F1 Endothelial | 0,39 | 0,64 | 0,85 | 0,91 | 0,96 | 0,98 | 0,98 | 0,99 | 0,98 |
| Geometria: Spearman das similaridades cosseno | 0,59 | 0,74 | 0,87 | 0,92 | 0,95 | 0,98 | 0,99 | 0,99 | – |

- **Não há ganho relevante acima de d ≈ 2.000.** As diferenças ficam abaixo de 0,3 ponto, o mesmo tamanho da variação entre bases (dp de cerca de 0,003).
- **O SWeeP-2000 empata com o HVG→PCA50.**
- Na projeção aleatória, PCA no SWeeP não ajuda e dimensões baixas perdem muito. A informação se espalha uniformemente pelas colunas, ao contrário do PCA.
- Bruto e L2 empatam dentro do Pan.
- **Proposta:** d = 2.000. Pela regra 1-SE só com o dp entre seeds, ela escolheria 3.000 ou 4.000. Usando o dp entre bases, 2.000 fica dentro de 1-SE.
- Arquivos: `etapa2b_dimensao_sweep_20261106_113938/` e `etapa2b_dimensao_sweep_extensao_20261106_123242/` (tabelas, decisão em JSON e figura).

## 2. Pan → Dharsini com os modelos TRAVADOS da Etapa 5 (8 classes, d = 600) — Etapa 6b

| Escopo (seed 42) | HVG→PCA50 | SWeeP600 | SWeeP600 + PCA40 |
|---|---|---|---|
| Homólogo (10X V3, All Nuclei; 16 doadores) | 0,996 | 0,986 | 0,984 |
| Geral (40 doadores) | 0,987 | 0,966 | 0,960 |
| DropSeq NeuN | 0,821 | 0,790 | 0,774 |
| BA17 | 0,965 | 0,902 | 0,897 |

- HVG − SWeeP600: **+0,010 [0,007; 0,014] no Homólogo** e **+0,019 [0,015; 0,024] no Geral**.
- A perda se concentra nos **neurônios inibitórios** (0,90 contra 0,98).
- Sem o doador 111, os resultados são idênticos.
- Pasta: `run_20261102_201556/etapa6_gse263468/etapa6b_inferencia_20261106_123201/`

## 3. Treino cruzado com o MESMO protocolo nas duas direções (6 classes, 3 seeds) — Etapa 7

**Pan → Dharsini** (`etapa7_pan_para_dharsini_20261106_131808/`)

| Escopo | HVG | SWeeP600 | **SWeeP2000** | HVG − SWeeP2000 [IC 95%] |
|---|---|---|---|---|
| Homólogo | 0,996 | 0,989 | **0,997** | −0,000 [−0,001; 0,001] |
| Geral (amostrado, 40 doadores) | 0,988 | 0,975 | **0,990** | −0,002 [−0,003; −0,001] |
| DropSeq, All Nuclei | 0,957 | 0,934 | **0,970** | −0,011 [−0,017; −0,006] |
| ~~DropSeq, NeuN~~ (inválido, ver correção abaixo) | ~~0,785~~ | ~~0,805~~ | ~~0,930~~ | ~~−0,146~~ |
| ADNC high | 0,986 | 0,968 | 0,989 | −0,002 [−0,005; 0,000] |

**Dharsini → Pan** (`etapa7_dharsini_para_pan_20261106_132919/`)

| Escopo | HVG | SWeeP600 | **SWeeP2000** | HVG − SWeeP2000 [IC 95%] |
|---|---|---|---|---|
| Pan, todos (8 doadores) | 0,967 | 0,969 | **0,975** | −0,011 [−0,027; −0,006] |
| Pan sem ALSP | 0,953 | 0,960 | **0,965** | −0,016 [−0,029; −0,010] |

## Leitura

1. Toda a desvantagem do SWeeP que aparecia nas Etapas 3 a 6 **vinha da dimensão 600**, não do método.
2. Com **d = 2.000**, o SWeeP:
   - empata com o HVG no dado homólogo;
   - no DropSeq All Nuclei, vence na **glia** e perde um pouco nos **neurônios inibitórios** (ver correção abaixo);
   - supera o HVG na direção Dharsini → Pan (cerca de 1 ponto; só 8 doadores).

   Isso sustenta que **a dimensão 600 era a causa da desvantagem**. A tese de "robustez à química" vale **só em parte** (glia) e precisa ser testada com as métricas por estrato definidas em `METODOLOGIA_coortes_e_avaliacao.md`.

> **⚠️ Correção de 07/11/2026: o "DropSeq NeuN +14 pontos" é um artefato da métrica.**
> - Nos estratos NeuN+ quase não há glia: no DropSeq NeuN da Etapa 7, 30 Oligo, 11 OPC e 9 Micro, contra 23.060 Exc e 5.190 Inh.
> - O F1-macro de 6 classes dá a essas dezenas de células o mesmo peso que aos neurônios.
> - **Só nos neurônios (F1 Exc/Inh): HVG 0,968, SWeeP600 0,890, SWeeP2000 0,960.** Ou seja, o HVG fica levemente à frente.
> - No **DropSeq All Nuclei** (suporte adequado em todas as classes), SWeeP2000 contra HVG:
>   - Astro 0,996 contra 0,971; Oligo 0,971 contra 0,944; Micro 0,969 contra 0,942;
>   - Inh 0,933 contra 0,952; Exc 0,979 contra 0,983.
> - O DropSeq NeuN passa a ser **exploratório**: química modificada, núcleos rasos, rótulo herdado da integração e glia residual. As regras de avaliação por estrato estão em `METODOLOGIA_coortes_e_avaliacao.md`.
3. A normalização L2 não muda o resultado.
4. **Ressalvas:**
   - O Pan tem só 8 doadores, então os ICs da direção Dharsini → Pan são grosseiros.
   - O SWeeP-2000 usa uma base aleatória nova. A W oficial só vai até 600. Para fechar, é preciso gerar e versionar uma W oficial de 2.000.
   - Na Etapa 7 o Geral é uma amostra estratificada de até 1.000 núcleos por doador × química × sorting.

## Próximos passos sugeridos
- Gerar e versionar a W-2000 oficial (com sha) e refazer as Etapas 3 a 5 (teste cego e LODO) com os braços HVG→PCA50 e SWeeP-2000, numa run nova.
- Rodar a 6b com os modelos de 2.000 travados.
- Usar o Mathys (10X V2, outra coorte) como juiz neutro das duas direções.
