# Metodologia: coortes, estratos e regras de avaliação

Rascunho para o capítulo de Métodos. Escrito em 08/11/2026, a partir das análises de 06 e 07/11/2026.
Fontes: Pan et al. 2024 (*Acta Neuropathol Commun*, GSE267301); Dharsini et al. 2024 (bioRxiv, GSE263468); Mathys et al. 2019 (*Nature*, ROSMAP). Os textos dos artigos estão em `Documents/artigos_referencia/`.

---

## 1. Coortes e protocolos

| | **Pan (GSE267301)**, treino | **Dharsini (GSE263468)**, coorte externa | **Mathys (2019)**, coorte neutra (prevista) |
|---|---|---|---|
| Laboratório | Cobos (Stanford) | Cobos (Stanford), o mesmo pipeline | Tsai (MIT), ROSMAP |
| Doadores | 8 (5 controles, 2 AD com Braak IV, 1 ALSP) | 46 (40 sem os 6 que também estão no Pan) | 48 (24 com amiloide baixo, 24 com amiloide alto) |
| Região | pré-frontal, **substâncias cinzenta e branca** (ALSP: 4 regiões) | BA9, BA7, BA17, **só substância cinzenta** | BA10 |
| Química | 10X 3' v3 (V3.1) | **DropSeq modificado**, **10X v2** e **10X v3** | 10X 3' v2 |
| Separação dos núcleos (FANS) | **PAX6+/NeuN−** (glia) nos controles e AD; Hoechst+ ou NeuN− no ALSP | **All Nuclei** (Hoechst+) **e** **NeuN+**, as duas a partir de cada bloco | sem separação |
| Contagem | kallisto-bustools, Ensembl 105, spliced + unspliced | o mesmo pipeline. Os autores fizeram QC e agrupamento com as contagens **unspliced** | CellRanger 2.0, pré-mRNA |
| Rótulos | Leiden + marcadores, depois do Harmony com o doador como lote | Leiden + marcadores, depois do **Harmony com química + região como lote** | agrupamento + marcadores (8 tipos principais) |
| Nosso QC (contagem total) | mito < 15%, 200 ≤ genes < 8.000, UMI ≥ 500 | o mesmo | a definir (o arquivo atual é um subconjunto já filtrado) |

**Pontos do protocolo do Dharsini que importam para a avaliação:**
- **DropSeq modificado:** lise com sarkosyl 1% + 72 °C por 5 min, usado em 184 das 243 amostras. Detecta cerca de metade dos genes do 10X v3.
- **Antes da extração**, a substância branca e as leptomeninges foram retiradas. Isso reduz oligodendrócitos e praticamente elimina VLMC.
- **Os rótulos dependem da integração.** Num núcleo DropSeq raso, o tipo celular é em parte herdado do agrupamento integrado, e é menos certo do que num núcleo 10X v3.

## 2. All Nuclei e NeuN+: dois sortings do mesmo tecido

- Cada bloco foi homogeneizado e dividido por FANS em **duas suspensões do mesmo homogenato**:
  - **All Nuclei:** núcleos Hoechst+ isolados, de todos os tipos;
  - **NeuN+:** Hoechst+ e NeuN+ (RBFOX3), ou seja, só neurônios.
- **Não há núcleos em comum:**
  - as 243 bibliotecas pertencem cada uma a um único sorting (115 All Nuclei e 128 NeuN+);
  - os índices são únicos;
  - as 288 coincidências de barcode de 16 nt entre sortings são esperadas por acaso, porque bibliotecas distintas usam a mesma lista de barcodes.
- **Os dois sortings são amostras pareadas, não independentes:** mesmo doador, mesma região, mesmo bloco. **A unidade estatística é o doador** (ou doador × região).
- **A glia encontrada no NeuN+** (de 8 a 224 células por classe) é impureza da separação, dupleto neurônio + glia (como os oligodendrócitos satélites) ou erro de anotação. **Seu rótulo não é confiável.**

### Estratos (Dharsini pós-QC, 40 doadores)
| Estrato | Doadores (low/int/high) | Bibliotecas | Núcleos | Genes por núcleo, mediana (total / unspliced) | Regiões | Glia |
|---|---|---|---|---|---|---|
| V3 · All | 16 (4/5/7) | 19 | 82.963 | 1.941 / 1.556 | BA9, BA7, BA17 | suficiente |
| V3 · NeuN | 19 (6/5/8) | 23 | 124.678 | 2.587 / 2.086 | BA9, BA7, BA17 | residual |
| V2 · All | 7 (4/2/1) | 7 | 24.494 | 1.223 / 832 | só BA9 | suficiente |
| V2 · NeuN | 5 (4/0/1) | 5 | 27.934 | 1.764 / 1.164 | só BA9 | residual |
| DropSeq · All | 29 (8/7/14) | 72 | 41.055 | 936 / 593 | BA9, BA7, BA17 | suficiente |
| DropSeq · NeuN | 33 (11/7/15) | 83 | 71.784 | 1.017 / 650 | BA9, BA7, BA17 | residual (24 Micro, 27 OPC) |

**Pareamentos disponíveis:**
- **72 de 88** combinações doador × região × química têm os dois sortings: isso permite medir o efeito do **sorting**;
- **11** combinações doador × região têm 10X v3 e DropSeq: isso permite medir o efeito da **química**.

## 3. Composição: tecido contra datasets

| Tipo | **Tecido (aproximado)¹** | Dharsini All Nuclei | Dharsini, todos os núcleos | Pan | Mathys |
|---|---|---|---|---|---|
| Neurônios (Exc + Inh) | ~35–40% | 64% | 85% | 15% | 65% |
| Oligodendrócitos | ~30–45% | 22% | 9% | 42% | 19% |
| Astrócitos | ~7–12% | 8% | 3,5% | 32% | 5,5% |
| Microglia | ~2–4% | 3% | 1,2% | 3% | 1,0% |
| OPC | (dentro da glia) | 3,4% | 1,4% | 5,7% | 4,7% |
| Vasculares (endotelial, pericito, VLMC) | ~10–18%² | 0,1% | 0,04% | 2,4% | 0,3% |

¹ Pelvig et al. 2008 (*Neurobiol Aging*), neocórtex humano adulto com substâncias cinzenta e branca: cerca de 26 bilhões de neurônios e 39 bilhões de células gliais, com a glia dividida em ~75% Oligo, 17–20% Astro e 5–6,5% Micro. Na substância cinzenta pura, a fração de neurônios é maior e a de oligodendrócitos é menor.
² Estimativa indireta: ~17% no córtex de rato (Davanlou & Smith 2004) e ~18% na amígdala humana (calculado a partir de García-Amado & Prensa 2012). Não há contagem equivalente para o neocórtex humano.

**Por que as classes minoritárias são tão raras:**
1. **Desenho:** 60% dos núcleos do Dharsini vêm do NeuN+, de propósito, porque o estudo é sobre neurônios.
2. **Dissecção:** a retirada da substância branca reduz oligodendrócitos; a das leptomeninges elimina VLMC.
3. **Extração:** homogeneização em Dounce, filtro de 40 µm e gradiente de iodixanol. Os núcleos vasculares tendem a ficar presos a fragmentos de vaso e detritos.
4. **FANS:** a janela que "descarta as menores partículas" penaliza núcleos pequenos (microglia, endoteliais).
5. **QC por número de genes:** núcleos pequenos ou com pouco RNA (glia, vasculares e até os neurônios Ex5) caem abaixo do corte. O DropSeq, menos sensível, agrava isso.
6. **Núcleo contra célula:** núcleos de microglia têm poucos transcritos (Thrupp et al. 2020, *Cell Reports*).
7. **Agrupamento:** grupos pequenos são absorvidos ou descartados. No Dharsini restaram 183 vasculares, 133 deles pericitos.

**Consequência:** a composição de cada dataset reflete **o protocolo, não o tecido**. Métricas calculadas somando todas as células medem em parte essa composição artificial.

## 4. Regras de avaliação

1. **Unidade de inferência = doador** (ou doador × região). Os intervalos de confiança vêm de bootstrap de doadores, pareado entre braços.
2. **Relatar por estrato química × sorting.** O "Geral" do Dharsini não deve ser usado como agregado bruto de células, porque o peso de cada estrato muda conforme a amostragem.
3. **F1-macro só com classes elegíveis**: ≥ 50 núcleos e ≥ 3 doadores no estrato. O F1 por classe sempre acompanha o suporte (núcleos e doadores).
4. **Estratos NeuN+: só a tarefa neuronal** (F1 de Exc e Inh). A glia residual entra só como descrição.
5. **Papel de cada estrato:**
   - **V3 · All:** confirmatório (química mais parecida com a do Pan);
   - **V2 · All e DropSeq · All:** robustez à química e à profundidade;
   - **V3 · NeuN e V2 · NeuN:** tarefa neuronal;
   - **DropSeq · NeuN:** **exploratório / teste de estresse**, sem peso nas conclusões (química modificada + núcleos rasos + rótulo herdado da integração + glia residual).
6. **Endotelial e VLMC** (cerca de 20 células cada no Dharsini) só como descrição, nunca no F1-macro.
7. **Comparações pareadas preferidas:**
   - química: os 11 pares V3 × DropSeq, com o mesmo doador e a mesma região;
   - sorting: os 72 pares All × NeuN, com o mesmo doador, a mesma região e a mesma química.
8. **Composição celular** (por exemplo, contra Braak) só no **All Nuclei**, e só como comparação **relativa** entre grupos de patologia, ajustada por química. Nunca como estimativa da proporção real do tecido.
9. **Região e ADNC** só dentro dos estratos com suporte. O BA17 se confunde com profundidade, porque tem muitos Ex5, que são núcleos de poucos genes.

## 5. O que cada etapa já rodada fez (e onde as regras ainda não foram aplicadas)

| Etapa | Treino → teste | Classes / métrica | Observações |
|---|---|---|---|
| 2b, dimensão do SWeeP | Pan → Pan (validação) | 8 classes; F1-macro, sonda linear, geometria | split por **célula**, com doadores em comum entre treino e validação |
| 6b, modelos travados | Pan (Etapa 5, d = 600) → Dharsini, todos os 372.908 núcleos | modelos de 8 classes avaliados em 6; F1-macro por estrato e IC por doador | os estratos NeuN usaram o F1-macro de 6 classes, o que **viola a regra 4** |
| 7, Pan → Dharsini | Pan → Homólogo + Geral **amostrado** (≤ 1.000 por doador × química × sorting) | 6 classes, 3 seeds, IC por doador | o "Geral" da 7 ≠ o "Geral" da 6b; os estratos NeuN violam a regra 4 |
| 7, Dharsini → Pan | Dharsini (todas as químicas e sortings juntos, ≤ 600 por doador × classe; validação por doador, estratificada por ADNC) → Pan, todos os núcleos | 6 classes, 3 seeds | química e sorting **não balanceados** no treino; o Pan tem só 8 doadores |

As predições salvas permitem re-pontuar a 6b e a 7 segundo as regras da seção 4 sem retreinar. O índice de teste da 7 pode ser reconstruído de forma determinística (166.852 núcleos, conferido).
