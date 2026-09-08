
# Projeto PAN-AIBIA: Pipeline de Classificação Celular (snRNA-seq)

Este diretório contém os dados, documentações e scripts desenvolvidos no Laboratório AIBIA (UFPR) para o processamento de dados transcriptômicos de Alzheimer e classificação celular automatizada utilizando Redes Neurais (MLP). O pipeline abrange desde o controle de qualidade até a projeção matemática e modelagem preditiva.

## 📂 Estrutura de Arquivos

### 1. Documentação e Referências Teóricas

* **`Pan - AIBIA - documentação.docx`**: Documentação oficial e completa do projeto. Contém as justificativas biológicas, thresholds aplicados e o fluxo de trabalho detalhado.
* **`s40478-024-01853-5-1.pdf`**: Artigo científico base de referência metodológica para os marcadores transcriptômicos da doença.

### 2. Matrizes de Dados

* **`Cópia de Matriz_Panv2.mtx`**: Matriz esparsa de expressão (formato Matrix Market) contendo o dataset limpo após as etapas de Controle de Qualidade (QC).
* **`Matriz_SWeeP_Pan.txt`**: Matriz densa gerada após a aplicação da técnica de projeção SWeeP (~600 dimensões), preservando a distância euclidiana das assinaturas celulares para ingestão do modelo.

### 3. Scripts de Processamento (Pipeline)

* **`sweep_pan_pca.R`**: Script analítico em R. Importa a matriz esparsa, aplica a base ortogonal aleatória (SWeeP), extrai os Componentes Principais (PCA) e plota os gráficos de diagnóstico visual (Scree plots, UMAP e Scatters).
* **`analise_ultimato.py`**: Script de inicialização e avaliação da Rede Neural (Multilayer Perceptron) no PyTorch, englobando o estudo de ablação de PCs, Early Stopping e geração das métricas finais (Matriz de Confusão e F1-Score).

---

### ⚙️ Módulo Central: Preparação do Modelo

* **`harmoniz_splitv2`**: **O motor de engenharia de dados do projeto.** Este script faz a ponte essencial entre a extração matemática e o treinamento da rede neural. Suas funções incluem:
* **Harmonização (Batch Effect):** Mitigação de variâncias técnicas e ruidosas inerentes aos diferentes lotes biológicos originais.
* **Data Splitting Estratificado:** Divisão rigorosa da matriz nos conjuntos de Treino e Validação (ex: 80/20), garantindo que a proporção biológica original das classes raras (como pericitos ou células ependimárias) seja mantida matematicamente em todos os tensores.
* **Prevenção de Vazamento (Leakage):** Aplicação da padronização de escala (`StandardScaler`) baseada exclusivamente nos dados de treino.
* **Balanceamento de Custo:** Cálculo e geração dos pesos invertidos para equalizar a predição da rede neural sobre a distribuição desigual de tipos celulares.
