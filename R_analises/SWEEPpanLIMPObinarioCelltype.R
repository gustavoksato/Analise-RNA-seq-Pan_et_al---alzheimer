library(rSWeeP)
library(Matrix)

# 1. Carrega a matriz esparsa
cat("1. Carregando Matriz Esparsa...\n")
matriz_esparsa <- readMM("Matriz_Pan_Binarizada_Limpa.mtx")
Smatriz_esparsa <- as(matriz_esparsa, "CsparseMatrix")

# 2. Configuração do SWeeP
psz <- 600
nfeat <- ncol(matriz_esparsa)
cat("2. Gerando Base Ortogonal (", nfeat, "genes ->", psz, "dimensões)...\n")
base160k = orthBase(nfeat, psz)

# 3. Projeção Matemática
cat("3. Executando projeção SWeeP...\n")
sw <- matriz_esparsa %*% base160k
matriz_densa <- as.matrix(sw)

# Limpeza de memória crítica
rm(matriz_esparsa, Smatriz_esparsa, base160k, sw)
gc()

# 4. Ingestão dos Rótulos Otimizados
cat("4. Carregando e Anexando Rótulos Numéricos...\n")
# Lê o arquivo que exportamos do Python (contém cabeçalho "Cell_Type_Num")
rotulos <- read.csv("Rotulos_Pan_Limpos.csv", header = TRUE)

# Verifica a integridade dimensional
if(nrow(rotulos) != nrow(matriz_densa)) {
  stop("ERRO FATAL: O número de linhas da matriz SWeeP não bate com o número de rótulos!")
}

# 5. Mesclagem e Exportação Final
cat("5. Mesclando dados e preparando exportação...\n")
sweep_df <- as.data.frame(matriz_densa)
# Adiciona a coluna de rótulos ao final do data frame do SWeeP
sweep_df$Cell_Type <- rotulos$Cell_Type_Num

cat("6. Exportando para o PyTorch (Isso pode demorar um pouco)...\n")
# Exporta o arquivo final que será lido pelo script harmoniz_splitv2
write.table(sweep_df, 
            file = "SWeeP_600_Classes_Alzheimer.txt", 
            sep = "\t", 
            quote = FALSE, 
            row.names = FALSE, # Remove os nomes das linhas para economizar espaço
            col.names = TRUE)

cat("Processo Concluído com Sucesso!\n")
gc()