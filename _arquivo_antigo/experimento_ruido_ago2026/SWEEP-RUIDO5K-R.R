library(rSWeeP)
library(Matrix)

# 1. Configuração inicial estática (Roda apenas uma vez)
cat("1. Carregando Rótulos Numéricos...\n")
rotulos <- read.csv("Rotulos_Pan_Limpos.csv", header = TRUE)

cat("2. Gerando Base Ortogonal (61541 genes -> 600 dimensões)...\n")
 B  psz <- 600
nfeat <- 61541 # Número exato de features do PAN
base160k <- orthBase(nfeat, psz)

cat("\n3. Iniciando Loop de Processamento em Lote (1k a 5k)...\n")

for(i in 8:10) {
  nivel <- paste0(i, "k")
  arquivo_mtx <- paste0("Matriz_Pan_Ruido_", nivel, ".mtx")
  arquivo_saida <- paste0("SWeeP_600_Classes_Alzheimer_Ruido_", nivel, ".txt")
  
  cat("\n========================================\n")
  cat("-> Processando:", arquivo_mtx, "\n")
  
  # Carrega a matriz esparsa
  cat("   Carregando matriz...\n")
  matriz_esparsa <- readMM(arquivo_mtx)
  
  # Projeção SWeeP
  cat("   Executando projeção matemática...\n")
  sw <- matriz_esparsa %*% base160k
  matriz_densa <- as.matrix(sw)
  
  # Limpeza parcial para liberar RAM antes de criar o DataFrame
  rm(matriz_esparsa, sw)
  gc()
  
  # Mesclagem
  cat("   Anexando rótulos...\n")
  sweep_df <- as.data.frame(matriz_densa)
  
  # Trava de segurança dimensional
  if(nrow(rotulos) != nrow(sweep_df)) {
    stop(paste("ERRO FATAL: O número de linhas do", nivel, "não bate com os rótulos!"))
  }
  
  sweep_df$Cell_Type <- rotulos$Cell_Type_Num
  
  # Exportação
  cat("   Exportando para o PyTorch:", arquivo_saida, "...\n")
  write.table(sweep_df, 
              file = arquivo_saida, 
              sep = "\t", 
              quote = FALSE, 
              # row.names = FALSE,
              col.names = TRUE)
  
  cat("   [OK] Concluído:", nivel, "\n")
  
  # Limpeza pesada de memória antes do próximo ciclo 
  rm(matriz_densa, sweep_df)
  gc()
}

cat("\n[SUCESSO] Todas as 5 matrizes SWeeP foram exportadas para o PyTorch!\n")

cat("Exportando a Base Ortogonal para o Python...\n")
write.csv(as.matrix(base160k), "base_sweep_pan.csv", row.names = FALSE)
cat("[OK] Base salva com sucesso!\n")