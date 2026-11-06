library(rSWeeP)
library(Matrix)

cat("1. Carregando matriz gigante do Matisse (Pode levar alguns minutos)...\n")
# Lê o arquivo gerado pelo streaming
matriz_esparsa_matisse <- readMM("Matriz_Matisse_Alinhada_05.mtx")

cat("2. Executando projeção matemática (SWeeP)...\n")
# CRÍTICO: Usando a exata mesma 'base160k' do PAN!
sw_matisse <- matriz_esparsa_matisse %*% base160k
matriz_densa_matisse <- as.matrix(sw_matisse)
# Limpeza de RAM pesada
rm(matriz_esparsa_matisse, sw_matisse)
gc()

sweep_df <- as.data.frame(matriz_densa_matisse)

# =========================================================
# 3. ANEXANDO O GABARITO (Descomente se você tiver os rótulos numéricos do Matisse)
# =========================================================
# Se você tiver um CSV com as classes reais do Matisse (de 1 a 8), você pode plugar aqui:
# rotulos_matisse <- read.csv("Rotulos_Matisse_Limpos.csv", header = TRUE)
# sweep_df$Cell_Type <- rotulos_matisse$Cell_Type_Num

arquivo_saida <- "SWeeP_600_Matisse.txt"
cat("4. Exportando para o PyTorch:", arquivo_saida, "...\n")
write.table(sweep_df, 
            file = arquivo_saida, 
            sep = "\t", 
            quote = FALSE, 
            row.names = FALSE, 
            col.names = TRUE)

# Limpeza final
rm(matriz_densa_matisse, sweep_df)
gc()

cat("\n[SUCESSO] Matisse SWeepado e pronto para a inferência no PyTorch!\n")


cat("Exportando a Base Ortogonal para o Python...\n")
write.csv(as.matrix(base160k), "base_sweep_pan.csv", row.names = FALSE)
cat("[OK] Base salva com sucesso!\n")