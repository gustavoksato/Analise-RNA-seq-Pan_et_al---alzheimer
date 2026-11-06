library(rSWeeP)
library(Matrix)
if (!requireNamespace("data.table", quietly = TRUE)) install.packages("data.table")
if (!requireNamespace("uwot", quietly = TRUE)) install.packages("uwot")
library(data.table)
library(uwot)

# ============================================================================
# 1. CONFIGURAÇÃO E CARREGAMENTO DA ORTHBASE (61.541 x 600)
# ============================================================================
pasta_in  <- "/media/gustavo-k-sato/D61CF48E1CF46B3F1/Analise-RNA-seq-Pan_et_al---alzheimer/Documents/export_GSE263468_SWeeP"
pasta_out <- "resultados_SWeeP_GSE263468"
if (!dir.exists(pasta_out)) dir.create(pasta_out, recursive = TRUE)

nfeat <- 61541
psz   <- 600

# Procura a orthbase na pasta_out ou na pasta atual
caminhos_base <- c(
  file.path(pasta_out, "orthbase_mproj_600d.rds"),
  "orthbase_mproj_600d.rds",
  file.path(pasta_in, "orthbase_mproj_600d.rds")
)
arquivo_base <- caminhos_base[file.exists(caminhos_base)][1]

if (!is.na(arquivo_base)) {
  cat("[1/5] Carregando base ortonormal existente de:", arquivo_base, "\n")
  base160k <- readRDS(arquivo_base)
} else {
  stop("Arquivo 'orthbase_mproj_600d.rds' não encontrado! Verifique o caminho.")
}

# Garante que base160k seja matrix densa padrão (61541 x 600)
base160k <- as.matrix(base160k)
stopifnot(nrow(base160k) == nfeat, ncol(base160k) == psz)

nomes_classes <- c(
  "1" = "Astrocytes",
  "2" = "Oligodendrocytes",
  "3" = "Exc. Neurons",
  "4" = "Inh. Neurons",
  "5" = "OPCs",
  "6" = "Microglia",
  "7" = "Endothelial",
  "8" = "VLMCs"
)

# ============================================================================
# 2. FUNÇÃO DE STREAMING EM LOTES (SWeeP 600D + PCA EXATO VIA COVARIÂNCIA)
# Nunca carrega o .mtx inteiro na RAM: usa pipe awk + fread por faixa de células
# ============================================================================
processar_mtx_em_lotes <- function(caminho_mtx, caminho_rot, arq_saida_sweep, arq_saida_pca50,
                                   batch_cells = 25000, retornar_pca_completo = FALSE) {
  
  cat("\n----------------------------------------------------------------------\n")
  cat("Iniciando processamento em lotes de:", basename(caminho_mtx), "\n")
  
  rotulos <- fread(caminho_rot)
  # Lê apenas as 2 primeiras linhas do .mtx para pegar dimensões (cabeçalho)
  cabecalho <- readLines(caminho_mtx, n = 2)
  dims <- as.numeric(strsplit( trimws(cabecalho[2]), "\\s+" )[[1]])
  n_cells <- dims[1]
  n_cols  <- dims[2]
  stopifnot(n_cols == nfeat, n_cells == nrow(rotulos))
  
  cat(sprintf("  Dimensões: %d células x %d genes | Lotes de %d células\n", n_cells, n_cols, batch_cells))
  
  if (file.exists(arq_saida_sweep)) file.remove(arq_saida_sweep)
  
  # Acumuladores suficientes para o PCA exato (ocupam apenas 2.8 MB na RAM!)
  soma_colunas <- numeric(psz)
  XtX          <- matrix(0.0, nrow = psz, ncol = psz)
  
  inicio_cel <- 1
  lote_id    <- 1
  
  # PASSADA 1: Lê blocos do .mtx, projeta SWeeP (600d), grava TSV da Letícia e acumula Covariância
  while (inicio_cel <= n_cells) {
    fim_cel <- min(inicio_cel + batch_cells - 1, n_cells)
    cat(sprintf("  -> [Lote %02d] Processando células %d a %d...\n", lote_id, inicio_cel, fim_cel))
    
    # Filtra apenas as linhas do lote atual diretamente em C via awk (como o .mtx é ordenado por linha,
    # o 'exit' interrompe a leitura imediatamente ao passar de fim_cel!)
    cmd_awk <- sprintf(
      "awk 'NR > 2 { if ($1 >= %d && $1 <= %d) print $1, $2; else if ($1 > %d) exit }' %s",
      inicio_cel, fim_cel, fim_cel, shQuote(caminho_mtx)
    )
    
    dt_coo <- fread(cmd = cmd_awk, col.names = c("r", "c"), showProgress = FALSE)
    n_lote <- fim_cel - inicio_cel + 1
    
    # Monta a matriz esparsa APENAS deste lote (25k linhas)
    mtx_lote <- sparseMatrix(
      i = dt_coo$r - inicio_cel + 1L,
      j = dt_coo$c,
      x = 1.0,
      dims = c(n_lote, nfeat)
    )
    rm(dt_coo); gc(verbose = FALSE)
    
    # Projeção SWeeP do lote (n_lote x 600)
    sw_lote <- as.matrix(mtx_lote %*% base160k)
    rm(mtx_lote); gc(verbose = FALSE)
    
    # Atualiza acumuladores do PCA
    soma_colunas <- soma_colunas + colSums(sw_lote)
    XtX          <- XtX + crossprod(sw_lote)
    
    # Prepara data.table com 600 colunas + Cell_Type e grava em streaming no disco (para a Hopfield da Letícia)
    colnames(sw_lote) <- paste0("SW", seq_len(psz))
    dt_out <- as.data.table(sw_lote)
    dt_out[, Cell_Type := rotulos$Cell_Type_Num[inicio_cel:fim_cel]]
    
    fwrite(dt_out, file = arq_saida_sweep, sep = "\t", append = (lote_id > 1), col.names = (lote_id == 1))
    
    rm(sw_lote, dt_out); gc(verbose = FALSE)
    inicio_cel <- fim_cel + 1
    lote_id    <- lote_id + 1
  }
  
  cat("  -> [OK] Arquivo SWeeP (600 colunas + Cell_Type) da Letícia finalizado em:\n     ", arq_saida_sweep, "\n")
  
  # CÁLCULO DO PCA EXATO A PARTIR DA MATRIZ DE COVARIÂNCIA (600 x 600)
  cat("  -> Calculando autovetores do PCA (600 x 600) sem sobrecarregar a RAM...\n")
  media_colunas <- soma_colunas / n_cells
  # Fórmula da covariância amostral centrada: (X^T X - N * mu * mu^T) / (N - 1)
  cov_mat <- (XtX - n_cells * tcrossprod(media_colunas)) / (n_cells - 1)
  eig     <- eigen(cov_mat, symmetric = TRUE)
  
  variancia      <- pmax(eig$values, 0)
  prop_variancia <- (variancia / sum(variancia)) * 100
  rotation_50    <- eig$vectors[, 1:50, drop = FALSE]
  colnames(rotation_50) <- paste0("PC", 1:50)
  
  # PASSADA 2 (Muito rápida): Lê o arquivo SWeeP600 recém-criado em lotes e projeta nos 50 PCs
  cat("  -> Projetando os 50 primeiros PCs em lotes a partir do arquivo SWeeP600...\n")
  if (file.exists(arq_saida_pca50)) file.remove(arq_saida_pca50)
  
  lista_pcs_plot <- if (retornar_pca_completo) list() else NULL
  inicio_cel <- 1
  lote_id    <- 1
  
  while (inicio_cel <= n_cells) {
    fim_cel <- min(inicio_cel + batch_cells - 1, n_cells)
    n_lote  <- fim_cel - inicio_cel + 1
    
    # Lê apenas as linhas do lote no TSV de 600 colunas (pula o cabeçalho de 1 linha)
    dt_sw <- fread(arq_saida_sweep, skip = inicio_cel, nrows = n_lote, header = FALSE, showProgress = FALSE)
    rot_lote <- dt_sw[[601]]
    mat_sw   <- as.matrix(dt_sw[, 1:600])
    rm(dt_sw)
    
    # Centraliza pela média global e multiplica pelos 50 autovetores
    mat_sw_cent <- sweep(mat_sw, 2, media_colunas, "-")
    pcs_lote    <- mat_sw_cent %*% rotation_50
    rm(mat_sw, mat_sw_cent)
    
    dt_pca <- as.data.table(pcs_lote)
    dt_pca[, Cell_Type := rot_lote]
    fwrite(dt_pca, file = arq_saida_pca50, sep = "\t", append = (lote_id > 1), col.names = (lote_id == 1))
    
    if (retornar_pca_completo) {
      # Guarda apenas os 30 primeiros PCs na RAM para os gráficos e UMAP (~85 MB no total!)
      lista_pcs_plot[[lote_id]] <- pcs_lote[, 1:30, drop = FALSE]
    }
    
    rm(pcs_lote, dt_pca); gc(verbose = FALSE)
    inicio_cel <- fim_cel + 1
    lote_id    <- lote_id + 1
  }
  
  cat("  -> [OK] Arquivo PCA50 salvo em:\n     ", arq_saida_pca50, "\n")
  
  if (retornar_pca_completo) {
    pcs_30_full <- do.call(rbind, lista_pcs_plot)
    rm(lista_pcs_plot); gc(verbose = FALSE)
    return(list(x30 = pcs_30_full, prop_variancia = prop_variancia, rotulos = rotulos$Cell_Type_Num))
  }
  return(NULL)
}

# ============================================================================
# 3. EXECUÇÃO MATRIZ 1: HOMÓLOGO (82.963 células)
# ============================================================================
cat("\n[2/5] Processando Matriz 1: Homólogo (All Nuclei + 10X-Chr-V3)...\n")
processar_mtx_em_lotes(
  caminho_mtx           = file.path(pasta_in, "Matriz_GSE263468_Homologo_10XV3_AllNuclei_Binarizada.mtx"),
  caminho_rot           = file.path(pasta_in, "Rotulos_GSE263468_Homologo_10XV3_AllNuclei.csv"),
  arq_saida_sweep       = file.path(pasta_out, "SWeeP600_GSE263468_Homologo_Hopfield.tsv"),
  arq_saida_pca50       = file.path(pasta_out, "PCA50_GSE263468_Homologo_MLP.txt"),
  batch_cells           = 25000,
  retornar_pca_completo = FALSE
)

# ============================================================================
# 4. EXECUÇÃO MATRIZ 2: GERAL COMPLETO (372.908 células)
# ============================================================================
cat("\n[3/5] Processando Matriz 2: Geral Completo (372.908 células) em lotes...\n")
res_geral <- processar_mtx_em_lotes(
  caminho_mtx           = file.path(pasta_in, "Matriz_GSE263468_Geral_SemVazamento_Binarizada.mtx"),
  caminho_rot           = file.path(pasta_in, "Rotulos_GSE263468_Geral_SemVazamento.csv"),
  arq_saida_sweep       = file.path(pasta_out, "SWeeP600_GSE263468_Geral_Hopfield.tsv"),
  arq_saida_pca50       = file.path(pasta_out, "PCA50_GSE263468_Geral_MLP.txt"),
  batch_cells           = 25000,
  retornar_pca_completo = TRUE
)

# Libera a base160k pois não é mais necessária nos gráficos
rm(base160k); gc()

# ============================================================================
# 5. PLOTAGEM DOS GRÁFICOS (SOMENTE MATRIZ GERAL)
# ============================================================================
cat("\n[4/5] Gerando gráficos de PCA e Scree Plot da Matriz Geral...\n")

tipos_celulares <- factor(
  res_geral$rotulos,
  levels = 1:8,
  labels = nomes_classes[as.character(1:8)]
)

paleta       <- palette.colors(n = 8, palette = "Tableau 10")
paleta_alpha <- adjustcolor(paleta, alpha.f = 0.4)
cores_plot   <- paleta_alpha[as.numeric(tipos_celulares)]

set.seed(42)
ordem_plot <- sample(nrow(res_geral$x30))

# --- 5.1 PCA 1 e 2 ---
png(filename = file.path(pasta_out, "Pca1-2_Geral.png"), width = 950, height = 650, res = 120)
par(mar = c(5, 4, 4, 11), xpd = TRUE)
plot(
  res_geral$x30[ordem_plot, 1], res_geral$x30[ordem_plot, 2],
  xlab = "PC-1", ylab = "PC-2",
  main = "PCA (PC1 vs PC2) - Matriz SWeeP Geral (GSE263468)",
  pch = 16, cex = 0.35, col = cores_plot[ordem_plot]
)
legend(
  "topright", inset = c(-0.28, 0),
  legend = levels(tipos_celulares),
  col = paleta, pch = 15, cex = 1.0, bty = "n", title = "Tipos Celulares"
)
dev.off()

# --- 5.2 PCA 1 e 3 ---
png(filename = file.path(pasta_out, "Pca1-3_Geral.png"), width = 950, height = 650, res = 120)
par(mar = c(5, 4, 4, 11), xpd = TRUE)
plot(
  res_geral$x30[ordem_plot, 1], res_geral$x30[ordem_plot, 3],
  xlab = "PC-1", ylab = "PC-3",
  main = "PCA (PC1 vs PC3) - Matriz SWeeP Geral (GSE263468)",
  pch = 16, cex = 0.35, col = cores_plot[ordem_plot]
)
legend(
  "topright", inset = c(-0.28, 0),
  legend = levels(tipos_celulares),
  col = paleta, pch = 15, cex = 1.0, bty = "n", title = "Tipos Celulares"
)
dev.off()

# --- 5.3 Scree Plot ---
cat("\nProporção de variância explicada (20 primeiros PCs - Geral):\n")
print(round(res_geral$prop_variancia[1:20], 2))

png(filename = file.path(pasta_out, "Scree_plot_Geral.png"), width = 800, height = 600, res = 120)
par(mar = c(5, 4, 4, 2) + 0.1, xpd = FALSE)
plot(
  res_geral$prop_variancia[1:30],
  xlab = "Componente Principal (PC)",
  ylab = "Proporção da Variância Explicada (%)",
  type = "b", pch = 16, col = "#1f77b4", lwd = 2,
  main = "Scree Plot - Matriz SWeeP Geral (GSE263468)"
)
grid(nx = NULL, ny = NULL, lty = "dotted")
dev.off()
gc()

# --- 5.4 UMAP (sobre os 30 primeiros PCs da Matriz Geral) ---
cat("\n[5/5] Calculando UMAP sobre os 30 primeiros PCs da Matriz Geral...\n")
set.seed(42)
# n_trees = 20 e n_epochs = 200 economizam bastante memória em datasets com >300k pontos
umap_resultado <- umap(
  res_geral$x30,
  n_threads = max(1, parallel::detectCores() - 1),
  n_trees = 20,
  verbose = TRUE
)

png(filename = file.path(pasta_out, "UMap_Geral_GSE263468.png"), width = 950, height = 650, res = 120)
par(mar = c(5, 4, 4, 11), xpd = TRUE)
plot(
  umap_resultado[ordem_plot, 1], umap_resultado[ordem_plot, 2],
  xlab = "UMAP 1", ylab = "UMAP 2",
  main = "UMAP da Matriz SWeeP Geral (GSE263468)",
  pch = 16, cex = 0.35, col = cores_plot[ordem_plot]
)
legend(
  "topright", inset = c(-0.28, 0),
  legend = levels(tipos_celulares),
  col = paleta, pch = 15, cex = 1.0, bty = "n", title = "Tipos Celulares"
)
dev.off()

cat("\n======================================================================\n")
cat("PROCESSAMENTO EM LOTES CONCLUÍDO SEM ESTOURAR RAM!\n")
cat("Arquivos salvos em:", pasta_out, "\n")
cat("======================================================================\n")