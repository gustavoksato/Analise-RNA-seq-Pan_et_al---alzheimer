library(rSWeeP)
library(Matrix)
install.packages("uwot")
library(uwot)

# Lê o arquivo exportado 
matriz_esparsa <- readMM("Matriz_Pan.mtx")
summary(matriz_esparsa)

matriz_esparsa <- as(matriz_esparsa, "CsparseMatrix")
read.table(matriz_esparsa)

psz <- 600#tamanho
nfeat <- ncol(matriz_esparsa)
nsamples <- 5

resultados_sweep <- list
base160k = orthBase(nfeat,psz)

summary(pca_output)

#sw = sWeeP(as.matrix(matriz_esparsa), base160k)

sw <- matriz_esparsa %*% base160k

matriz_densa <- as.matrix(sw)
pca_output <- prcomp (matriz_densa, scale = FALSE)

par(mar = c(1,2))
par(xpd = TRUE)
print(pca_output)

hcl.pals()

pca_output <- prcomp (matriz_esparsa, scale = FALSE)


gc()
#VENDO OS TIPOS CELULARES PARA COLORIR
anotacoes <- read.table("anotacoes_tecidos.txt", sep='\t', header = TRUE, row.names = 1)
tipos_celulares <- as.factor(anotacoes$manual_annotation)
num_tipos <- length(levels(tipos_celulares))
paleta <- palette.colors(n = num_tipos, palette = "Tableau 10")
cores_plot <- paleta[as.numeric(tipos_celulares)]

#============PRINTANDO O UMAP================================
umap_resultado <- umap(pca_output$x[, 1:30])
dev.off()
png(filename= "UMap_alzheimer.png", width = 800, height = 600)
plot(umap_resultado[,1], umap_resultado[,2], 
     xlab = "UMAP1", ylab = "UMAP2",
     main = "UMAP da Matriz SWeeP",
     pch = 16, 
     cex = 0.5,
     col = cores_plot)
legend("topright", 
       legend = levels(tipos_celulares), 
       col = paleta,          # Voltamos para o col
       pch = 15,              # pch 15 = quadrado sólido
       cex = 1.3,             
       bty = "n",
       x.intersp = 0.5)
dev.off()
gc()
par(mar = c(5, 4, 4, 2) + 0.1, xpd = FALSE)


#PRINTANDO O PCA 1 E 3========================================================
dev.off()
png(filename= "Pca1-3.png", width = 800, height = 600)
par(mar = c(5, 4, 4, 12), xpd = TRUE)
# 2. Desenhar os pontos
plot(pca_output$x[,1], pca_output$x[,3], 
     xlab = "PC-1", ylab = "PC-3",
     main = "PCA da Matriz SWeeP",
     pch = 16, cex = 0.5, col = cores_plot)
legend("bottomleft", 
       legend = levels(tipos_celulares), 
       col = paleta,          # Voltamos para o col
       pch = 15,              # pch 15 = quadrado sólido
       cex = 1.3,             
       bty = "n",
       x.intersp = 0.5)
dev.off()
gc()
par(mar = c(5, 4, 4, 2) + 0.1, xpd = FALSE)

#PRINTANDO O PCA 1 E 2===========================================

dev.off()
png(filename= "Pca1-2.png", width = 800, height = 600)
par(mar = c(5, 4, 4, 12), xpd = TRUE)
# 2. Desenhar os pontos
plot(pca_output$x[,1], pca_output$x[,2], 
     xlab = "PC-1", ylab = "PC-2",
     main = "PCA da Matriz SWeeP-1e2",
     pch = 16, cex = 0.5, col = cores_plot)
legend("bottomleft", 
       legend = levels(tipos_celulares), 
       col = paleta,          # Voltamos para o col
       pch = 15,              # pch 15 = quadrado sólido
       cex = 1.3,             
       bty = "n",
       x.intersp = 0.5)
dev.off()
gc()
par(mar = c(5, 4, 4, 2) + 0.1, xpd = FALSE)


#=====PRINTANDO O SCREEPLOT================
variancia <- pca_output$sdev^2

prop_variancia <- (variancia / sum(variancia)) * 100

print(round(prop_variancia[1:20], 2))

dev.off()
png(filename= "Scree_plot.png", width = 800, height = 600)
par(mar = c(5, 4, 4, 12), xpd = TRUE)
plot(prop_variancia[1:30],
     xlab = "Componente principal", 
     ylab = "Proporção da variância explicada", 
     type = "b", pch = 16,
     main = "Scree plot")
dev.off()
gc()
