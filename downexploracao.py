import scanpy as sc
import tarfile
import os

pasta_base = "/media/gustavo-k-sato/D61CF48E1CF46B3F1/Analise-RNA-seq-Pan_et_al---alzheimer/Documents/dataset_down_syndrome/"

# ==========================================
# 1. ALVO EXATO: Coloque o nome do arquivo aqui
# ==========================================
arquivo_exato = "GSM9162706_S379_counts_matrix.tar.gz" 

caminho_targz = os.path.join(pasta_base, arquivo_exato)
nome_amostra = arquivo_exato.replace(".tar.gz", "")
pasta_amostra = os.path.join(pasta_base, nome_amostra)

# ==========================================
# 2. EXTRAÇÃO DIRETA
# ==========================================
if not os.path.exists(pasta_amostra):
    os.makedirs(pasta_amostra)
    print(f"Extraindo {arquivo_exato} para a pasta {nome_amostra}...")
    with tarfile.open(caminho_targz, "r:gz") as tar:
        tar.extractall(path=pasta_amostra)
else:
    print(f"Pasta da amostra {nome_amostra} já existe. Pulando extração.")

# ==========================================
# 3. LEITURA INTELIGENTE
# ==========================================
# Varre a pasta recém-extraída para descobrir os nomes reais que os autores usaram
arquivos_extraidos = os.listdir(pasta_amostra)
prefixo = ""
for arq in arquivos_extraidos:
    if "matrix.mtx" in arq and arq not in ["matrix.mtx", "matrix.mtx.gz"]:
        prefixo = arq.split("matrix.mtx")[0]
        break

print(f"Carregando matriz no Scanpy (Prefixo detectado: '{prefixo}')...")
adata = sc.read_10x_mtx(
    pasta_amostra, 
    prefix=prefixo,
    var_names='gene_symbols', 
    cache=True
)

# ==========================================
# 4. RESULTADO
# ==========================================
print("\n" + "="*40)
print(f" RADIOGRAFIA DA AMOSTRA: {arquivo_exato}")
print("="*40)
print(f"-> Formato (Células x Genes): {adata.shape}")
print("\n[Genes - adata.var]:")
print(adata.var.head(3))
print(adata.var.columns)
print(adata.obs.columns)