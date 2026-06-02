
#------------------------------CHECAGEM 5K FUJITA ---------------------------------


listafujita_limpa = []
genes_oficial= []
genes_fujita = pd.read_csv("Documents/BIOINFO/analise.pan_etal/features.tsv.gz")

genes_fujita.to_csv("Features_leticia", index = False)
print(genes_fujita[:10]) 

#LISTAFUJITA
listafujita = genes_fujita.iloc[:, 0].tolist()

#LISTAFUJITA LIMPA
for gene in listafujita:
    gene_limpo = gene.split('\t')[0]                
    listafujita_limpa.append(gene_limpo)

#SALVANDO CODIGOS ORIGINAIS
codigos_original = []
for gene in adata.var['index1']:
    codigos_original.append(gene)

print(codigos_original[:10])
#ESCREVENDO O TXT DOS CÓDIGOS ORIGINAIS
with open("códigos_originais_Pan.txt", "w") as f:
    for gene in codigos_original: 
        f.write(f"{gene}\n")

#REESCREVENDO A COLUNA INDEX1

adata.var['index1'] = adata.var['index1'].str.split('.').str[0]
print(adata.var['index1'].values[:10])

for gene in listafujita_limpa:
    if gene in adata.var['index1']:
        genes_oficial.append(gene)

print(genes_oficial[:10])
len(genes_oficial)

genes5k = adata.var.nlargest(5000, 'total_counts')['index1']

genes5K = adata.var.sort_values(by='total_counts', ascending = False)['index1'].head(5000)

len(genes5k)

top5000_var['total_counts'].describe()

#MONTANDO TXT COM OS 5K MAIS FREQ E AS COUNTS TOTAIS

top5000_var = adata.var.nlargest(5000, 'total_counts')
top5000_var[['index1', 'total_counts']].to_csv("5kfujita-Pan", sep="\t", index=False)


tem_decimais = (adata.var['total_counts'] % 1 != 0).any()
print(tem_decimais)

