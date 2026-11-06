import scanpy as sc
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. ARQUITETURA DA REDE
# ==========================================
class ClassificadorCelular(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.camadas = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.camadas(x)

input_dim = 600
num_classes = 8
nomes_classes = ['Astrocytes', 'Oligodendrocytes', 'Exc. Neurons', 'Inh. Neurons', 
                 'OPCs', 'Microglia', 'Endothelial', 'VLMCs/Pericytes']

# ==========================================
# 2. EXTRAINDO O GABARITO DIRETO DO .h5ad
# ==========================================
print("[INFO] Extraindo Gabarito do Matisse (.h5ad)...")
adata_matisse = sc.read_h5ad("matriz_anotada_finalM.h5ad", backed='r')
NOME_DA_COLUNA = 'celltype'

rotulos_texto = adata_matisse.obs[NOME_DA_COLUNA].astype(str)

# Dicionário atualizado para a taxonomia do Matisse
dicionario_map = {
    'Ast': 0, # Astrocytes
    'Oli': 1, # Oligodendrocytes
    'Ex': 2,  # Exc. Neurons
    'In': 3,  # Inh. Neurons
    'Opc': 4, # OPCs
    'Mic': 5, # Microglia
    'End': 6, # Endothelial
    'Per': 7  # Pericytes (Equivalente ao VLMCs)
    # A string 'nan' ficará de fora de propósito para ser ignorada no cálculo
}

print(" -> Mapeando rótulos biológicos para as chaves numéricas do PyTorch...")
y_true = rotulos_texto.map(dicionario_map).values

# ==========================================
# 3. CARREGANDO MODELO E SCALER
# ==========================================
print("\n[INFO] Carregando Modelo Ouro e Scaler do PAN...")
modelo = ClassificadorCelular(input_dim, num_classes)
modelo.load_state_dict(torch.load("pesos_mlp_sweep_pan_ouro.pth", weights_only=True))
modelo.eval()  

scaler_ouro = joblib.load("scaler_sweep_pan_ouro.pkl")

# ==========================================
# 4. LENDO DADOS DO MATISSE E PADRONIZANDO
# ==========================================
arquivo_matisse_sweep = "SWeeP_600_Matisse_Python.txt"
print(f"[INFO] Carregando a matriz projetada: {arquivo_matisse_sweep}")
matriz_X = pd.read_csv(arquivo_matisse_sweep, sep='\t')
X_raw = matriz_X.values

if len(X_raw) != len(y_true):
    print(f"\n[ERRO] A matriz SWeeP tem {len(X_raw)} células, mas o gabarito tem {len(y_true)}.")
    exit()

print("[INFO] Padronizando com as médias do PAN (Scaler original)...")
X_escalado = scaler_ouro.transform(X_raw)
X_tensor = torch.tensor(X_escalado, dtype=torch.float32)
dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=2048, shuffle=False)

# ==========================================
# 5. INFERÊNCIA CRUZADA
# ==========================================
print("\n[INFO] Iniciando diagnóstico celular (rede no modo Juiz)...")
previsoes_finais = []

with torch.no_grad():
    for lote_X in loader:
        saida = modelo(lote_X[0])
        _, previsao = torch.max(saida, 1)
        previsoes_finais.extend(previsao.numpy())

# ==========================================
# 6. AVALIAÇÃO ACADÊMICA: MÉTRICAS E GRÁFICOS
# ==========================================
print("\n" + "="*50)
print(" RELATÓRIO DE DESEMPENHO: PAN -> MATISSE ")
print("="*50)

# Isolando apenas as células que o dicionário conseguiu mapear (Removendo os 'nan')
mask_validos = ~pd.isna(y_true)
y_true_limpo = y_true[mask_validos].astype(int)
previsoes_limpas = np.array(previsoes_finais)[mask_validos]

acc = accuracy_score(y_true_limpo, previsoes_limpas)
f1_macro = f1_score(y_true_limpo, previsoes_limpas, average='macro')
f1_weighted = f1_score(y_true_limpo, previsoes_limpas, average='weighted')

print(f" -> Células avaliadas (excluindo NaN): {len(y_true_limpo)}")
print(f" -> Acurácia Global Cross-Dataset: {acc:.4f} ({acc*100:.2f}%)")
print(f" -> F1-Score (Macro): {f1_macro:.4f}")
print(f" -> F1-Score (Weighted): {f1_weighted:.4f}")

print("\n[INFO] Relatório por Tipo Celular:")
print(classification_report(y_true_limpo, previsoes_limpas, target_names=nomes_classes))

print("\n[INFO] Desenhando e salvando a Matriz de Confusão...")
cm = confusion_matrix(y_true_limpo, previsoes_limpas)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=nomes_classes, yticklabels=nomes_classes)
plt.title('Matriz de Confusão: Treino PAN vs Teste Matisse', fontsize=14)
plt.ylabel('Verdadeiro Biológico (Gabarito Matisse)', fontsize=12)
plt.xlabel('Previsão do Modelo (Rede Ouro PAN)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

nome_grafico = 'Matriz_Confusao_Matisse_Real.png'
plt.savefig(nome_grafico, dpi=300)
print(f"[SUCESSO] Gráfico salvo como '{nome_grafico}' na sua pasta!")