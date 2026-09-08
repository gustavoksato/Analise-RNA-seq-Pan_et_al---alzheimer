import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # BIBLIOTECA ADICIONADA AQUI
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

# 1. CARREGAMENTO DOS DADOS (SWeeP Direto)
print("[INFO] Carregando a matriz SWeeP (600 Dimensões)...")
# Lê o arquivo exportado diretamente do R
matriz = pd.read_csv("SWeeP_600_Classes_Alzheimer.txt", sep='\t')

# Isola os 600 vetores matemáticos e o alvo biológico
X_raw = matriz.drop(columns=['Cell_Type']).values  
y_raw = matriz['Cell_Type'].values - 1  # Ajuste para base 0 (0 a 7) para o PyTorch

input_dim = X_raw.shape[1]
nomes_classes = [f"Classe {i+1}" for i in range(8)]

print(f"-> Dimensão de Entrada: {input_dim} features SWeeP.")

# 2. ARQUITETURA DA REDE NEURAL MLP
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

# 3. DIVISÃO E PADRONIZAÇÃO (80/20 Estratificado)
X_treino_bruto, X_val_bruto, y_treino, y_val = train_test_split(
    X_raw, y_raw, test_size=0.20, random_state=42, stratify=y_raw
)

scaler = StandardScaler()
X_treino_escalado = scaler.fit_transform(X_treino_bruto)
X_val_escalado = scaler.transform(X_val_bruto)

# =====================================================================
# SALVAMENTO DO SCALER PARA EVITAR DATA LEAKAGE NA INFERÊNCIA
# =====================================================================
caminho_scaler = "scaler_sweep_pan_ouro.pkl"
joblib.dump(scaler, caminho_scaler)
print(f"[SUCESSO] Scaler original salvo no disco em: {caminho_scaler}")
# =====================================================================

# 4. PESOS BALANCEADOS E TENSORES
pesos = compute_class_weight(class_weight='balanced', classes=np.unique(y_treino), y=y_treino)
pesos_tensor = torch.tensor(pesos, dtype=torch.float32)

dataset_treino = TensorDataset(torch.tensor(X_treino_escalado, dtype=torch.float32), torch.tensor(y_treino, dtype=torch.long))
loader_treino = DataLoader(dataset_treino, batch_size=256, shuffle=True)

dataset_val = TensorDataset(torch.tensor(X_val_escalado, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
loader_val = DataLoader(dataset_val, batch_size=512, shuffle=False)

# 5. TREINAMENTO COM EARLY STOPPING
funcao_erro = nn.CrossEntropyLoss(weight=pesos_tensor)
modelo = ClassificadorCelular(input_dim=input_dim, num_classes=8)
otimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)

epochs = 100
historico_erro_treino, historico_erro_val = [], []
melhor_erro_val = float('inf')
melhor_estado_modelo = None
epoca_ideal = 0

print("\n[INFO] Iniciando Treinamento Ouro (PAN Dataset)...")

for epoca in range(epochs):
    modelo.train()
    erro_treino = 0.0
    for lote_X, lote_y in loader_treino:
        previsoes = modelo(lote_X)
        erro = funcao_erro(previsoes, lote_y)
        otimizador.zero_grad()
        erro.backward()
        otimizador.step()
        erro_treino += erro.item() * lote_X.size(0)
    
    erro_medio_treino = erro_treino / len(loader_treino.dataset)
    historico_erro_treino.append(erro_medio_treino)
    
    modelo.eval()
    erro_val = 0.0
    with torch.no_grad():
        for lote_X_val, lote_y_val in loader_val:
            previsoes_val = modelo(lote_X_val)
            erro_val += funcao_erro(previsoes_val, lote_y_val).item() * lote_X_val.size(0)
            
    erro_medio_val = erro_val / len(loader_val.dataset)
    historico_erro_val.append(erro_medio_val)

    if erro_medio_val < melhor_erro_val:
        melhor_erro_val = erro_medio_val
        melhor_estado_modelo = modelo.state_dict().copy()
        epoca_ideal = epoca

    if (epoca + 1) % 10 == 0 or epoca == 0 or epoca == (epochs - 1):
        print(f" -> Época {epoca+1:03d}/{epochs} | Erro Treino: {erro_medio_treino:.4f} | Erro Validação: {erro_medio_val:.4f}")

# 6. CONGELAMENTO E SALVAMENTO DOS PESOS
modelo.load_state_dict(melhor_estado_modelo)
modelo.eval()

caminho_pesos = "pesos_mlp_sweep_pan_ouro.pth"
torch.save(modelo.state_dict(), caminho_pesos)
print(f"\n[SUCESSO] Pesos ideais salvos em: {caminho_pesos}")

# 7. AVALIAÇÃO FINAL
todas_previsoes = []
with torch.no_grad():
    for lote_X_val, _ in loader_val:
        saida = modelo(lote_X_val)
        _, previsao = torch.max(saida, 1)
        todas_previsoes.extend(previsao.numpy())

acc = accuracy_score(y_val, todas_previsoes)
f1_weighted = f1_score(y_val, todas_previsoes, average='weighted')

print(f"\n--- PERFORMANCE OURO (Fase 1) ---")
print(f"Acurácia Global: {acc:.4f}")
print(f"F1-Score (Weighted): {f1_weighted:.4f}")
print(classification_report(y_val, todas_previsoes, target_names=nomes_classes))