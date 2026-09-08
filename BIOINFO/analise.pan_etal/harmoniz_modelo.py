import torch as torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

matriz = pd.read_csv("W50pc.dat", sep=r'\s+', engine='python')
matriz = pd.read_csv("PCA50_Classes_Alzheimer.txt", sep=r'\s+', engine='python')
celulas_tipo_2 = (matriz['Cell_Type'] == 2).sum()
celulas_tipo_3 = (matriz['Cell_Type'] == 3).sum()
celulas_tipo_3
celulas_tipo_1 = (matriz['Cell_Type'] == 1).sum()
celulas_tipo_1
celulas_tipo_8 = (matriz['Cell_Type'] == 8).sum()
celulas_tipo_8


features_num = matriz.iloc[:, 1:51].values
matriz.head
matriz.shape
print(matriz.columns)
matriz.describe()
matriz1 = torch.tensor(features_num, dtype = torch.float32)

print(matriz.head())

X_features_bruto = matriz.drop(columns=['Cell_Type']).values  # Pega os PCs
y_labels_bruto = matriz['Cell_Type'].values

X_features = scaler.fit_transform(X_features_bruto)
y_labels = y_labels_bruto - 1


pesos_calculados = compute_class_weight(
    class_weight='balanced', #ajusta os pesos inversamente proporcionais à frequência das classes
    classes=np.unique(y_labels), 
    y=y_labels
)
pesos_tensor = torch.tensor(pesos_calculados, dtype=torch.float32) #PyTorch exige que os pesos sejam tensores de ponto flutuante (
funcao_erro = nn.CrossEntropyLoss(weight=pesos_tensor)

print(f"Formato do X: {X_features.shape}") # Deve ser (58325, 50)
print(f"Formato do y: {y_labels.shape}")

X_tensor = torch.tensor(X_features, dtype=torch.float32)
y_tensor = torch.tensor(y_labels, dtype=torch.long)

print(f"Formato do X: {X_tensor.shape}")
print(f"Formato do y: {y_tensor.shape}")


dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

#Arquitetura da rede neural(MLP)
class ClassificadorCelular(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # Para 50 PCs, usar 2 camadas ocultas ajuda a isolar padrões biológicos complexos
        self.camadas = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.camadas(x)

modelo = ClassificadorCelular(input_dim=50, num_classes=8)
funcao_erro = nn.CrossEntropyLoss(weight=pesos_tensor)

# Otimizador alterar a matriz W
otimizador = torch.optim.SGD(modelo.parameters(), lr=0.001)

epochs = 100
print("Iniciando o treinamento (acompanhe o erro caindo)...\n")
for epoca in range(epochs):
    modelo.train()
    erro_acumulado = 0.0
    
    # O loop interno passa as 58 mil células aos poucos (256 por vez)
    for lote_X, lote_y in loader:
        
        # Previsão
        previsoes = modelo(lote_X)
        erro = funcao_erro(previsoes, lote_y)
        
        # Atualização dos pesos
        otimizador.zero_grad()
        erro.backward()
        otimizador.step()
        
        erro_acumulado += erro.item()
        
    # Calcula a média do erro no final da época
    erro_medio = erro_acumulado / len(loader)
    print(f"Época {epoca+1:02d}/{epochs} | Erro matemático médio (Loss): {erro_medio:.4f}")




    #================++#+========================++#+=================++#+======================++++#

print("\n--- AVALIAÇÃO DO MODELO ---")

# 1. Congelar o modelo (desativa dropout e layers de treino)
modelo.eval()    
    # 2. Desabilitar o cálculo de gradientes (economiza RAM e acelera a inferência)
with torch.no_grad():
    # Passamos os tensores pela rede para obter as previsões finais
    saidas_finais = modelo(X_tensor)
    
    # A saída retorna as probabilidades para as 8 classes. Pegamos o índice com maior valor (argmax)
    _, previsoes = torch.max(saidas_finais, 1)

# Converte de Tensors do PyTorch de volta para Arrays do NumPy para usar no scikit-learn
y_real = y_tensor.numpy()
y_previsto = previsoes.numpy()

# 3. Acurácia Global
acuracia = np.mean(y_previsto == y_real) * 100
print(f"Acurácia Geral: {acuracia:.2f}%\n")

# 4. Relatório Detalhado por Classe
# Como subtraímos 1 lá no começo, as classes estão 0 a 7. Somamos 1 aqui apenas para o rótulo do print.
nomes_classes = [f"Classe {i+1}" for i in range(8)]
print(classification_report(y_real, y_previsto, target_names=nomes_classes))

# 5. Plotando a Matriz de Confusão
matriz_conf = confusion_matrix(y_real, y_previsto)

plt.figure(figsize=(10, 8))
sns.heatmap(matriz_conf, annot=True, fmt='d', cmap='Blues', 
            xticklabels=nomes_classes, 
            yticklabels=nomes_classes)

plt.xlabel('Previsão do Modelo')
plt.ylabel('Classe Real (Biológica)')
plt.title('Matriz de Confusão - Identificação Celular')
plt.tight_layout()
plt.savefig("matriz_confusao_SWeeP.png", dpi=300)
print("\nGráfico salvo como 'matriz_confusao_SWeeP.png'.")

