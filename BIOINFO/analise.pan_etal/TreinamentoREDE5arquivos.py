import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, f1_score

# ==========================================
# ARQUITETURA DA REDE NEURAL MLP
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

# ==========================================
# CONFIGURAÇÕES DO LOOP DE EXPERIMENTAÇÃO
# ==========================================
niveis_ruido = ['1k', '2k', '3k', '4k', '5k']
historico_acc = []
historico_f1 = []

print("[INÍCIO] Iniciando Bateria de Testes de Degradação (1k a 5k)...")

for nivel in niveis_ruido:
    print(f"\n{'='*50}")
    print(f" PROCESSANDO RUÍDO: {nivel} GENES")
    print(f"{'='*50}")
    
    arquivo_entrada = f"SWeeP_600_Classes_Alzheimer_Ruido_{nivel}.txt"
    caminho_pesos = f"pesos_mlp_sweep_pan_ruido_{nivel}.pth"
    
    # 1. CARREGAMENTO DOS DADOS
    print(f"[INFO] Lendo {arquivo_entrada}...")
    matriz = pd.read_csv(arquivo_entrada, sep='\t')

    X_raw = matriz.drop(columns=['Cell_Type']).values  
    y_raw = matriz['Cell_Type'].values - 1  # Base 0 para o PyTorch
    input_dim = X_raw.shape[1]
    nomes_classes = [f"Classe {i+1}" for i in range(8)]

    # 2. DIVISÃO E PADRONIZAÇÃO
    X_treino_bruto, X_val_bruto, y_treino, y_val = train_test_split(
        X_raw, y_raw, test_size=0.20, random_state=42, stratify=y_raw
    )

    scaler = StandardScaler()
    X_treino_escalado = scaler.fit_transform(X_treino_bruto)
    X_val_escalado = scaler.transform(X_val_bruto)

    # 3. PESOS BALANCEADOS E TENSORES
    pesos = compute_class_weight(class_weight='balanced', classes=np.unique(y_treino), y=y_treino)
    pesos_tensor = torch.tensor(pesos, dtype=torch.float32)

    dataset_treino = TensorDataset(torch.tensor(X_treino_escalado, dtype=torch.float32), torch.tensor(y_treino, dtype=torch.long))
    loader_treino = DataLoader(dataset_treino, batch_size=256, shuffle=True)

    dataset_val = TensorDataset(torch.tensor(X_val_escalado, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    loader_val = DataLoader(dataset_val, batch_size=512, shuffle=False)

    # 4. TREINAMENTO
    funcao_erro = nn.CrossEntropyLoss(weight=pesos_tensor)
    modelo = ClassificadorCelular(input_dim=input_dim, num_classes=8)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)

    epochs = 100
    melhor_erro_val = float('inf')
    melhor_estado_modelo = None

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
        
        modelo.eval()
        erro_val = 0.0
        with torch.no_grad():
            for lote_X_val, lote_y_val in loader_val:
                previsoes_val = modelo(lote_X_val)
                erro_val += funcao_erro(previsoes_val, lote_y_val).item() * lote_X_val.size(0)
                
        erro_medio_val = erro_val / len(loader_val.dataset)

        if erro_medio_val < melhor_erro_val:
            melhor_erro_val = erro_medio_val
            melhor_estado_modelo = modelo.state_dict().copy()

    # 5. CONGELAMENTO E SALVAMENTO
    modelo.load_state_dict(melhor_estado_modelo)
    modelo.eval()
    torch.save(modelo.state_dict(), caminho_pesos)
    print(f"[SUCESSO] Pesos salvos em: {caminho_pesos}")

    # 6. AVALIAÇÃO DA RODADA
    todas_previsoes = []
    with torch.no_grad():
        for lote_X_val, _ in loader_val:
            saida = modelo(lote_X_val)
            _, previsao = torch.max(saida, 1)
            todas_previsoes.extend(previsao.numpy())

    acc = accuracy_score(y_val, todas_previsoes)
    f1_weighted = f1_score(y_val, todas_previsoes, average='weighted')
    
    # Armazena as métricas para o gráfico final
    historico_acc.append(acc)
    historico_f1.append(f1_weighted)

    print(f"-> Acurácia Global ({nivel}): {acc:.4f}")
    print(f"-> F1-Score ({nivel}): {f1_weighted:.4f}")

# ==========================================
# PLOTAGEM DA CURVA DE DEGRADAÇÃO
# ==========================================
print("\n[INFO] Gerando Gráfico de Degradação...")

plt.figure(figsize=(10, 6))
plt.plot(niveis_ruido, historico_acc, marker='o', linestyle='-', color='blue', label='Acurácia Global')
plt.plot(niveis_ruido, historico_f1, marker='s', linestyle='--', color='red', label='F1-Score (Weighted)')

plt.title('Degradação da Rede Neural vs Injeção de Ruído (Genes Exclusivos PAN)', fontsize=14)
plt.xlabel('Quantidade de Genes Ruidosos', fontsize=12)
plt.ylabel('Desempenho (Métrica)', fontsize=12)
plt.ylim(0.0, 1.05)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='lower left')

nome_grafico = "Curva_Degradacao_MLP.png"
plt.savefig(nome_grafico, dpi=300, bbox_inches='tight')
print(f"[SUCESSO] Gráfico salvo como '{nome_grafico}'!")

plt.show()