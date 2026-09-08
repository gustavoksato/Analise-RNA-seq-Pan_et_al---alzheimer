import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import 


# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
matriz = pd.read_csv("PCA50_Classes_Alzheimer.txt", sep=r'\s+', engine='python')
X_all = matriz.drop(columns=['Cell_Type']).values  
y_raw = matriz['Cell_Type'].values - 1  # Ajuste para base 0 (0 a 7)

#testando em diferentes PC'S
lista_pcs = [50, 45, 40, 35, 30, 25, 20, 15, 10]
historico_experimento = []

# 7. ARQUITETURA DA REDE NEURAL (Capacidade expandida para suportar os pesos)
class ClassificadorCelular(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.camadas = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.camadas(x)


for n_pcs in lista_pcs:
    print(f"\n================ Executando com {n_pcs} PCs ================")
    
    X_raw = X_all[:, :n_pcs]
    # 2. DIVISÃO EM TREINO E VALIDAÇÃO (80% treino / 20% validação)
    # O parâmetro 'stratify=y_raw' garante que a proporção das classes raras seja mantida em ambos os conjuntos
    X_treino_bruto, X_val_bruto, y_treino, y_val = train_test_split(
        X_raw, y_raw, test_size=0.20, random_state=42, stratify=y_raw
    )

# 3. PADRONIZAÇÃO (Ajusta a escala com base APENAS no treino para evitar vazamento de dados)
    scaler = StandardScaler()
    X_treino_escalado = scaler.fit_transform(X_treino_bruto)
    X_val_escalado = scaler.transform(X_val_bruto)

# 4. CÁLCULO DOS PESOS BALANCEADOS (Apenas com base no conjunto de treino)
    pesos = compute_class_weight(class_weight='balanced', classes=np.unique(y_treino), y=y_treino)
    pesos_tensor = torch.tensor(pesos, dtype=torch.float32)

# 5. CONVERSÃO PARA TENSORES DO PYTORCH
    X_train_tensor = torch.tensor(X_treino_escalado, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_treino, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val_escalado, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# 6. CRIAÇÃO DOS LOADERS (Apenas o de treino precisa de shuffle)
    dataset_treino = TensorDataset(X_train_tensor, y_train_tensor)
    loader_treino = DataLoader(dataset_treino, batch_size=256, shuffle=True)

    dataset_val = TensorDataset(X_val_tensor, y_val_tensor)
    loader_val = DataLoader(dataset_val, batch_size=512, shuffle=False)

#Inicialização do modelo dinâmico
    funcao_erro = nn.CrossEntropyLoss(weight=pesos_tensor)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)
    modelo = ClassificadorCelular(input_dim=n_pcs, num_classes=8)

#treinamento otimizado
    epochs = 80
    melhor_erro_val = float('inf')
    melhor_estado_modelo = None

    print("Iniciando o treinamento com monitoramento de Overfitting...\n")

    for epoca in range(epochs):
        # FASE DE TREINO
        modelo.train()
        erro_acumulado_treino = 0.0
        for lote_X, lote_y in loader_treino:
            previsoes = modelo(lote_X)
            erro = funcao_erro(previsoes, lote_y)
            
            otimizador.zero_grad()
            erro.backward()
            otimizador.step()
            
            erro_acumulado_treino += erro.item()
        
        # FASE DE VALIDAÇÃO (Teste Cego)
        modelo.eval()
        erro_acumulado_val = 0.0
        with torch.no_grad():
            for lote_X_val, lote_y_val in loader_val:
                previsoes_val = modelo(lote_X_val)
                erro_v = funcao_erro(previsoes_val, lote_y_val)
                erro_acumulado_val += erro_v.item() * lote_X_val.size(0)
                
        erro_medio_val = erro_acumulado_val / len(loader_val.dataset)

        # Early Stopping manual (salva o melhor estado)
        if erro_medio_val < melhor_erro_val:
            melhor_erro_val = erro_medio_val
            melhor_estado_modelo = modelo.state_dict().copy()
    
    modelo.load_state_dict(melhor_estado_modelo)
    modelo.eval()
    todas_previsoes = []
    with torch.no_grad():
        for lote_X_val, _ in loader_val:
            saida = modelo(lote_X_val)
            _, previsao = torch.max(saida, 1)
            todas_previsoes.extend(previsao.numpy())
    
    acc = accuracy_score(y_val, todas_previsoes)
    f1_weighted = f1_score(y_val, todas_previsoes, average='weighted')
    f1_macro = f1_score(y_val, todas_previsoes, average='macro')
    
    print(f"Resultado - PCs: {n_pcs} | Acurácia: {acc:.4f} | F1-Score (Weighted): {f1_weighted:.4f}")
    
    historico_experimento.append({
        'PCs': n_pcs,
        'Acurácia': acc,
        'F1_Weighted': f1_weighted,
        'F1_Macro': f1_macro
    })    
    print("\nTreinamento concluído. Gerando o gráfico...")

if (epoca + 1) % 5 == 0 or epoca == 0:
            print(f"Época {epoca+1:03d}/{epochs} | Erro Treino: {erro_medio_treino:.4f} | Erro Validação: {erro_medio_val:.4f}")

        erro_medio_treino = erro_acumulado_treino / len(loader_treino)
        historico_erro_treino.append(erro_medio_treino)
        

# 9. CONSTRUÇÃO DO GRÁFICO DE OVERFITTING
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), historico_erro_treino, label='Erro de Treino (Aprendizado)', linewidth=2)
plt.plot(range(1, epochs + 1), historico_erro_val, label='Erro de Validação (Teste Cego)', linewidth=2)

# Identificar visualmente o ponto mínimo de validação (onde deveria ocorrer o Early Stopping)
ponto_min_val = np.argmin(historico_erro_val)
plt.axvline(x=ponto_min_val+1, color='red', linestyle='--', label=f'Ponto Ideal (Época {ponto_min_val+1})')

plt.xlabel('Épocas de Treinamento')
plt.ylabel('Erro Matemático (Loss)')
plt.title('Curvas de Erro - Diagnóstico de Overfitting')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# Salva e exibe o gráfico
plt.savefig("curva_aprendizado_overfitting.png", dpi=300)
plt.show()
print("Gráfico salvo com sucesso como 'curva_aprendizado_overfitting.png'.")

print("\n--- AVALIAÇÃO FINAL NO CONJUNTO DE VALIDAÇÃO (TESTE CEGO) ---")

# Garante que o modelo está congelado para avaliação
modelo.eval()

# Passa os tensores de validação (aqueles 20% que a rede não usou para atualizar pesos)
with torch.no_grad():
    saidas_val = modelo(X_val_tensor)
    _, previsoes_finais = torch.max(saidas_val, 1)

# Converte de tensores PyTorch para arrays NumPy para usar no scikit-learn
y_real_val = y_val_tensor.numpy()
y_previsto_val = previsoes_finais.numpy()

# Gera os nomes das 8 classes para os eixos dos gráficos
nomes_classes = [f"Classe {i+1}" for i in range(8)]

# Imprime o relatório textual com Precisão, Recall e F1-Score
print(classification_report(y_real_val, y_previsto_val, target_names=nomes_classes))

# Calcula a matriz matemática
matriz_conf = confusion_matrix(y_real_val, y_previsto_val)

# Plota e salva o gráfico
plt.figure(figsize=(10, 8))
sns.heatmap(matriz_conf, annot=True, fmt='d', cmap='Blues', 
            xticklabels=nomes_classes, 
            yticklabels=nomes_classes)

plt.xlabel('Previsão do Modelo')
plt.ylabel('Classe Real (Biológica)')
plt.title('Matriz de Confusão (Validação) - Identificação Celular')
plt.tight_layout()

# Salva com um nome ligeiramente diferente para você saber que é a do teste cego
plt.savefig("matriz_confusao_SWeeP_Validacao.png", dpi=300)
print("\nMatriz de Confusão salva com sucesso como 'matriz_confusao_SWeeP_Validacao.png'.")




