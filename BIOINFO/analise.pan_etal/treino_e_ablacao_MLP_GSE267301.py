import os
import time
import copy
import joblib
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)

# ==============================================================================
# 0. CAMINHOS E CONFIGURAÇÕES GLOBAIS
# ==============================================================================
CAMINHO_SWEEP_600 = "/media/gustavo-k-sato/D61CF48E1CF46B3F1/AIBIA/R_analises/SWeeP_600_Classes_Alzheimer_Oficial.tsv"
CAMINHO_METADADOS = "/media/gustavo-k-sato/D61CF48E1CF46B3F1/analisesPAN/Metadados_Pan_Limpos.csv"
CAMINHO_H5AD_LOG  = "/media/gustavo-k-sato/D61CF48E1CF46B3F1/analisesPAN/Pan_58k_LogNorm_Para_BaselineHVG.h5ad"

PASTA_SAIDA     = "/media/gustavo-k-sato/D61CF48E1CF46B3F1/analisesPAN/resultados_MLP_GSE267301_Oficial"
PASTA_ARTEFATOS = os.path.join(PASTA_SAIDA, "artefatos_para_inferencia")
os.makedirs(PASTA_SAIDA, exist_ok=True)
os.makedirs(PASTA_ARTEFATOS, exist_ok=True)

# Sementes para média e desvio padrão na ablação principal (H1)
SEEDS_AVALIACAO = [42, 100, 2026]
SEED_PRINCIPAL  = 42
EPOCHS          = 80
BATCH_TRAIN     = 256
BATCH_EVAL      = 512
LR              = 1e-3

# Dicionário Oficial do Projeto (Padrão Original Pan et al., 1 a 8)
NOMES_CLASSES_PAN = [
    "1: Astrocytes",
    "2: Oligodendrocytes",
    "3: Exc. Neurons",
    "4: Inh. Neurons",
    "5: OPCs",
    "6: Microglia",
    "7: Endothelial",
    "8: VLMCs"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Dispositivo de execução PyTorch: {DEVICE}")

def fixar_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==============================================================================
# 1. ARQUITETURA MODULAR DA MLP
# ==============================================================================
class ClassificadorCelular(nn.Module):
    def __init__(self, input_dim, num_classes=8):
        super().__init__()
        # Para 600D usa 128 na 1ª camada oculta; para <= 50 PCs mantém a arquitetura 64 -> 32
        hidden1 = 128 if input_dim > 100 else 64
        self.camadas = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.camadas(x)

# ==============================================================================
# 2. FUNÇÃO CENTRAL DE TREINAMENTO COM EARLY STOPPING REAL (DEEPCOPY)
# ==============================================================================
def treinar_e_avaliar_mlp(X_train, y_train, X_val, y_val, X_test, y_test,
                          usar_pesos=True, seed=42, epochs=EPOCHS, verbose=False):
    fixar_seed(seed)
    t0 = time.time()

    input_dim = X_train.shape[1]
    modelo = ClassificadorCelular(input_dim=input_dim, num_classes=8).to(DEVICE)

    if usar_pesos:
        classes_presentes = np.unique(y_train)
        pesos_calc = compute_class_weight(class_weight='balanced', classes=classes_presentes, y=y_train)
        # Garante vetor de 8 posições mesmo se alguma classe faltar no fold por doador
        pesos_full = np.ones(8, dtype=np.float32)
        for c, p in zip(classes_presentes, pesos_calc):
            pesos_full[c] = p
        pesos_tensor = torch.tensor(pesos_full, dtype=torch.float32).to(DEVICE)
        funcao_erro = nn.CrossEntropyLoss(weight=pesos_tensor)
    else:
        funcao_erro = nn.CrossEntropyLoss()

    otimizador = torch.optim.Adam(modelo.parameters(), lr=LR)

    loader_treino = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.long)),
        batch_size=BATCH_TRAIN, shuffle=True
    )
    loader_val = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                      torch.tensor(y_val, dtype=torch.long)),
        batch_size=BATCH_EVAL, shuffle=False
    )
    loader_test = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                      torch.tensor(y_test, dtype=torch.long)),
        batch_size=BATCH_EVAL, shuffle=False
    )

    hist_treino, hist_val = [], []
    melhor_loss_val = float('inf')
    melhor_estado = None
    epoca_ideal = 0

    for epoca in range(epochs):
        modelo.train()
        loss_acum_tr = 0.0
        for bx, by in loader_treino:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            otimizador.zero_grad()
            out = modelo(bx)
            loss = funcao_erro(out, by)
            loss.backward()
            otimizador.step()
            loss_acum_tr += loss.item() * bx.size(0)

        loss_tr = loss_acum_tr / len(loader_treino.dataset)
        hist_treino.append(loss_tr)

        modelo.eval()
        loss_acum_val = 0.0
        with torch.no_grad():
            for bx_v, by_v in loader_val:
                bx_v, by_v = bx_v.to(DEVICE), by_v.to(DEVICE)
                out_v = modelo(bx_v)
                loss_v = funcao_erro(out_v, by_v)
                loss_acum_val += loss_v.item() * bx_v.size(0)

        loss_val = loss_acum_val / len(loader_val.dataset)
        hist_val.append(loss_val)

        # CORREÇÃO CRÍTICA: copy.deepcopy congela de verdade os tensores na melhor época
        if loss_val < melhor_loss_val:
            melhor_loss_val = loss_val
            melhor_estado = {k: v.cpu().clone() for k, v in copy.deepcopy(modelo.state_dict()).items()}
            epoca_ideal = epoca + 1

        if verbose and ((epoca + 1) % 20 == 0 or epoca == 0):
            print(f"   [Época {epoca+1:02d}/{epochs:02d}] Loss Treino: {loss_tr:.4f} | Loss Val: {loss_val:.4f}")

    # Restaura o melhor estado de validação e avalia uma única vez no Teste Cego
    modelo.load_state_dict({k: v.to(DEVICE) for k, v in melhor_estado.items()})
    modelo.eval()

    y_pred = []
    with torch.no_grad():
        for bx_t, _ in loader_test:
            bx_t = bx_t.to(DEVICE)
            out_t = modelo(bx_t)
            _, prev = torch.max(out_t, 1)
            y_pred.extend(prev.cpu().numpy())

    y_pred = np.array(y_pred)
    tempo_seg = time.time() - t0

    acc = accuracy_score(y_test, y_pred)
    acc_bal = balanced_accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_wei = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_por_classe = f1_score(y_test, y_pred, average=None, labels=list(range(8)), zero_division=0)

    return {
        'modelo_state_dict': melhor_estado,
        'epoca_ideal': epoca_ideal,
        'melhor_loss_val': melhor_loss_val,
        'hist_treino': hist_treino,
        'hist_val': hist_val,
        'y_pred': y_pred,
        'Acuracia': acc,
        'Acuracia_Balanceada': acc_bal,
        'F1_Macro': f1_mac,
        'F1_Weighted': f1_wei,
        'F1_Por_Classe': f1_por_classe,
        'Tempo_Treino_s': tempo_seg
    }

# ==============================================================================
# 3. FUNÇÕES DE PLOTAGEM (CURVA DE LOSS E MATRIZES DE CONFUSÃO)
# ==============================================================================
def salvar_graficos_avaliacao(res, y_test, prefixo_arquivo, titulo_experimento):
    # 1. Curva de Loss
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, len(res['hist_treino']) + 1), res['hist_treino'], label='Loss de Treino', linewidth=2)
    plt.plot(range(1, len(res['hist_val']) + 1), res['hist_val'], label='Loss de Validação', linewidth=2)
    plt.axvline(x=res['epoca_ideal'], color='red', linestyle='--', label=f"Early Stopping (Época {res['epoca_ideal']})")
    plt.xlabel('Épocas')
    plt.ylabel('Cross-Entropy Loss')
    plt.title(f'Convergência e Early Stopping - {titulo_experimento}')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_SAIDA, f"curva_loss_{prefixo_arquivo}.png"), dpi=150)
    plt.close()

    # 2. Matriz de Confusão Normalizada por Linha (% Recall por Classe)
    cm_norm = confusion_matrix(y_test, res['y_pred'], labels=list(range(8)), normalize='true') * 100
    plt.figure(figsize=(9.5, 7.5))
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=NOMES_CLASSES_PAN, yticklabels=NOMES_CLASSES_PAN)
    plt.xlabel('Predição da MLP')
    plt.ylabel('Classe Real (Pan et al.)')
    plt.title(f"Matriz de Confusão Normalizada (%) - {titulo_experimento}\n"
              f"F1-Macro: {res['F1_Macro']:.4f} | Acc Balanceada: {res['Acuracia_Balanceada']:.4f}")
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_SAIDA, f"matriz_confusao_norm_{prefixo_arquivo}.png"), dpi=150)
    plt.close()

    # 3. Matriz de Confusão em Contagens Absolutas
    cm_abs = confusion_matrix(y_test, res['y_pred'], labels=list(range(8)))
    plt.figure(figsize=(9.5, 7.5))
    sns.heatmap(cm_abs, annot=True, fmt='d', cmap='Blues',
                xticklabels=NOMES_CLASSES_PAN, yticklabels=NOMES_CLASSES_PAN)
    plt.xlabel('Predição da MLP')
    plt.ylabel('Classe Real (Pan et al.)')
    plt.title(f"Matriz de Confusão Absoluta - {titulo_experimento}")
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_SAIDA, f"matriz_confusao_abs_{prefixo_arquivo}.png"), dpi=150)
    plt.close()

# ==============================================================================
# 4. CARREGAMENTO DOS DADOS DO PAN ET AL. (GSE267301)
# ==============================================================================
print("\n[1/5] Carregando SWeeP 600D Oficial e Metadados do GSE267301...")
df_sweep = pd.read_csv(CAMINHO_SWEEP_600, sep='\t')
X_sweep_600 = df_sweep.drop(columns=['Cell_Type']).values.astype(np.float32)
y_all = df_sweep['Cell_Type'].values.astype(int) - 1  # Base 0 (0 a 7) para PyTorch
del df_sweep

df_meta = pd.read_csv(CAMINHO_METADADOS)
doadores_all = df_meta['Donor'].astype(str).values

assert X_sweep_600.shape[0] == len(y_all) == len(doadores_all), "ERRO: Desalinhamento entre SWeeP e Metadados!"
print(f" -> Matriz SWeeP carregada: {X_sweep_600.shape[0]} células x {X_sweep_600.shape[1]} dimensões.")
print(f" -> Total de doadores únicos: {len(np.unique(doadores_all))} ({np.unique(doadores_all)})")

# Índices universais (70% Treino / 15% Validação / 15% Teste Cego) para comparação justa entre todos os braços
indices = np.arange(len(y_all))
idx_train, idx_temp, y_train, y_temp = train_test_split(
    indices, y_all, test_size=0.30, random_state=SEED_PRINCIPAL, stratify=y_all
)
idx_val, idx_test, y_val, y_test = train_test_split(
    idx_temp, y_temp, test_size=0.50, random_state=SEED_PRINCIPAL, stratify=y_temp
)
print(f" -> Split Oficial: Treino = {len(idx_train)} | Validação = {len(idx_val)} | Teste Cego = {len(idx_test)}")

# ==============================================================================
# 5. EXPERIMENTO 1: ABLAÇÃO DE REPRESENTAÇÃO (H0 vs H1) COM MÚLTIPLAS SEEDS
#    Compara: (A) SWeeP 600D Puro | (B) SWeeP + PCA 50D | (C) 2000 HVGs + PCA 50D
# ==============================================================================
print("\n======================================================================")
print("[2/5] EXECUTANDO EXPERIMENTO 1: ABLAÇÃO DE REPRESENTAÇÃO (H0 vs H1)")
print("======================================================================")

# --- Preparação Braço A: SWeeP 600D Puro (Sem PCA) ---
t_prep_a = time.time()
scaler_600d = StandardScaler()
X_tr_600 = scaler_600d.fit_transform(X_sweep_600[idx_train])
X_va_600 = scaler_600d.transform(X_sweep_600[idx_val])
X_te_600 = scaler_600d.transform(X_sweep_600[idx_test])
tempo_prep_a = time.time() - t_prep_a

# --- Preparação Braço B: SWeeP 600D + PCA (50 PCs ajustado APENAS no treino) ---
t_prep_b = time.time()
pca_sweep_50 = PCA(n_components=50, random_state=SEED_PRINCIPAL)
X_tr_sw_pca50_raw = pca_sweep_50.fit_transform(X_sweep_600[idx_train])
X_va_sw_pca50_raw = pca_sweep_50.transform(X_sweep_600[idx_val])
X_te_sw_pca50_raw = pca_sweep_50.transform(X_sweep_600[idx_test])

scaler_pca50 = StandardScaler()
X_tr_sw_pca50 = scaler_pca50.fit_transform(X_tr_sw_pca50_raw)
X_va_sw_pca50 = scaler_pca50.transform(X_va_sw_pca50_raw)
X_te_sw_pca50 = scaler_pca50.transform(X_te_sw_pca50_raw)
tempo_prep_b = time.time() - t_prep_b

# --- Preparação Braço C (Baseline H0): 2000 HVGs + PCA (50 PCs ajustado APENAS no treino) ---
registros_ablacao = []
tem_baseline_hvg = os.path.exists(CAMINHO_H5AD_LOG)

if tem_baseline_hvg:
    print(" -> Extraindo 2.000 HVGs e PCA (50 PCs) exclusivamente sobre o conjunto de treino...")
    t_prep_c = time.time()
    adata_log = sc.read_h5ad(CAMINHO_H5AD_LOG)
    adata_treino_hvg = adata_log[idx_train, :].copy()
    sc.pp.highly_variable_genes(adata_treino_hvg, n_top_genes=2000, flavor='seurat')
    mask_hvg = adata_treino_hvg.var['highly_variable'].values
    del adata_treino_hvg

    X_hvg_tr_dense = adata_log[idx_train, mask_hvg].X
    X_hvg_va_dense = adata_log[idx_val, mask_hvg].X
    X_hvg_te_dense = adata_log[idx_test, mask_hvg].X
    if sp.issparse(X_hvg_tr_dense):
        X_hvg_tr_dense = X_hvg_tr_dense.toarray()
        X_hvg_va_dense = X_hvg_va_dense.toarray()
        X_hvg_te_dense = X_hvg_te_dense.toarray()
    del adata_log

    pca_hvg_50 = PCA(n_components=50, random_state=SEED_PRINCIPAL)
    X_tr_hvg_pca50_raw = pca_hvg_50.fit_transform(X_hvg_tr_dense)
    X_va_hvg_pca50_raw = pca_hvg_50.transform(X_hvg_va_dense)
    X_te_hvg_pca50_raw = pca_hvg_50.transform(X_hvg_te_dense)
    del X_hvg_tr_dense, X_hvg_va_dense, X_hvg_te_dense

    scaler_hvg50 = StandardScaler()
    X_tr_hvg50 = scaler_hvg50.fit_transform(X_tr_hvg_pca50_raw)
    X_va_hvg50 = scaler_hvg50.transform(X_va_hvg_pca50_raw)
    X_te_hvg50 = scaler_hvg50.transform(X_te_hvg_pca50_raw)
    tempo_prep_c = time.time() - t_prep_c

bracos_h1 = [
    ("SWeeP_600D_Puro", X_tr_600, X_va_600, X_te_600, 600, tempo_prep_a),
    ("SWeeP_600D_PCA50", X_tr_sw_pca50, X_va_sw_pca50, X_te_sw_pca50, 50, tempo_prep_b)
]
if tem_baseline_hvg:
    bracos_h1.append(("Baseline_HVG2000_PCA50", X_tr_hvg50, X_va_hvg50, X_te_hvg50, 50, tempo_prep_c))

for nome_braco, X_tr, X_va, X_te, dim, t_prep in bracos_h1:
    print(f"\n--- Avaliando Braço: {nome_braco} ({dim}D) em {len(SEEDS_AVALIACAO)} seeds ---")
    for s in SEEDS_AVALIACAO:
        res = treinar_e_avaliar_mlp(X_tr, y_train, X_va, y_val, X_te, y_test, usar_pesos=True, seed=s)
        print(f"  [Seed {s}] Época Ideal: {res['epoca_ideal']:02d} | "
              f"F1-Macro: {res['F1_Macro']:.4f} | Acc Bal: {res['Acuracia_Balanceada']:.4f} | Acc: {res['Acuracia']:.4f}")

        linha = {
            'Representacao': nome_braco,
            'Dimensao': dim,
            'Seed': s,
            'Epoca_Ideal': res['epoca_ideal'],
            'Acuracia': res['Acuracia'],
            'Acuracia_Balanceada': res['Acuracia_Balanceada'],
            'F1_Macro': res['F1_Macro'],
            'F1_Weighted': res['F1_Weighted'],
            'Tempo_Prep_s': t_prep,
            'Tempo_Treino_s': res['Tempo_Treino_s']
        }
        for idx_c, nome_c in enumerate(NOMES_CLASSES_PAN):
            linha[f"F1_{nome_c.split(':')[1].strip()}"] = res['F1_Por_Classe'][idx_c]
        registros_ablacao.append(linha)

        # Na seed principal (42), salva gráficos, relatórios e congela artefatos para inferência externa
        if s == SEED_PRINCIPAL:
            salvar_graficos_avaliacao(res, y_test, prefixo_arquivo=nome_braco, titulo_experimento=nome_braco)
            with open(os.path.join(PASTA_SAIDA, f"relatorio_{nome_braco}.txt"), "w") as f:
                f.write(classification_report(y_test, res['y_pred'], target_names=NOMES_CLASSES_PAN, digits=4))

            if nome_braco == "SWeeP_600D_Puro":
                torch.save(res['modelo_state_dict'], os.path.join(PASTA_ARTEFATOS, "mlp_600d_puro.pt"))
                joblib.dump(scaler_600d, os.path.join(PASTA_ARTEFATOS, "scaler_600d.pkl"))
            elif nome_braco == "SWeeP_600D_PCA50":
                torch.save(res['modelo_state_dict'], os.path.join(PASTA_ARTEFATOS, "mlp_pca50d.pt"))
                joblib.dump(pca_sweep_50, os.path.join(PASTA_ARTEFATOS, "pca_50d.pkl"))
                joblib.dump(scaler_pca50, os.path.join(PASTA_ARTEFATOS, "scaler_pca50d.pkl"))

df_ablacao = pd.DataFrame(registros_ablacao)
df_ablacao.to_csv(os.path.join(PASTA_SAIDA, "E1_E2_Ablacao_Representacoes_Seeds.csv"), index=False)

# Resumo com Média e Desvio Padrão (Tabela pronta para a Monografia)
resumo_h1 = df_ablacao.groupby('Representacao')[['Acuracia', 'Acuracia_Balanceada', 'F1_Macro', 'F1_Weighted', 'Tempo_Treino_s']].agg(['mean', 'std'])
print("\n[RESUMO H0 vs H1 - MÉDIA ± DESVIO PADRÃO NO TESTE CEGO]:")
print(resumo_h1.to_string())
resumo_h1.to_csv(os.path.join(PASTA_SAIDA, "Resumo_H1_Media_Desvio.csv"))

# ==============================================================================
# 6. EXPERIMENTO 2: CURVA DE DIMENSIONALIDADE (50..10 PCs) E PESOS DE CLASSE (H2/H4)
# ==============================================================================
print("\n======================================================================")
print("[3/5] EXECUTANDO EXPERIMENTO 2: CURVA DE PCs (50..10) & ABLAÇÃO DE PESOS")
print("======================================================================")

lista_pcs = [50, 45, 40, 35, 30, 25, 20, 15, 10]
registros_curva_pcs = []

for k in lista_pcs:
    # Recorta as primeiras k componentes do PCA já ajustado apenas no treino (X_tr_sw_pca50_raw)
    sc_k = StandardScaler()
    X_tr_k = sc_k.fit_transform(X_tr_sw_pca50_raw[:, :k])
    X_va_k = sc_k.transform(X_va_sw_pca50_raw[:, :k])
    X_te_k = sc_k.transform(X_te_sw_pca50_raw[:, :k])

    res_k = treinar_e_avaliar_mlp(X_tr_k, y_train, X_va_k, y_val, X_te_k, y_test, usar_pesos=True, seed=SEED_PRINCIPAL)
    print(f" -> [SWeeP + {k:02d} PCs] Época Ideal: {res_k['epoca_ideal']:02d} | "
          f"F1-Macro: {res_k['F1_Macro']:.4f} | Acc Bal: {res_k['Acuracia_Balanceada']:.4f}")

    registros_curva_pcs.append({
        'PCs': k,
        'Epoca_Ideal': res_k['epoca_ideal'],
        'Var_Explicada_Acum_%': np.sum(pca_sweep_50.explained_variance_ratio_[:k]) * 100,
        'Acuracia': res_k['Acuracia'],
        'Acuracia_Balanceada': res_k['Acuracia_Balanceada'],
        'F1_Macro': res_k['F1_Macro'],
        'F1_Weighted': res_k['F1_Weighted']
    })

df_curva = pd.DataFrame(registros_curva_pcs)
df_curva.to_csv(os.path.join(PASTA_SAIDA, "E4_Curva_Dimensionalidade_PCs.csv"), index=False)

# Teste de Ablação de Pesos (H2): Com Pesos ('balanced') vs Sem Pesos em 50 PCs e 600D
print("\n -> Avaliando impacto da ponderação de classes (Com Pesos vs Sem Pesos)...")
registros_pesos = []
for nome_repr, X_tr, X_va, X_te in [("SWeeP_600D_Puro", X_tr_600, X_va_600, X_te_600),
                                    ("SWeeP_600D_PCA50", X_tr_sw_pca50, X_va_sw_pca50, X_te_sw_pca50)]:
    for flag_peso in [True, False]:
        res_p = treinar_e_avaliar_mlp(X_tr, y_train, X_va, y_val, X_te, y_test, usar_pesos=flag_peso, seed=SEED_PRINCIPAL)
        item = {
            'Representacao': nome_repr,
            'Estrategia_Pesos': 'Balanced' if flag_peso else 'Sem_Pesos',
            'Acuracia_Global': res_p['Acuracia'],
            'Acuracia_Balanceada': res_p['Acuracia_Balanceada'],
            'F1_Macro': res_p['F1_Macro'],
            'F1_Weighted': res_p['F1_Weighted'],
            'F1_Endothelial_Rara': res_p['F1_Por_Classe'][6],
            'F1_VLMCs_Rara': res_p['F1_Por_Classe'][7]
        }
        registros_pesos.append(item)

df_pesos = pd.DataFrame(registros_pesos)
df_pesos.to_csv(os.path.join(PASTA_SAIDA, "E3_Ablacao_Pesos_Classes_Raras.csv"), index=False)
print("\n[RESULTADO H2 - IMPACTO DOS PESOS NAS CLASSES RARAS]:")
print(df_pesos.to_string(index=False))

# ==============================================================================
# 7. EXPERIMENTO 3: VALIDAÇÃO AGRUPADA POR DOADOR (GROUPKFOLD - H3 INTERNA)
# ==============================================================================
print("\n======================================================================")
print("[4/5] EXECUTANDO EXPERIMENTO 3: VALIDAÇÃO POR DOADOR (GroupKFold)")
print("======================================================================")

n_doadores = len(np.unique(doadores_all))
gkf = GroupKFold(n_splits=n_doadores)
registros_doador = []

for fold_idx, (idx_tr_val, idx_te_donor) in enumerate(gkf.split(X_sweep_600, y_all, groups=doadores_all)):
    doador_retido = np.unique(doadores_all[idx_te_donor])[0]
    
    # Separa 15% do treino dos outros 7 doadores para Early Stopping
    idx_tr_d, idx_va_d = train_test_split(
        idx_tr_val, test_size=0.15, random_state=SEED_PRINCIPAL, stratify=y_all[idx_tr_val]
    )

    # Ajusta PCA(50) e StandardScaler estritamente nos 7 doadores de treino
    pca_d = PCA(n_components=50, random_state=SEED_PRINCIPAL)
    X_tr_d_pca = pca_d.fit_transform(X_sweep_600[idx_tr_d])
    X_va_d_pca = pca_d.transform(X_sweep_600[idx_va_d])
    X_te_d_pca = pca_d.transform(X_sweep_600[idx_te_donor])

    sc_d = StandardScaler()
    X_tr_d = sc_d.fit_transform(X_tr_d_pca)
    X_va_d = sc_d.transform(X_va_d_pca)
    X_te_d = sc_d.transform(X_te_d_pca)

    res_d = treinar_e_avaliar_mlp(
        X_tr_d, y_all[idx_tr_d], X_va_d, y_all[idx_va_d], X_te_d, y_all[idx_te_donor],
        usar_pesos=True, seed=SEED_PRINCIPAL
    )

    # Calcula F1-Macro apenas sobre as classes que realmente existem neste doador retido
    classes_no_doador = np.unique(y_all[idx_te_donor])
    f1_macro_presentes = f1_score(
        y_all[idx_te_donor], res_d['y_pred'], labels=classes_no_doador, average='macro', zero_division=0
    )

    print(f" -> [Fold {fold_idx+1}/{n_doadores} | Doador Retido: {doador_retido} ({len(idx_te_donor)} céls)] "
          f"Acc: {res_d['Acuracia']:.4f} | F1-Weighted: {res_d['F1_Weighted']:.4f} | F1-Macro ( Classes Presentes ): {f1_macro_presentes:.4f}")

    registros_doador.append({
        'Fold': fold_idx + 1,
        'Doador_Retido': doador_retido,
        'N_Celulas_Teste': len(idx_te_donor),
        'N_Classes_Presentes': len(classes_no_doador),
        'Acuracia': res_d['Acuracia'],
        'Acuracia_Balanceada': res_d['Acuracia_Balanceada'],
        'F1_Weighted': res_d['F1_Weighted'],
        'F1_Macro_Classes_Presentes': f1_macro_presentes
    })

df_doador = pd.DataFrame(registros_doador)
df_doador.to_csv(os.path.join(PASTA_SAIDA, "E5_Leave_One_Donor_Out_GSE267301.csv"), index=False)

print("\n======================================================================")
print("[5/5] TODOS OS EXPERIMENTOS INTERNOS E ARTEFATOS FORAM CONCLUÍDOS!")
print(f" -> Tabelas e Gráficos salvos em: {PASTA_SAIDA}")
print(f" -> Modelos e Scalers congelados para Inferência Externa em: {PASTA_ARTEFATOS}")
print("======================================================================")