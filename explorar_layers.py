import h5py
import numpy as np

caminho_h5ad = "/home/gustavo-k-sato/Downloads/GSE263468_processed_data.h5ad"

print("--- INSPEÇÃO MATEMÁTICA PROFUNDA DAS CAMADAS (LAYERS) ---")
try:
    with h5py.File(caminho_h5ad, 'r') as f:
        if 'layers' in f:
            camadas = list(f['layers'].keys())
            print(f"Camadas detectadas no disco: {camadas}\n")
            
            for camada in camadas:
                print(f"Analisando camada: '{camada}'")
                obj = f['layers'][camada]
                
                # Extrai a amostra dependendo da estrutura (Esparsa vs Densa)
                if isinstance(obj, h5py.Group) and 'data' in obj:
                    amostra = obj['data'][0:100000]
                    tipo = "Esparsa (comprimida)"
                else:
                    amostra = obj[0:500, :].flatten() # Extrai um bloco denso e achata
                    tipo = "Densa"
                
                # Testes de integridade
                is_integer = np.all(amostra % 1 == 0)
                min_val = amostra.min()
                max_val = amostra.max()
                
                print(f"  -> Estrutura: {tipo}")
                print(f"  -> Valores Inteiros Absolutos? {'SIM (Puro/Raw)' if is_integer else 'NÃO (Processado/Normalizado)'}")
                print(f"  -> Range de Valores: Min {min_val:.4f} | Max {max_val:.4f}")
                
                if min_val < 0:
                    print("  -> ALERTA: Valores negativos detectados (Possível regressão/SCTransform)!")
                print("-" * 50)
        else:
            print("Nenhuma sub-pasta 'layers' encontrada no arquivo.")
except Exception as e:
    print(f"Erro ao acessar o arquivo: {e}")