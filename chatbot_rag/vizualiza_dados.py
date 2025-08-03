import pandas as pd

# Coloque o nome exato do seu arquivo aqui
nome_do_arquivo = r'train.gz' 

print(f"Tentando ler o arquivo '{nome_do_arquivo}'...")

try:
    # O pandas detecta a compressão .gz e lê o arquivo CSV diretamente
    # O parâmetro 'error_bad_lines=False' pode ser útil se algumas linhas estiverem corrompidas
    # O 'warn_bad_lines=True' irá avisá-lo sobre linhas com problemas.
    # Em versões mais recentes do pandas, esses parâmetros foram substituídos por on_bad_lines='skip' ou 'warn'.
    # Vamos usar um try-except para compatibilidade.
    
    try:
        df = pd.read_csv(nome_do_arquivo, compression='gzip', sep=',')
    except Exception:
        # Tenta a sintaxe mais nova se a antiga falhar
        df = pd.read_csv(nome_do_arquivo, compression='gzip', sep=',', on_bad_lines='warn')

    print("\n✅ Arquivo lido com sucesso! Aqui estão as primeiras 5 linhas:")
    # .head() mostra as primeiras linhas de forma organizada
    print(df.head())

    print("\n----------------------------------------------------------")
    print("\n🧾 Informações Gerais sobre o DataFrame (colunas, tipos, etc.):")
    # .info() dá um resumo excelente das colunas, contagem de nulos e tipos de dados
    df.info(verbose=True, show_counts=True)
    
    print("\n----------------------------------------------------------")
    print(f"\n📄 O DataFrame tem {df.shape[0]} linhas e {df.shape[1]} colunas.")


except FileNotFoundError:
    print(f"❌ ERRO: O arquivo '{nome_do_arquivo}' não foi encontrado no mesmo diretório do script.")
except Exception as e:
    print(f"❌ ERRO ao tentar ler o arquivo: {e}")
    print("Dica: Verifique se o separador é mesmo uma vírgula (sep=','). Se for ponto e vírgula, troque para sep=';'.")