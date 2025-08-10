import pandas as pd

nome_do_arquivo = r'train.gz' 

print(f"Tentando ler o arquivo '{nome_do_arquivo}'...")

try:
    
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
