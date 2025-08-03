import pandas as pd


# --- SETTINGS ---

# 1. Local input file
arquivo_gz_local = 'train.gz' 

# 2. ORIGINAL columns to select from the file
colunas_desejadas = [
    'REF_DATE',
    'TARGET',
    'VAR2',
    'IDADE',
    'VAR4',
    'VAR5',
    'VAR8'
]

# 3. Mapping for renaming columns (From: To)
mapa_renomear_colunas = {
    'REF_DATE': 'data_referencia',
    'TARGET': 'inadimplente',
    'VAR2': 'sexo',
    'IDADE': 'idade',
    'VAR4': 'flag_obito',
    'VAR5': 'uf',
    'VAR8': 'classe_social'
}


def preparar_e_enviar():
    """
    Reads the GZ file, selects, renames, cleans, converts, and prepares for upload to S3.
    """
    print(f"📖 Reading local file: {arquivo_gz_local}...")
    try:
        df = pd.read_csv(arquivo_gz_local, compression='gzip', sep=',')
        print("✅ File successfully read.")
    except FileNotFoundError:
        print(f"❌ ERROR: File '{arquivo_gz_local}' not found. Please check the path.")
        return

    # Step 1: Select only the columns defined as important
    print(f"🔪 Selecting original columns: {colunas_desejadas}...")
    # Ensure that all desired columns exist before selecting
    colunas_existentes = [col for col in colunas_desejadas if col in df.columns]
    df = df[colunas_existentes]

    # Step 2: Rename columns to more friendly names
    print("✨ Renaming columns...")
    df.rename(columns=mapa_renomear_colunas, inplace=True)
    print("New column names:", list(df.columns))

    # Step 3: Cleaning and transformation on the already renamed column
    print("🔄 Converting 'data_referencia' to datetime format...")
    # Use the new column name 'data_referencia'
    df['data_referencia'] = pd.to_datetime(df['data_referencia'], errors='coerce')
    
    # Drop rows where date conversion failed
    df.dropna(subset=['data_referencia'], inplace=True)
    print("✅ Data cleaned and transformed.")
    
    # Step 4: Convert the cleaned DataFrame to Parquet format
    arquivo_parquet_local = 'temp_dataset.parquet'
    print(f"📄 Converting to Parquet: '{arquivo_parquet_local}'...")
    df.to_parquet(arquivo_parquet_local, index=False)
    
    print("✅ Conversion completed.")


# --- EXECUTION ---
if __name__ == "__main__":
    preparar_e_enviar()
