# Required installations:
# pip install langchain langchain-aws pypdf2 sentence-transformers faiss-cpu chromadb
from langchain_community.document_loaders import S3FileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- CONFIGURATION ---
NOME_DO_BUCKET = "chatbot-analise-dados" 
CAMINHO_DO_ARQUIVO_NO_S3 = "documentos-rag/Taboa_PoliticaDeCredito.pdf" 

def preparar_documentos_do_s3():
    """
    Reads a PDF from S3, splits it, creates embeddings, and saves into a local Vector Store.
    """
    print(f"1. Carregando o documento de s3://chatbot-analise-dados/documentos-rag/Taboa_PoliticaDeCredito.pdf")
    loader = S3FileLoader(
        NOME_DO_BUCKET,
        CAMINHO_DO_ARQUIVO_NO_S3,
        loader_kwargs={"languages": ["por"]}
    )
    documentos = loader.load()

    print("2. Dividindo o documento em pedaços (chunks)...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documentos)
    print(f"   Documento dividido em {len(chunks)} pedaços.")

    print("3. Carregando o modelo de embedding...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    print("4. Criando o Vector Store com ChromaDB e salvando localmente...")
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db_rag")
    
    print("✅ Documentos do S3 processados e salvos com sucesso na pasta local 'chroma_db_rag'!")

# --- EXECUTION ---
if __name__ == '__main__':
    preparar_documentos_do_s3()
