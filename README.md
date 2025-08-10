# 💬 Chatbot de Análise de Dados Híbrido com RAG e SQL usando API OPENAI

Dir chatbot_rag (Pasta usada no vídeo), Dir chatbot (Pasta Organizada)

Demo [Vídeo](https://youtu.be/zghsB5Qvx2Y)

## 📖 Sobre o Projeto  
Este projeto é um **chatbot avançado de análise de dados** desenvolvido em **Python** com interface **Streamlit**.  
O sistema combina duas formas de análise:  
- **Consultas a dados estruturados (Text-to-SQL)** no **AWS Athena**.  
- **Busca em documentos não estruturados (RAG - Retrieval-Augmented Generation)**.  

O **roteador** do chatbot decide automaticamente qual abordagem usar com base na pergunta do usuário.

---

1. Pré-requisitos
Python 3.9 ou superior.

Uma conta na AWS.

Uma chave de API da OpenAI (ou outro LLM configurado) (passada via `config.json`).



## Sobre os dados
Foi feito um pré processamento de remoção de NaNs e remoção de colunas que não eram úteis que está no arquivo `pre_data.py`, além de uma conversão para parquet que melhora a perfomance no ambiente Athena.

## ✨ Funcionalidades  
- **Interface Web Interativa** → Desenvolvida com **Streamlit** para navegação simples e visual agradável.  
- **Roteador** → O **modelo** escolhe entre SQL ou RAG para responder à pergunta.  
- **Text-to-SQL** → Converte perguntas em linguagem natural para queries SQL e executa no **AWS Athena**.  
- **Busca em PDFs com RAG** → Indexa e consulta documentos PDF via **ChromaDB**.  
- **Respostas Inteligentes** → Retorna resumos claros em linguagem natural.  
- **Visualização de Dados** → Geração de gráficos  com **Plotly**.  

---

## 📂 Estrutura do Projeto
```
/chatbot_rag
├── chatbot-env/                # Ambiente virtual (opcional)
├── chroma_db_rag/              # Base vetorial do RAG
├── Taboa_PoliticaDeCredito.pdf # Documento de exemplo para o RAG
├── chatbot_app.py               # Aplicação principal (Streamlit)
├── send_documents_s3.py     # Script para indexação dos PDFs
├── requirements.txt             # Dependências do projeto
└── README.md                    # Documentação
```

**Principais Arquivos:**  
- **chatbot_app.py** → Interface + lógica de roteamento (SQL ou RAG).  
- **send_documents_s3.py** → Indexa PDFs do S3 no **ChromaDB**.  
- **requirements.txt** → Lista de dependências.  

---

## 🛠️ Dependências  
Arquivo `requirements.txt`:
```
streamlit
openai
pandas
boto3
langchain
langchain-community
langchain-aws
langchain-huggingface
pypdf2
sentence-transformers
faiss-cpu
chromadb
plotly
seaborn
matplotlib
torch
transformers
unstructured
langchain-embeddings-huggingface
langchain-vectorstores-chroma
langchain-document-loaders-s3
langchain_community.document_loaders
unstructured[pdf]
```

---

## 🚀 Como Executar o Projeto  

### 1️⃣ Pré-requisitos  
- **Python** ≥ 3.9  
- Conta na **AWS** com acesso ao **S3**, **Glue** e **Athena**.  
- **API Key** da OpenAI ou outro provedor de LLM.  

### 2️⃣ Configuração AWS
Configuração do Ambiente AWS (Guia Detalhado)
Esta etapa prepara toda a infraestrutura na nuvem necessária para o projeto.

1.  Criando uma Conta na AWS
Se você ainda não tem uma conta, acesse [Amazon](aws.amazon.com) e clique em "Crie uma conta da AWS".

O processo de cadastro é similar a outros serviços online e exigirá um e-mail e um cartão de crédito (mesmo que os serviços utilizados se enquadrem no nível gratuito, um método de pagamento é necessário para verificação).

2. Criando o Bucket no S3
O bucket S3 será nosso "armazém" na nuvem para guardar tanto os dados estruturados (Parquet) quanto os documentos não estruturados (PDF).

Faça login no Console de Gerenciamento da AWS.

Na barra de pesquisa, digite S3 e acesse o serviço.

Clique no botão laranja "Criar bucket".

Nome do bucket: Escolha um nome único globalmente (nenhum outro usuário da AWS no mundo pode ter um bucket com o mesmo nome). Ex: chatbot-analise-dados.

Região da AWS: Selecione a região onde o bucket será criado. Recomenda-se usar "América do Sul (São Paulo) sa-east-1" para baixa latência.

Configurações de acesso: Mantenha a opção padrão "Bloquear todo o acesso público" marcada por segurança.

Clique em "Criar bucket" no final da página.

3. Criando o Usuário IAM para Acesso Programático
Por segurança, nunca usamos nossa conta principal (root) para acesso via código. Criamos um "usuário" com permissões limitadas apenas para o que nossa aplicação precisa fazer.

No console da AWS, pesquise por IAM e acesse o serviço.

No menu à esquerda, clique em "Usuários".

Clique no botão "Criar usuário".

Nome de usuário: Dê um nome descritivo, como chatbot-app-user. Clique em "Próximo".

Na tela de permissões, selecione "Anexar políticas diretamente".

Na barra de pesquisa de políticas, procure e marque a caixa de seleção para cada uma das seguintes políticas:

AmazonS3FullAccess (Permite ler e escrever arquivos no S3)

AmazonAthenaFullAccess (Permite executar consultas no Athena)

AWSGlueConsoleFullAccess (Permite ao Glue criar e gerenciar o catálogo de dados)

Clique em "Próximo", revise as informações e clique em "Criar usuário".

ETAPA CRÍTICA: Após criar, clique no nome do usuário na lista. Vá para a aba "Credenciais de segurança", role a página até "Chaves de acesso" e clique em "Criar chave de acesso".

Selecione "Interface de linha de comando (CLI)", marque a caixa de confirmação e clique em "Próximo".

A AWS exibirá a ID da chave de acesso e a Chave de acesso secreta. Copie ambos imediatamente para um local seguro ou clique em "Fazer download do arquivo .csv". A chave secreta não será mostrada novamente.  
4. Criar **bucket** no **S3** (ex: `chatbot-analise-dados`).  
5. **Dados estruturados** → Enviar `.parquet` para `s3://chatbot-analise-dados/dados_credito/`.  
6. **Documentos** → Enviar PDFs para `s3://chatbot-analise-dados/documentos-rag/`.  
7. Criar **Crawler no AWS Glue** apontando para os dados estruturados.  
8. Configurar **credenciais da AWS** localmente via variáveis de ambiente:  
   ```bash
   export AWS_ACCESS_KEY_ID=seu_access_key
   export AWS_SECRET_ACCESS_KEY=sua_secret_key
   export AWS_REGION=us-east-1
   ```

### 3️⃣ Instalação Local  
```bash
# Clonar repositório
git clone https://github.com/ifs55/Chatbot-SQL-RAG/tree/main
cd seu-repositorio

# Criar ambiente virtual
python -m venv chatbot-env
source chatbot-env/bin/activate  # Linux/Mac
chatbot-env\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt
```

### 4️⃣ Preparar Base de Conhecimento (RAG)  
```bash
python send_documents_s3.py
```

### 5️⃣ Executar a Aplicação  
```bash
streamlit run chatbot_app.py
```
### 7. Como Usar
Interaja com o chatbot fazendo perguntas em linguagem natural.

Para consultar o banco de dados (SQL):

"Qual a taxa de inadimplência por estado?"

"mostre um gráfico de barras da idade média por classe social"

Para consultar os documentos (RAG):

"Qual a idade mínima para solicitar crédito?"

"Descreva o processo de cobrança em caso de atraso."

---

## 📌 Observações  
- Certifique-se de ter **AWS CLI** configurado localmente.  
- O roteador usa **LLM** para decidir entre SQL e RAG, portanto o custo depende do provedor escolhido.  
- Para grandes volumes de PDFs, considere otimizar a indexação no **ChromaDB**.  

