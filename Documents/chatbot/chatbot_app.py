import streamlit as st
import pandas as pd
import boto3
import openai 
import time
import json
import plotly.express as px
from langchain_community.document_loaders import S3FileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# --- CONFIGURATION ---
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    OPENAI_API_KEY = config.get('openai_api_key')
    if not OPENAI_API_KEY:
        raise ValueError("The 'openai_api_key' key was not found in config.json")
except FileNotFoundError:
    raise FileNotFoundError("The configuration file 'config.json' was not found. Please create the file.")


GLUE_DATABASE = "chatbot_db"
GLUE_TABLE = "dataset"
S3_OUTPUT_LOCATION = "s3://chatbot-analise-dados/athena_results/" 
openai.api_key = OPENAI_API_KEY 

@st.cache_resource
def load_vector_store():
    print("Loading embedding model and Vector Store...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vector_store = Chroma(persist_directory="./chroma_db_rag", embedding_function=embeddings)
    print("✅ Vector Store loaded.")
    return vector_store

vector_store = load_vector_store()

def decide_tool(question):
    """Use the LLM to decide whether the question is for SQL or for documents."""
    prompt = f"""
    Sua tarefa é classificar a pergunta do usuário e decidir qual ferramenta usar.
    As ferramentas disponíveis são:
    1. 'SQL': Para perguntas sobre dados quantitativos, agregações, médias, contagens, taxas de inadimplência, etc., que podem ser respondidas com uma consulta SQL.
    2. 'DOCUMENTO': Para perguntas sobre políticas, regras, definições, explicações ou informações qualitativas que provavelmente estão em um documento de texto.

    Exemplos:
    - Pergunta: "qual a taxa de inadimplência por uf?" -> Ferramenta: SQL
    - Pergunta: "quais são os critérios para aprovação de crédito?" -> Ferramenta: DOCUMENTO
    - Pergunta: "qual a idade média dos clientes de MG?" -> Ferramenta: SQL
    - Pergunta: "explique a política de renegociação de dívida." -> Ferramenta: DOCUMENTO

    Analise a seguinte pergunta e retorne APENAS a palavra 'SQL' ou 'DOCUMENTO'.
    Pergunta do usuário: "{question}"
    Ferramenta:
    """
    response = openai.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=5)
    return response.choices[0].message.content.strip()

def answer_with_rag(question, vector_store):
    """Run the RAG flow to answer a question."""
    # 1. Retrieve relevant documents
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(question)
    
    context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])

    # 2. Generate answer using the context
    prompt = f"""
    Você é um assistente especialista que responde perguntas com base no contexto fornecido.
    Use APENAS as informações do contexto abaixo para responder à pergunta.
    Se a resposta não estiver no contexto, diga "Não encontrei informações sobre isso nos meus documentos."

    **Contexto:**
    {context}

    **Pergunta:**
    {question}

    **Resposta:**
    """
    response = openai.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}], temperature=0.3)
    return response.choices[0].message.content

def generate_summary_with_llm(question, df):
    """Generate a natural language summary from a DataFrame."""
    if df.empty:
        return "Não há dados para resumir."
        
    df_string = df.to_csv(index=False)
    
    prompt = f"""
    Você é um analista de dados sênior.
    A pergunta original do usuário foi: "{question}"
    Os dados resultantes da consulta SQL são os seguintes (em formato CSV):
    ---
    {df_string}
    ---
    Com base nesses dados e na pergunta original, escreva um resumo conciso (2-3 frases) explicando o resultado para o usuário em português.
    Seja direto e foque nos insights principais.
    """
    response = openai.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}], temperature=0.5, max_tokens=200)
    return response.choices[0].message.content

def generate_plot_code_with_llm(question, df):
    """
    Generate Python code with Plotly to create a chart, stripping extra text and formatting.
    """
    if df.empty:
        return None
        
    df_string = df.to_csv(index=False)
    
    prompt = f"""
    Você é um especialista em visualização de dados usando Plotly em Python.
    A pergunta original do usuário foi: "{question}".
    Os dados para plotar estão no DataFrame abaixo (em formato CSV), que será carregado em uma variável chamada `df`.
    ---
    {df_string}
    ---
    Com base na pergunta e nos dados, gere APENAS o código Python para criar uma figura com Plotly Express.
    - O código deve começar com 'import plotly.express as px'.
    - Use a variável `df` que já contém os dados.
    - Atribua a figura final a uma variável chamada `fig`.
    - Não inclua explicações, textos extras ou blocos de código. APENAS O CÓDIGO.
    - Exemplo: `import plotly.express as px\nfig = px.bar(df, x='uf', y='taxa_inadimplencia', title='Taxa de Inadimplência por UF')`
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4", 
            messages=[{"role": "user", "content": prompt}], 
            temperature=0, 
            max_tokens=300
        )
        
        raw_text = response.choices[0].message.content

        # Clean up potential markdown code fences
        if "```python" in raw_text:
            python_code = raw_text.split("```python")[1].split("```")[0].strip()
        elif "'''python" in raw_text:
            python_code = raw_text.split("'''python")[1].split("'''")[0].strip()
        elif "```" in raw_text:
            parts = raw_text.split("```")
            python_code = parts[1] if len(parts) > 1 else parts[0]
        else:
            python_code = raw_text.strip()
            
        return python_code

    except Exception as e:
        print(f"❌ Error from LLM when generating plot code: {e}")
        return None

def generate_sql_with_llm(question):
    """
    Convert the user question into an SQL query for AWS Athena, removing markdown formatting.
    """
    prompt = f"""
    Você é um assistente especialista em SQL que escreve queries para o AWS Athena.
    Sua tarefa é converter a pergunta do usuário em uma única query SQL.
    **Nome da Tabela:** "{GLUE_TABLE}"
    **Esquema Final da Tabela:**
    - data_referencia (timestamp)
    - inadimplente (bigint)
    - sexo (string)
    - idade (double)
    - flag_obito (binary)
    - uf (string)
    - classe_social (string)
    **Instruções Cruciais:**
    1. Gere APENAS a query SQL. Sem explicações.
    2. Para "taxa de inadimplência", use AVG(CAST(inadimplente AS DOUBLE)).
    3. Use aspas duplas ("") para se referir à tabela: FROM "{GLUE_TABLE}".
    **Pergunta do Usuário:** "{question}"
    **Sua query SQL:**
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4", 
            messages=[{"role": "user", "content": prompt}], 
            temperature=0, 
            max_tokens=250
        )
        
        raw_text = response.choices[0].message.content
        
        if "```" in raw_text:
            parts = raw_text.split("```")
            sql_query = parts[1] if len(parts) > 1 else parts[0]
        elif "'''" in raw_text:
            parts = raw_text.split("'''")
            sql_query = parts[1] if len(parts) > 1 else parts[0]
        else:
            sql_query = raw_text

        if sql_query.lower().strip().startswith('sql'):
            sql_query = sql_query.strip()[3:].strip()

        return sql_query.replace(';', '').strip()

    except Exception as e:
        print(f"❌ Error from LLM: {e}")
        return None

def execute_athena_query(query):
    """
    Execute a query on Athena and return a tuple (DataFrame, Error).
    """
    athena_client = boto3.client('athena', region_name='sa-east-1')
    try:
        response = athena_client.start_query_execution(
            QueryString=query,
            QueryExecutionContext={'Database': GLUE_DATABASE},
            ResultConfiguration={'OutputLocation': S3_OUTPUT_LOCATION}
        )
        query_execution_id = response['QueryExecutionId']
        state = 'RUNNING'
        while state in ['RUNNING', 'QUEUED']:
            time.sleep(1)
            result_status = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
            state = result_status['QueryExecution']['Status']['State']
            
            if state == 'FAILED':
                error_message = result_status['QueryExecution']['Status']['StateChangeReason']
                return None, f"Athena query failed: {error_message}"
            elif state == 'CANCELLED':
                return None, "The query was cancelled."

        results_response = athena_client.get_query_results(QueryExecutionId=query_execution_id)
        rows = results_response['ResultSet']['Rows']
        
        if not rows or len(rows) < 2:
            return pd.DataFrame(), None 
        
        header = [col['VarCharValue'] for col in rows[0]['Data']]
        data = [[item.get('VarCharValue') for item in row['Data']] for row in rows[1:]]
        
        return pd.DataFrame(data, columns=header), None 

    except Exception as e:
        return None, f"An error occurred while communicating with Athena: {e}"


# --- STREAMLIT INTERFACE ---

st.set_page_config(page_title="Chatbot de Análise de Dados", layout="wide")
st.title("🤖 Chatbot de Análise de Dados com AWS Athena")

# Initialize chat history in session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], pd.DataFrame):
            st.dataframe(message["content"])
        else:
            st.markdown(message["content"])

# Capture new user question
if prompt := st.chat_input("Faça sua pergunta sobre os dados ou documentos..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analisando sua pergunta..."):
            chosen_tool = decide_tool(prompt)
            st.info(f"Ferramenta escolhida: **{chosen_tool}**")

        if chosen_tool == "SQL":
            with st.spinner("Gerando SQL e consultando o Athena..."):
                sql_query = generate_sql_with_llm(prompt)
                st.markdown(f"**SQL Gerado:**\n```sql\n{sql_query}\n```")
                df_result, error = execute_athena_query(sql_query)

            if error:
                st.error(f"Ocorreu um erro: {error}")
            elif not df_result.empty:
                st.success("Consulta SQL concluída!")

                summary = generate_summary_with_llm(prompt, df_result)
                st.markdown(summary)
                st.session_state.messages.append({"role": "assistant", "content": summary})

                chart_keywords = ['gráfico', 'visualização', 'plot', 'desenhe', 'mostre um gráfico']
                if any(keyword in prompt.lower() for keyword in chart_keywords):
                    with st.spinner("Gerando visualização..."):
                        chart_code = generate_plot_code_with_llm(prompt, df_result)
                        
                        if chart_code:
                            st.markdown(f"**Código do Gráfico Gerado:**\n```python\n{chart_code}\n```")
                            try:
                                namespace = {'df': df_result, 'px': px, 'fig': None}
                                exec(chart_code, namespace)
                                fig = namespace.get('fig')
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("O código do gráfico foi gerado, mas não criou um objeto 'fig'.")
                            except Exception as e:
                                st.error(f"Erro ao executar o código do gráfico: {e}")
                else:
                    st.dataframe(df_result)
                    st.session_state.messages.append({"role": "assistant", "content": df_result})
            else:
                st.warning("A consulta SQL não retornou resultados.")
        
        elif chosen_tool == "DOCUMENTO":
            with st.spinner("Buscando nos documentos e gerando resposta..."):
                rag_answer = answer_with_rag(prompt, vector_store)
                st.markdown(rag_answer)
                st.session_state.messages.append({"role": "assistant", "content": rag_answer})
        
        else:
            st.error("Não consegui decidir qual ferramenta usar. Por favor, reformule a pergunta.")
