import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Bot", page_icon="⚖️")
st.title("Consulta Normativa")

# Recuperar API Key de Secrets de Streamlit
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

@st.cache_resource
def init():
    if not os.path.exists("documento.pdf"):
        return None
    loader = PyPDFLoader("documento.pdf")
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())
    return vectorstore.as_retriever()

retriever = init()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Escribe tu duda aquí:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        template = "Responde usando solo este contexto: {context}\n\nPregunta: {question}"
        prompt_tpl = ChatPromptTemplate.from_template(template)
        
        chain = (
            RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
            | {
                "answer": (lambda x: {"context": "\n".join(d.page_content for d in x["context"]), "question": x["question"]}) | prompt_tpl | llm | StrOutputParser(),
                "sources": lambda x: x["context"]
            }
        )
        
        res = chain.invoke(prompt)
        st.markdown(res["answer"])
        st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
