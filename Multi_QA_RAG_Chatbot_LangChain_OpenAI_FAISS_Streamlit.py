import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
import tempfile
from io import StringIO
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Streamlit page config
st.set_page_config(page_title=" Chat with PDF", page_icon="🤖")
st.title("Chat with Your PDF using RAG + OpenAI")

# Sidebar: API key + PDF upload
openai_key = st.sidebar.text_input(" Enter OpenAI API Key", type="password")
uploaded_file = st.sidebar.file_uploader("📎 Upload PDF", type=["pdf"])

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# RAG pipeline builder
def build_rag_chain(file_path, openai_key):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say you don't know.
Use ten sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(openai_api_key=openai_key, streaming=True)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# Build the chain after file upload
if uploaded_file and openai_key and st.session_state.rag_chain is None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name
        st.session_state.rag_chain = build_rag_chain(file_path, openai_key)
        st.sidebar.success("✅ PDF loaded and RAG chain is ready!")

# Helpful messages if inputs missing
if not uploaded_file:
    st.info(" Please upload a PDF to begin.")
elif not openai_key:
    st.info(" Please enter your OpenAI API key in the sidebar.")

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input + streaming output
if st.session_state.rag_chain:
    user_input = st.chat_input("Ask a question about your PDF...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            response = st.write_stream(st.session_state.rag_chain.stream(user_input))
    

        st.session_state.messages.append({"role": "assistant", "content": response})
