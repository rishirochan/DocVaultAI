"""
RAG Q&A Bot - Answer questions from PDF documents using Chroma vector store

How it works:
1. PDFs are loaded from ../documents/ folder
2. Documents are split into chunks and embedded using Ollama (nomic-embed-text)
3. Embeddings are stored in Chroma vector database (persisted to disk)
4. User questions are embedded and matched against document chunks
5. Relevant chunks are sent to Groq LLM to generate answers
"""

import streamlit as st
import os

# LangChain components for RAG pipeline
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import FAISS  
from langchain_community.vectorstores import Chroma 
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables (.env file must contain GROQ_API_KEY)
from dotenv import load_dotenv
load_dotenv()

# Initialize Groq LLM
# Using llama-3.3-70b-versatile: good balance of speed and quality for RAG tasks
groq_api_key = os.getenv("GROQ_API_KEY")
chatgroq = ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile")

# Define RAG prompt template
# Instructs LLM to answer ONLY based on provided context (prevents hallucination)
prompt = ChatPromptTemplate.from_template(
    '''
    Answer the question based on the context provided only.
    Please provide the most accurate response based on the question.

    Context 
    <context>
    {context}
    </context>
    Question: {input}
    '''
)

st.title("RAG Q&A Bot")

def create_vectors_embedding():
    """
    Load existing vector DB or create new one from PDFs
    
    Persistence benefit: After first run, loading takes 2-3 seconds vs 2-5 minutes
    Only re-embed if you delete ../rag_vector_store/ or add new PDFs using the button
    """
    if "vectors" not in st.session_state:
        # Initialize Ollama embeddings (runs locally, no API calls)
        # Model: nomic-embed-text - optimized for semantic search
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        persist_dir = "../rag_vector_store"
        
        if os.path.exists(persist_dir):
            # Load existing database from disk (FAST - no re-embedding needed!)
            st.info("📂 Loading existing vector database from disk...")
            st.session_state.vectors = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings
            )
            st.success("✅ Vector database loaded!")
        else:
            # Create new database from PDFs (SLOW - first time only)
            st.info("🔄 Creating new vector database from PDFs (this may take a few minutes)...")
            loaders = PyPDFDirectoryLoader("../documents")
            documents = loaders.load()
            
            if not documents:
                st.error("No PDF documents found! Please add PDF files to the documents folder.")
                return
            
            # Split documents into chunks for better retrieval
            # chunk_size=1000: Each chunk is ~1000 characters (balance context vs precision)
            # chunk_overlap=200: Overlap prevents splitting important context across chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_documents = text_splitter.split_documents(documents)
            
            # Create Chroma database and persist to disk
            # This saves the embeddings so we don't need to re-compute them
            st.session_state.vectors = Chroma.from_documents(
                documents=final_documents,
                embedding=embeddings,
                persist_directory=persist_dir
            )
            st.success(f"✅ Created and saved vector database with {len(final_documents)} chunks!")
        
        # Create retriever for similarity search (finds most relevant chunks)
        st.session_state.retriever = st.session_state.vectors.as_retriever()

# OLD FAISS CODE (kept for reference):
# def create_vectors_embedding():
#     if "vectors" not in st.session_state:
#         st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")
#         st.session_state.loaders = PyPDFDirectoryLoader("../documents")
#         st.session_state.documents = st.session_state.loaders.load()
#         
#         if not st.session_state.documents:
#             st.error("No PDF documents found! Please add PDF files to the documents folder.")
#             return
#         
#         st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.documents)
#         st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
#         st.session_state.retriever = st.session_state.vectors.as_retriever()

# User input
question = st.text_input("Enter your query from the research paper")

# Button: Load or create vector database
if st.button("Document Embedding"):
    create_vectors_embedding()
    st.write("Vector database is ready")

# Button: Add new documents to existing database
if st.button("Add New Documents"):
    if "vectors" not in st.session_state:
        st.warning("Please click 'Document Embedding' button first to create the database!")
    else:
        st.info("Scanning for documents to add...")
        loaders = PyPDFDirectoryLoader("../documents")
        documents = loaders.load()
        
        if not documents:
            st.error("No PDF documents found in the documents folder!")
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_documents = text_splitter.split_documents(documents)
            
            # Add to existing database
            st.session_state.vectors.add_documents(final_documents)
            st.success(f"✅ Added {len(final_documents)} new chunks to the database!")
            
            st.session_state.retriever = st.session_state.vectors.as_retriever()

# Answer question using RAG (Retrieval-Augmented Generation)
if question:
    if "vectors" not in st.session_state:
        st.warning("Please click 'Document Embedding' button first to load and process the documents!")
    else:
        # Step 1: Create document chain (combines retrieved docs with LLM)
        document_chain = create_stuff_documents_chain(chatgroq, prompt)
        
        # Step 2: Get retriever (finds relevant document chunks via similarity search)
        retriever = st.session_state.vectors.as_retriever()  
        
        # Step 3: Create retrieval chain (full RAG pipeline)
        # This will: 1) Embed question, 2) Find similar chunks, 3) Send to LLM with prompt
        retrieval_chain = create_retrieval_chain(retriever, document_chain)  
        
        # Step 4: Get answer from the RAG pipeline
        response = retrieval_chain.invoke({"input": question})
        
        # Display the generated answer
        st.write(response['answer'])
        
        # Show which document chunks were used (for transparency)
        with st.expander('Document similarity search'):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('-----------------------')