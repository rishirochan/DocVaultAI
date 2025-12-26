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
        persist_dir = "./rag_vector_store"
        
        if os.path.exists(persist_dir):
            # Load existing database from disk (FAST - no re-embedding needed!)
            st.info("📂 Loading existing vector database from disk...")
            st.session_state.vectors = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings,
                collection_metadata={"hnsw:space": "cosine"} # CHANGE: Enforce Cosine Similarity
            )
            st.success("✅ Vector database loaded!")
        else:
            # Create new database from PDFs (SLOW - first time only)
            st.info("🔄 Creating new vector database from PDFs (this may take a few minutes)...")
            loaders = PyPDFDirectoryLoader("./documents") # CHANGE: Path relative to execution root
            documents = loaders.load()
            
            if not documents:
                st.error(f"No PDF documents found in './documents'. Current working directory: {os.getcwd()}")
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
                persist_directory=persist_dir,
                collection_metadata={"hnsw:space": "cosine"} # CHANGE: Enforce Cosine Similarity
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

# User input & Embedding Button Layout
# vertical_alignment="bottom" ensures the button sits on the same baseline as the text input
col1, col2 = st.columns([4, 1], vertical_alignment="bottom")

with col1:
    question = st.text_input("Enter your query from the research paper", label_visibility="visible")

with col2:
    # Button: Load or create vector database (Init)
    if st.button("Document Embedding", use_container_width=True):
        create_vectors_embedding()
        st.toast("✅ Vector database is ready!", icon="🦅") # Ephemeral toast instead of permanent text

# File Upload Section (Clean separator)
st.divider()

with st.expander("📂 Add New PDFs"):
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        if st.button("Process Uploaded PDFs", use_container_width=True):
            if "vectors" not in st.session_state:
                st.toast("⚠️ Please create the database first!", icon="🚫")
            else:
                progress_text = "Operation in progress. Please wait..."
                my_bar = st.progress(0, text=progress_text)
                
                # Save uploaded files to ./documents
                save_dir = "./documents"
                os.makedirs(save_dir, exist_ok=True)
                
                saved_files_count = 0
                for i, uploaded_file in enumerate(uploaded_files):
                    file_path = os.path.join(save_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_files_count += 1
                    my_bar.progress((i + 1) / len(uploaded_files))
                
                my_bar.empty()
                st.toast(f"Saved {saved_files_count} files!", icon="💾")
                
                # Now load and embed them
                with st.spinner("Embedding new documents..."):
                    loaders = PyPDFDirectoryLoader(save_dir)
                    documents = loaders.load()
                    
                    if not documents:
                       st.error("Something went wrong. No documents loaded.")
                    else:
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        final_documents = text_splitter.split_documents(documents)
                        
                        # Add to existing database
                        st.session_state.vectors.add_documents(final_documents)
                        st.session_state.retriever = st.session_state.vectors.as_retriever(search_type="similarity", search_kwargs={"k": 3})
                        
                        st.toast(f"Added {len(final_documents)} new chunks!", icon="✅")

# Answer question using RAG (Retrieval-Augmented Generation)
if question:
    if "vectors" not in st.session_state:
        st.warning("Please click 'Document Embedding' button first to load and process the documents!")
    else:
        # Step 1: Create document chain (combines retrieved docs with LLM)
        document_chain = create_stuff_documents_chain(chatgroq, prompt)
        
        # Step 2: Get retriever
        retriever = st.session_state.vectors.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        
        # Step 3: Create retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)  
        
        # Step 4: Get answer
        response = retrieval_chain.invoke({"input": question})
        
        # Display Answer FIRST (User Request)
        st.markdown("### 🤖 Answer")
        st.write(response['answer'])
        st.write("-----------------------")

        # Display Context & Scores SECOND
        st.markdown("### 🔍 Context & Similarity Scores")
        
        # We need to re-run the score search to get the numbers (since retrieval_chain doesn't return scores)
        results_with_scores = st.session_state.vectors.similarity_search_with_relevance_scores(question, k=3)
        
        for i, (doc, score) in enumerate(results_with_scores):
            with st.expander(f"Rank {i+1} (Score: {score:.4f})"):
                st.write(doc.page_content)
                st.markdown(f"**Similarity Score:** `{score:.4f}`")