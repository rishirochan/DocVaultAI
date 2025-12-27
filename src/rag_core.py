import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma 
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

# Define Project Roots
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) # Go up from RAG_Dev to root
DOCS_DIR = os.path.join(PROJECT_ROOT, "documents")
VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, "rag_vector_store")
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

# Load environment variables
load_dotenv(ENV_PATH)

def get_llm():
    """Initialize and return the Groq LLM."""
    # Ensure env is loaded
    load_dotenv(ENV_PATH)
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError(f"GROQ_API_KEY not found. Checked: {ENV_PATH}")
    return ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile")

def get_embeddings():
    """Initialize and return the Ollama embeddings model."""
    return OllamaEmbeddings(model="nomic-embed-text")

def clean_document_text(text):
    """
    Clean document text by removing:
    - Inline citations: [1], [2,3], [1-5]
    - Author-year citations: (Smith, 2020), (Smith et al., 2020)
    - Page numbers (standalone lines with just numbers)
    - Page headers like "Page 5 of 10"
    - Reference section markers and bibliography entries
    """
    # Remove inline numeric citations: [1], [2,3], [1-5], [12]
    text = re.sub(r'\[\d+(?:[\-,]\s*\d+)*\]', '', text)
    
    # Remove author-year citations: (Smith, 2020), (Smith et al., 2020)
    text = re.sub(r'\([A-Z][a-z]+\s+et\s+al\.?,?\s*\d{4}\)', '', text)
    text = re.sub(r'\([A-Z][a-z]+,?\s*\d{4}\)', '', text)
    
    # Remove standalone page numbers (lines that are just 1-3 digits)
    text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)
    
    # Remove "Page X" or "Page X of Y" patterns
    text = re.sub(r'[Pp]age\s+\d+(\s+of\s+\d+)?', '', text)
    
    # Remove common reference section headers (entire line)
    text = re.sub(r'^(References|Bibliography|Works Cited|Citations)\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove bibliography-style entries (lines that look like author lists)
    # Pattern: "LastName, F. M." or "LastName, FirstName" type entries
    text = re.sub(r'^[A-Z][a-z]+,\s*[A-Z]\..*$', '', text, flags=re.MULTILINE)
    
    # Remove lines that are mostly author initials like "Wang, L. C. Lima, C."
    text = re.sub(r'^[A-Z][a-z]+,\s*[A-Z]\.\s*[A-Z]?\.?.*$', '', text, flags=re.MULTILINE)
    
    # Remove arXiv/DOI references
    text = re.sub(r'arXiv:\d+\.\d+', '', text)
    text = re.sub(r'doi:\s*\S+', '', text, flags=re.IGNORECASE)
    
    # Clean up excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()

def get_semantic_chunks(documents, max_chunk_size=1500, min_chunk_size=200):
    """
    Recursive Semantic Chunking:
    1. Clean text (remove citations, page numbers, references)
    2. Split by topic shifts using SemanticChunker
    3. Merge tiny chunks, split oversized chunks
    """
    # Step 0: Clean each document's text
    for doc in documents:
        doc.page_content = clean_document_text(doc.page_content)
    
    # Filter out documents with very little content after cleaning
    documents = [doc for doc in documents if len(doc.page_content.strip()) > 50]
    
    if not documents:
        return []
    
    embeddings = get_embeddings()
    
    # Step 1: Semantic split by topic (higher threshold = fewer, larger chunks)
    semantic_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95.0,  # Higher = fewer splits
        buffer_size=3  # Group more sentences together
    )
    semantic_chunks = semantic_splitter.split_documents(documents)
    
    # Step 2: Handle chunk sizes
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    final_chunks = []
    accumulated_text = ""
    accumulated_metadata = None
    
    for chunk in semantic_chunks:
        content = chunk.page_content.strip()
        
        # Skip empty chunks
        if len(content) < 10:
            continue
        
        # If chunk is too small, accumulate it
        if len(content) < min_chunk_size:
            accumulated_text += "\n\n" + content if accumulated_text else content
            if accumulated_metadata is None:
                accumulated_metadata = chunk.metadata
        else:
            # First, flush any accumulated small chunks
            if accumulated_text:
                merged_doc = Document(page_content=accumulated_text, metadata=accumulated_metadata or {})
                final_chunks.append(merged_doc)
                accumulated_text = ""
                accumulated_metadata = None
            
            # Handle current chunk
            if len(content) > max_chunk_size:
                final_chunks.extend(fallback_splitter.split_documents([chunk]))
            else:
                final_chunks.append(chunk)
    
    # Flush any remaining accumulated text
    if accumulated_text and len(accumulated_text) >= min_chunk_size:
        merged_doc = Document(page_content=accumulated_text, metadata=accumulated_metadata or {})
        final_chunks.append(merged_doc)
    
    return final_chunks

def load_vector_store(persist_dir=VECTOR_STORE_DIR):
    """
    Attempt to load an existing Chroma vector store from disk.
    Returns the vector store if successful, else None.
    """
    if os.path.exists(persist_dir):
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=get_embeddings(),
            collection_metadata={"hnsw:space": "cosine"}
        )
        return vector_store
    return None

def create_vector_store(doc_dir=DOCS_DIR, persist_dir=VECTOR_STORE_DIR):
    """
    Create a new Chroma vector store from PDFs in the specified directory.
    Returns the new vector store.
    """
    # Load all PDFs in directory using PyMuPDF
    documents = []
    for filename in os.listdir(doc_dir):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(doc_dir, filename)
            loader = PyMuPDFLoader(file_path)
            documents.extend(loader.load())
    
    if not documents:
        return None, 0

    final_documents = get_semantic_chunks(documents)
    
    vector_store = Chroma.from_documents(
        documents=final_documents,
        embedding=get_embeddings(),
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    return vector_store, len(final_documents)

def add_documents_to_store(doc_dir, vector_store):
    """
    Load PDFs from doc_dir, split them, and add to the existing vector_store.
    Returns the number of chunks added.
    """
    # Load all PDFs in directory using PyMuPDF
    documents = []
    for filename in os.listdir(doc_dir):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(doc_dir, filename)
            loader = PyMuPDFLoader(file_path)
            documents.extend(loader.load())
    
    if not documents:
        return 0
    
    final_documents = get_semantic_chunks(documents)
    
    vector_store.add_documents(final_documents)
    return len(final_documents)

def get_indexed_documents(vector_store):
    """
    Get list of unique document filenames from the vector store.
    Returns a list of document filenames.
    """
    try:
        collection = vector_store._collection
        existing_data = collection.get(include=["metadatas"])
        sources = set()
        for meta in existing_data.get("metadatas", []):
            if meta and "source" in meta:
                sources.add(os.path.basename(meta["source"]))
        return sorted(list(sources))
    except:
        return []

def delete_documents_from_store(filenames_to_delete, vector_store, doc_dir=DOCS_DIR):
    """
    Delete documents from the vector store by filename.
    Also removes the PDF files from disk.
    Returns tuple: (num_deleted_chunks, num_deleted_files)
    """
    deleted_chunks = 0
    deleted_files = 0
    
    try:
        collection = vector_store._collection
        
        # Get all documents from collection
        all_data = collection.get(include=["metadatas"])
        
        for filename in filenames_to_delete:
            ids_to_delete = []
            for i, meta in enumerate(all_data.get("metadatas", [])):
                if meta and "source" in meta:
                    # Match by basename (filename only)
                    if os.path.basename(meta["source"]) == filename:
                        ids_to_delete.append(all_data["ids"][i])
            
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                deleted_chunks += len(ids_to_delete)
            
            # Delete PDF from disk
            file_path = os.path.join(doc_dir, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted_files += 1
    except Exception as e:
        print(f"Error deleting documents: {e}")
    
    return deleted_chunks, deleted_files

def add_files_to_store(file_paths, vector_store):
    """
    Load specific PDF files, split them, and add to the existing vector_store.
    Skips files that are already indexed (checks by filename).
    Returns tuple: (chunks_added, files_skipped)
    """
    # Get already indexed filenames from vector store metadata
    try:
        collection = vector_store._collection
        existing_data = collection.get(include=["metadatas"])
        indexed_sources = set()
        for meta in existing_data.get("metadatas", []):
            if meta and "source" in meta:
                indexed_sources.add(os.path.basename(meta["source"]))
    except:
        indexed_sources = set()
    
    all_documents = []
    skipped = 0
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        if filename in indexed_sources:
            skipped += 1
            continue
        loader = PyMuPDFLoader(file_path)
        all_documents.extend(loader.load())
    
    if not all_documents:
        return 0, skipped
    
    final_documents = get_semantic_chunks(all_documents)
    
    vector_store.add_documents(final_documents)
    return len(final_documents), skipped

def get_rag_chain_response(vector_store, question):
    """
    Create the RAG chain and invoke it with the question.
    Returns the full response dictionary.
    """
    llm = get_llm()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    # Retrieve relevant documents
    docs = retriever.invoke(question)
    
    # Format context from documents
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create prompt and get response
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question based on the context provided only. Be accurate and concise."),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    
    return {
        "answer": response.content,
        "context": docs
    }

def get_similarity_scores(vector_store, question, k=10):
    """
    Perform similarity search with scores and deduplicate results.
    Returns a list of tuples (document, score) with unique content.
    """
    results = vector_store.similarity_search_with_relevance_scores(question, k=k)
    
    # Deduplicate by content (keep first occurrence = highest score)
    seen_content = set()
    unique_results = []
    for doc, score in results:
        # Use first 200 chars as fingerprint to catch near-duplicates
        fingerprint = doc.page_content[:200].strip()
        if fingerprint not in seen_content:
            seen_content.add(fingerprint)
            unique_results.append((doc, score))
    
    return unique_results[:3]  # Return max 3 unique results