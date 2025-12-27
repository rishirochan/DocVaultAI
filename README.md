# 🧠 RAG Cortex

A powerful Retrieval-Augmented Generation (RAG) chatbot that lets you chat with your PDF documents. Built with **Streamlit**, **LangChain**, **Groq**, and **ChromaDB**.

## ✨ Features
- **Chat with PDFs** — Ask questions and get accurate answers based on your documents
- **Dark Notion-style UI** — Clean, minimal dark theme interface
- **Document Management** — Add and delete documents from the sidebar
- **Duplicate Detection** — Warns before re-uploading files already indexed
- **Semantic Chunking** — Splits by topic, not arbitrary character counts
- **Text Preprocessing** — Removes citations, page numbers, and bibliography noise
- **Persistent Memory** — ChromaDB saves embeddings to disk (load in seconds)
- **High-Performance LLM** — Groq API running Llama 3.3 70B
- **Local Embeddings** — Ollama `nomic-embed-text` for private processing

## 🛠️ Tech Stack
| Component | Tool | Why? |
|-----------|------|------|
| **Frontend** | Streamlit | Fast, interactive UI in pure Python |
| **Framework** | LangChain | Orchestrates the RAG pipeline |
| **LLM** | Groq API | Extremely fast inference for Llama 3 |
| **Embeddings** | Ollama | Runs `nomic-embed-text` locally |
| **Vector Store** | ChromaDB | Persists to disk (unlike RAM-only FAISS) |
| **PDF Parser** | PyMuPDF | Better text extraction than PyPDF |

## 📁 Project Structure
```
RAG-Chatbot/
├── src/
│   ├── app.py          # Streamlit UI
│   ├── rag_core.py     # RAG logic
│   └── styles.css      # Dark theme CSS
├── documents/          # Your PDFs go here
├── rag_vector_store/   # ChromaDB persistence
└── .env                # API keys
```

## ⚙️ Setup

1. **Prerequisites**:
   - Python 3.13+
   - [Ollama](https://ollama.com/) installed
   - Pull the embedding model: `ollama pull nomic-embed-text`

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Run the App**:
   ```bash
   cd src
   streamlit run app.py
   ```

## 🧠 How It Works
1. **Ingestion** — Scans `documents/` folder for PDFs (PyMuPDF)
2. **Cleaning** — Removes citations, page numbers, bibliography entries
3. **Chunking** — Semantic splitting by topic shifts, with size limits
4. **Embedding** — Converts text to vectors via `nomic-embed-text`
5. **Storage** — Saves vectors to `rag_vector_store/` (ChromaDB)
6. **Retrieval** — Finds most similar chunks for your question
7. **Generation** — Sends question + context to Groq (Llama 3.3)

## 💡 Lessons Learned

### Vector Store: FAISS vs Chroma
- **FAISS**: Stores in RAM, requires re-processing on restart
- **Chroma**: Persists to disk, instant 2-second reload ✓

### Embeddings: Speed vs Accuracy
- **HuggingFace** (`all-MiniLM-L6-v2`): Fast but lower accuracy
- **Ollama** (`nomic-embed-text`): Best balance, 8192 token context ✓
- **FastEmbed** (`BAAI/bge-small`): Future option for 1000+ docs

### Chunking: Character vs Semantic
| Method | How it works | Pros | Cons |
|--------|--------------|------|------|
| **Character** | Cut every N chars | Simple, fast | Breaks mid-sentence |
| **Semantic** | Split by topic shifts | Coherent chunks | Variable sizes |
| **Recursive Semantic** | Semantic + size limits | Best of both ✓ | More complex |
| **Small-to-Big** | Search small chunks, return parent context | Very precise search + full context | Complex metadata linking |


### PDF Parsing: PyPDF vs PyMuPDF
- **PyPDF**: Simple but breaks text with unusual fonts (`"Ar e W e"`)
- **PyMuPDF**: Handles styled text, fonts, and formatting better ✓

### Text Preprocessing Trade-offs
Regex cleaning removes citations and page numbers but may catch valid content like "Table 1". A **reranker** post-retrieval is planned to filter irrelevant results more intelligently.
