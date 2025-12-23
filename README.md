# 🤖 RAG Q&A Bot with Groq & Ollama

A powerful Retrieval-Augmented Generation (RAG) chatbot that lets you chat with your PDF documents. Built with **Streamlit**, **LangChain**, **Groq**, and **ChromaDB**.

## 🚀 Features
- **Chat with PDFs**: Ask questions and get accurate answers based *only* on your document content.
- **Persistent Memory**: Uses **ChromaDB** to save embeddings to disk. You only need to embed documents once!
- **Incremental Updates**: Add new PDFs to your existing knowledge base without rebuilding everything.
- **High-Performance LLM**: Uses Groq's super-fast API running **Llama 3.3 70B**.
- **Local Embeddings**: Uses Ollama (`nomic-embed-text`) for high-quality, private document processing.

## 🛠️ Tech Stack
| Component | Tool Used | Why? |
|-----------|-----------|------|
| **Frontend** | Streamlit | Fast, interactive UI building in pure Python. |
| **Framework** | LangChain | Orchestrates the RAG pipeline (retrieval + generation). |
| **LLM** | Groq API | Extremley fast inference for Llama 3 models. |
| **Embeddings** | Ollama | Runs `nomic-embed-text` locally. Free, private, and high accuracy. |
| **Vector Store**| ChromaDB | Persists data to disk unlike FAISS (which is RAM-only). |

## ⚙️ Setup & Installation

1.  **Prerequisites**:
    *   Python 3.13+
    *   [Ollama](https://ollama.com/) installed
    *   Pull the embedding model: `ollama pull nomic-embed-text`

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Variables**:
    Create a `.env` file in the root directory:
    ```env
    GROQ_API_KEY=your_groq_api_key_here
    ```

4.  **Run the App**:
    ```bash
    streamlit run "RAG Dev/chatbot.py"
    ```

## 🧠 How It Works (The RAG Pipeline)
1.  **Ingestion**: Logic checks `documents/` folder for PDFs.
2.  **Chunking**: Splits text into 1000-character chunks (with 200 overlap) to preserve context.
3.  **Embedding**: Converts text to vectors using `nomic-embed-text`.
4.  **Storage**: Saves vectors to `rag_vector_store/` (ChromaDB).
5.  **Retrieval**: When you ask a question, the system finds the most similar chunks.
6.  **Generation**: Sends the question + relevant chunks to Groq (Llama 3.3) to write the answer.

## 💡 Lessons Learned & Optimization
During development, we explored several approaches:

### 1. Vector Store: FAISS vs Chroma
*   **Initial Approach**: Used **FAISS** (Facebook AI Similarity Search).
*   **Problem**: It stores vectors in RAM. Every time the app restarted, we had to re-process all PDFs (slow!).
*   **Solution**: Switched to **Chroma**. It saves to disk, allowing instant 2-second reloading vs 5-minute rebuilding.

### 2. Embeddings: Speed vs Accuracy
We evaluated three options:
*   **HuggingFace (`all-MiniLM-L6-v2`)**: Very fast (CPU optimized), but lower accuracy/nuance.
*   **Ollama (`nomic-embed-text`)**: **Current Choice**. Best balance. High accuracy (8192 token context window) and runs locally.
*   **FastEmbed (`BAAI/bge-small`)**: Identified as a future optimization. It runs "in-process" (no API calls) and is ~50% faster than Ollama, making it ideal for scaling to 1000+ docs.

### 3. Dependency Management
*   Encountered a conflict between `chromadb` (needs Pydantic v2) and `openai-agent` (needs Pydantic v1).
*   **Fix**: Removed unused `openai-agent` dependency to allow upgrading to Pydantic v2, unlocking the latest ChromaDB features.
