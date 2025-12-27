import streamlit as st
import os
import sys
import rag_core

# Load CSS from external file
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Page Config
st.set_page_config(
    page_title="RAG Cortex",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Session State
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "processing" not in st.session_state:
    st.session_state.processing = False

# Try to auto-load vector store on first run
if st.session_state.vectors is None:
    st.session_state.vectors = rag_core.load_vector_store()

# Sidebar: Document Management
with st.sidebar:
    st.markdown("### 📁 Documents")
    
    # Status
    if st.session_state.processing:
        st.markdown('<p class="status-processing">◉ Processing documents...</p>', unsafe_allow_html=True)
    elif st.session_state.vectors:
        # Get collection info
        try:
            collection = st.session_state.vectors._collection
            chunk_count = collection.count()
            st.markdown(f'<p class="status-active">● Database ready ({chunk_count} chunks)</p>', unsafe_allow_html=True)
        except:
            st.markdown('<p class="status-active">● Database ready</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-inactive">○ No database</p>', unsafe_allow_html=True)
        st.caption("Upload PDFs to get started")
        
        # File uploader for first documents
        uploaded_files = st.file_uploader(
            "Add PDFs",
            type="pdf",
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="initial_upload"
        )
        
        if uploaded_files:
            st.caption(f"{len(uploaded_files)} file(s) ready")
            if st.button("Upload & Initialize", use_container_width=True, type="primary"):
                st.session_state.processing = True
                st.session_state.new_files = uploaded_files
                st.rerun()
    
    st.markdown("---")
    
    # Document Management
    if st.session_state.vectors:
        indexed_docs = rag_core.get_indexed_documents(st.session_state.vectors)
        with st.expander("📋 Manage Documents", expanded=True):
            # Show indexed documents
            if indexed_docs:
                st.caption(f"{len(indexed_docs)} document(s) indexed")
                docs_to_delete = []
                for doc in indexed_docs:
                    if st.checkbox(doc, key=f"del_{doc}"):
                        docs_to_delete.append(doc)
                
                if docs_to_delete:
                    st.warning(f"⚠️ {len(docs_to_delete)} selected for deletion")
                    if st.button("🗑️ Delete Selected", use_container_width=True):
                        st.session_state.processing = True
                        st.session_state.docs_to_delete = docs_to_delete
                        st.rerun()
            
            st.markdown("---")
            
            # File uploader
            uploaded_files = st.file_uploader(
                "Add PDFs",
                type="pdf",
                accept_multiple_files=True,
                label_visibility="collapsed"
            )
            
            if uploaded_files:
                duplicates = [f.name for f in uploaded_files if f.name in indexed_docs]
                new_files = [f for f in uploaded_files if f.name not in indexed_docs]
                
                if duplicates:
                    st.warning(f"⚠️ Duplicate(s): {', '.join(duplicates)}")
                
                if new_files:
                    st.caption(f"{len(new_files)} new file(s)")
                    if st.button("Upload & Process", use_container_width=True, type="primary"):
                        st.session_state.processing = True
                        st.session_state.new_files = new_files
                        st.rerun()
                elif duplicates:
                    st.info("Already indexed")
    
    # Handle deletion
    if st.session_state.processing and hasattr(st.session_state, 'docs_to_delete') and st.session_state.docs_to_delete:
        progress = st.progress(0, text="Deleting documents...")
        deleted_chunks, deleted_files = rag_core.delete_documents_from_store(
            st.session_state.docs_to_delete, st.session_state.vectors
        )
        progress.progress(1.0, text=f"Removed {deleted_chunks} chunks from {deleted_files} file(s)")
        st.session_state.processing = False
        st.session_state.docs_to_delete = None
        st.rerun()
    
    # Handle file upload
    if st.session_state.processing and hasattr(st.session_state, 'new_files') and st.session_state.new_files:
        progress = st.progress(0, text="Saving files...")
        os.makedirs(rag_core.DOCS_DIR, exist_ok=True)
        
        saved_paths = []
        for i, file in enumerate(st.session_state.new_files):
            file_path = os.path.join(rag_core.DOCS_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            saved_paths.append(file_path)
            progress.progress((i + 1) / len(st.session_state.new_files) * 0.5, text=f"Saving {file.name}")
        
        progress.progress(0.6, text="Processing documents...")
        
        if st.session_state.vectors is None:
            vector_store, num_chunks = rag_core.create_vector_store()
            st.session_state.vectors = vector_store
            progress.progress(1.0, text=f"Added {num_chunks} chunks")
        else:
            num_new, _ = rag_core.add_files_to_store(saved_paths, st.session_state.vectors)
            progress.progress(1.0, text=f"Added {num_new} chunks")
        
        st.session_state.processing = False
        st.session_state.new_files = None
        st.rerun()

# Main: Chat Interface
st.markdown("# 🧠 RAG Cortex")
st.markdown('<p class="muted">Ask questions about your documents</p>', unsafe_allow_html=True)

# Input
question = st.text_input(
    "Ask a question",
    placeholder="What would you like to know?",
    label_visibility="collapsed"
)

# Response
if question:
    if st.session_state.vectors is None:
        st.info("Please upload documents to get started (sidebar)")
    else:
        # Check if database has documents
        try:
            chunk_count = st.session_state.vectors._collection.count()
        except:
            chunk_count = 0
        
        if chunk_count == 0:
            st.info("No documents indexed yet. Add PDFs from the sidebar.")
        else:
            with st.spinner(""):
                response = rag_core.get_rag_chain_response(st.session_state.vectors, question)
                answer = response.get("answer", "No answer found.")
            
            st.markdown(f"""
            <div class="response-box">
                {answer}
            </div>
            """, unsafe_allow_html=True)
            
            # Sources
            with st.expander("View sources", expanded=False):
                results = rag_core.get_similarity_scores(st.session_state.vectors, question)
                for i, (doc, score) in enumerate(results):
                    st.markdown(f"**Source {i+1}** (relevance: {score:.2f})")
                    st.caption(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                    st.markdown("---")