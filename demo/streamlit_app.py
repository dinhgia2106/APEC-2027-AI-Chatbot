#!/usr/bin/env python3
"""
APEC AI Chatbot - Streamlit Interface

A simple web application for the APEC chatbot using Streamlit.
Ensure you have completed the following steps before running:
1. python backend/data/crawler.py (to collect data)
2. python backend/Embed/embed_data.py (to create embeddings)
3. python backend/VectorStore/build_faiss_index.py (to build the FAISS index)
4. Start the Ollama server with LLaMA3: ollama serve

Run the application:
    streamlit run demo/streamlit_app.py
"""

import sys
import streamlit as st
from pathlib import Path

# Ensure the project root (one level above 'demo') is on the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from backend.rag_pipeline import RAGPipeline

# Page configuration
st.set_page_config(
    page_title="APEC AI Chatbot",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS for custom styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #e6e6e6;
        margin-bottom: 2rem;
    }
    
    .quick-buttons {
        display: flex;
        gap: 10px;
        margin-bottom: 20px;
        flex-wrap: wrap;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }
    
    .user-message {
        background-color: #f0f2f6;
        border-left: 4px solid #0068c9;
    }
    
    .assistant-message {
        background-color: #ffffff;
        border-left: 4px solid #00c851;
        border: 1px solid #e6e6e6;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_rag_pipeline():
    """Initializes the RAG Pipeline with caching to prevent reloading."""
    try:
        with st.spinner("Initializing RAG Pipeline..."):
            return RAGPipeline()
    except Exception as e:
        st.error(f"Error initializing RAG Pipeline: {e}")
        st.error("Please ensure you have run the data preparation steps:")
        st.error("1. python backend/data/crawler.py")
        st.error("2. python backend/Embed/embed_data.py") 
        st.error("3. python backend/VectorStore/build_faiss_index.py")
        st.stop()

def display_chat_message(role, content):
    """Displays a chat message with custom styling."""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong> {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>Chatbot:</strong> {content}
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main function for the Streamlit application."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>APEC 2027 AI Chatbot</h1>
        <p>An intelligent AI assistant for APEC 2025 information in South Korea</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize the RAG Pipeline
    rag_pipeline = load_rag_pipeline()
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "show_context" not in st.session_state:
        st.session_state.show_context = False
    
    # Sidebar with options
    with st.sidebar:
        st.header("Options")
        
        # Toggle to show context
        st.session_state.show_context = st.checkbox(
            "Show reference context", 
            value=st.session_state.show_context
        )
        
        # Button to clear chat history
        if st.button("Clear chat history"):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("**User Guide:**")
        st.markdown("- Ask a question about APEC 2025.")
        st.markdown("- Use the quick buttons for common queries.")
        st.markdown("- Enable 'Show reference context' to see the source data.")
    
    # Quick action buttons
    st.subheader("Frequently Asked Questions")
    
    col1, col2, col3 = st.columns(3)
    
    quick_questions = {
        "Today's Schedule": "What is the schedule for today?",
        "Directions to Busan": "How do I get to Busan?",
        "About APEC 2025": "Tell me about APEC 2025"
    }
    
    # A helper function to avoid code repetition for buttons
    def handle_quick_question(user_input):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Searching for information..."):
            try:
                result = rag_pipeline.answer(user_input)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": result["answer"],
                    "context": result.get("context", "")
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"Sorry, an error occurred: {e}",
                    "context": ""
                })
        st.rerun()

    with col1:
        if st.button("Today's Schedule", use_container_width=True):
            handle_quick_question(quick_questions["Today's Schedule"])
    
    with col2:
        if st.button("Directions to Busan", use_container_width=True):
            handle_quick_question(quick_questions["Directions to Busan"])
    
    with col3:
        if st.button("About APEC 2025", use_container_width=True):
            handle_quick_question(quick_questions["About APEC 2025"])
    
    st.markdown("---")
    
    # Chat area
    st.subheader("Chat with AI")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            display_chat_message(message["role"], message["content"])
            
            # Display context if toggled on and context exists
            if (st.session_state.show_context and 
                message["role"] == "assistant" and 
                "context" in message and 
                message["context"]):
                with st.expander("Reference Context"):
                    st.text(message["context"])
    
    # Input for a new message
    user_input = st.chat_input("Ask a question about APEC 2025...")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Process and add assistant's response
        with st.spinner("Searching for information..."):
            try:
                result = rag_pipeline.answer(user_input)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": result["answer"],
                    "context": result.get("context", "")
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"Sorry, an error occurred: {e}",
                    "context": ""
                })
        
        # Rerun the app to display the new message
        st.rerun()

if __name__ == "__main__":
    main()