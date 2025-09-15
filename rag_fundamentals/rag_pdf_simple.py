import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import os
from dotenv import load_dotenv
import PyPDF2       # dealing, managing, extracting,... with pdf file
import uuid

load_dotenv()

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

class SimpleModelSelector:
    """Simple class to handle model selection"""

    def __init__(self):
        # available LLM models
        self.llm_models = {
            "openai": "gpt-4o",
            "ollama": "llama3.2:latest"
        }

        self.embedding_models = {
            "openai": {
                "name": "OpenAI Embeddings",
                "dimensions": 1536,
                "model_name": "text-embedding-3-small",
            },
            "chroma": {"name": "Chroma Default", "dimensions": 384, "model_name": None},
            "nomic": {
                "name": "Nomic Embed Text",
                "dimensions": 768,
                "model_name": "nomic-embed-text",
            },
        }

    def select_models(self):
        """Let user select model through Streamlit UI"""
        st.sidebar.title("Model Selection")

                # Select LLM
        llm = st.sidebar.radio(
            "Choose LLM Model:",
            options=list(self.llm_models.keys()),
            format_func=lambda x: self.llm_models[x],
        )

        # Select Embeddings
        embedding = st.sidebar.radio(
            "Choose Embedding Model:",
            options=list(self.embedding_models.keys()),
            format_func=lambda x: self.embedding_models[x]["name"],
        )

        return llm, embedding
    
class SimplePDFProcessor:
    """Handle PDF processing and chunking"""

    def __init__(self, chunk_size = CHUNK_SIZE, chunk_overlap = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def read_pdf(self, pdf_file):
        """Read PDF and extract text"""
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    


        
