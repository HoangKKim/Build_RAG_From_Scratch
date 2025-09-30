# UI
import streamlit as st

# database
import chromadb
from chromadb.utils import embedding_functions

# LLM
from openai import OpenAI

# system
from dotenv import load_dotenv
import os
import PyPDF2       # dealing, managing, extracting,... with pdf file
import uuid

class FundamentalRAGSystem:
    def __init__(self, llm_model, embedding_model = None):
        self.llm_model = llm_model
        self.embedding_model = embedding_model

        # init llm model
        self.client = OpenAI(
            api_key = 'ollama',
            base_url = 'http://localhost:11434/v1'
        )

        # inti embedding function
        if self.embedding_model is None:    # using ChromaDB embedding function
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        else:
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key = 'ollama',
                aip_base = 'http://localhost:11434/v1',
                model_name = 'mxbai-embed-large'
            )
        
        # setup collection
        self.chroma_client = chromadb.PersistentClient(path = './rag_fundamentals/database')
        self.collection = self._setup_collection()

    def _setup_collection(self):
        collection = self.chroma_client.get_or_create_collection(
            name = 'funda_db',
            embedding_functions = self.embedding_fn
        )
        return collection
    
    def add_documents(self, chunks):
        "Add chunks of documnents to Database"

        for chunk in chunks:
            self.collection.add(
                ids = [chunk['id']],
                documents = [chunk['text']],
                metadatas = [chunk['metadata']]
            )
        
    def query_documents(self, query, n_results = 3):
        """Query documents and return relevant chunks"""
        results = self.collection.query(
            query_texts = [query],
            n_results = n_results
        )
        return results
    
    def generate_response(self, query, results):
        prompt = f"""
            Based on the following context, please answer the questions.
            If you cannot find the answer in the context, say so, or "I don't know"

            <Context>{results}</Context>
            <Query>{query}</Query>"""
        
        response = self.llm.chat.completions.create(
            model = self.llm_model,
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )
        return response.choices[0].message.content
    
class PDFProcessor:
    def __init__(self, chunk_size, chunk_overlap):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def read_pdf(self, pdf_file):
        """read pdf and extract text"""
        reader = PyPDF2.PdfReader(pdf_file)
        text = ''

        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def create_chunks(self, text, pdf_file):
        """split text into chunks"""

        chunks = []
        start = 0

        while start < len(text):


    
    