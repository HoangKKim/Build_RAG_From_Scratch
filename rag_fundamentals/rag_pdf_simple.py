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
            "chroma": {
                "name": "Chroma Default", 
                "dimensions": 384, 
                "model_name": None},
            "ollama": {
                "name": "Mxbai Embed Text",
                "dimensions": 768,
                "model_name": "mxbai-embed-large",
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
        
        # combine all text in page into a variable "text"
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def create_chunks(self, text, pdf_file):
        """split text into chunks
        
        args:
        - text: all extracted text from pdf file
        - pdf_file: to add into metadata of chunk
        """

        chunks = []
        start = 0

        while start < len(text):
            # find end of chunk
            end = start + self.chunk_size

            # if not at the start ( # 0): include overlap
            if start != 0:
                start = start - self.chunk_overlap

            # get chunk
            chunk = text[start:end]

            # try to break at sentences 
            # tránh việc cắt chunk ở các câu lở dở -> cắt trọn câu
            if end < len(text):         # is not the last chunk
                last_period = chunk.rfind('.')      # do not have any period -> -1
                if last_period != -1:   
                    chunk = chunk[:last_period+1]   # get text at the last period
                    end = start + last_period + 1   # modify end position
            
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk,
                "metadata": {"source" : pdf_file.name}
            })

            start = end
        return chunks
    
class SimpleRAGSystem:
    def __init__(self, embedding_model= 'ollama', llm_model= 'openai'):
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        # initialize chromadb
        self.database = chromadb.PersistentClient(path = './chroma_db')

        # setup embedding function
        self.setup_embedding_function()     # assign for self.embedding_fn

        # setup llm 
        if llm_model == 'openai':
            self.llm = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
        else:
            self.llm = OpenAI(
                api_key = 'ollama',
                base_url = 'http://localhost:11434/v1',
                # model_name = 'llama3.2:latest'
            )

        # get or create collection
        self.collection = self.setup_collection()


    def setup_embedding_function(self):
        """setup the appropriate embedding function"""
        try:
            if self.embedding_model == 'openai':
                self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                    api_key = os.getenv("OPENAI_API_KEY"),
                    model_name = 'text-embedding-3-small'
                )
            elif self.embedding_model == 'ollama':
                self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                    api_key = "ollama",
                    api_base ="http://localhost:11434/v1",
                    model_name = 'mxbai-embed-large'
                )
            else:       # chroma default fn
                self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        except Exception as e:
            st.error(f"Error in setting up embedding function: {str(e)}")
            raise e
        
    def setup_collection(self):
        """Setup collection with proprer dimension handling"""

        collection_name = f"documents_{self.embedding_model}"

        try:
            # try to get existing collection first
            try:
                collection = self.database.get_collection(
                    name = collection_name,
                    embedding_function = self.embedding_fn
                )
                st.info(f"Using existing collecion for {self.embedding_model} embedding")
            except:
                # in case does not exist any appropriate collection:
                collection = self.database.create_collection(
                    name = collection_name,
                    embedding_function = self.embedding_fn,
                    metadata = {
                        "model": self.embedding_model
                    }
                )
                st.success(f"Create new collection for {self.embedding_model} embedding ")
            return collection
        except Exception as e:
            st.error(f"Error setting up collection: {str(e)}")
            raise e
        
    def add_documents(self, chunks):
        """Add documents to ChromaDB"""
        try:
            if not self.collection:
                self.collection = self.setup_collection()
            
            # add documents:
            for chunk in chunks:
                self.collection.add(
                    ids = [chunk['id']],
                    documents = [chunk['text']],
                    metadatas = [chunk['metadata']]
                )

            return True
        except Exception as e:
            st.error(f"Error adding documents: {str(e)}")
            return False
        
    def query_documents(self, query, n_results = 3):
        """Query documents and return relevant chunks"""
        try:
            # ensure collection exist
            if not self.collection:
                raise ValueError("No collection in database")
            
            results = self.collection.query(
                query_texts = [query],
                n_results = n_results
            )
            return results
        except Exception as e:
            st.error(f"Error querying documents: {str(e)}")
            return None
        
    def generate_response(self, query, context):
        """Generate response using LLM"""

        try:
            prompt = f"""
                Based on the following context, please answer the questions.
                If you cannot find the answer in the context, say so, or "I don't know"

                <Context>{context}</Context>
                <Query>{query}</Query>
            """

            response = self.llm.chat.completions.create(
                model = 'gpt-4o' if self.llm_model == 'openai' else "llama3.2:latest",
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return None
        
    def get_embedding_info(self):
        """Get information about current embedding model"""
        model_selector = SimpleModelSelector()
        model_info = model_selector.embedding_models[self.embedding_model]
        return {
            "name": model_info["name"],
            "dimensions": model_info["dimensions"],
            "model": self.embedding_model,
        }
    
def main():
    st.title("🤖 Simple RAG System")

    # initialize session state
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    else:
        st.session_state.processed_files.clear()

    if "current_embedding_model" not in st.session_state:
        st.session_state.current_embedding_model = None
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None

    # initialize model selector
    model_selector = SimpleModelSelector()
    llm_model, embedding_model = model_selector.select_models()

    # check if embedding model changed
    if embedding_model != st.session_state.current_embedding_model:
        # reset all things 
        st.session_state.processed_files.clear()
        st.session_state.current_embedding_model = embedding_model
        st.session_state.rag_system = None
        st.warning(f"Embedding model changed. Please re-upload your documents")

    # initialize RAG system
    try:
        if st.session_state.rag_system is None:
            st.session_state.rag_system = SimpleRAGSystem(embedding_model, llm_model)
        
        # display current embedding model info
        embedding_info = st.session_state.rag_system.get_embedding_info()
        st.sidebar.info(
            f"Current Embedding Model:\n"
            f"- Name: {embedding_info['name']}"
            f"- Dimension: {embedding_info['dimensions']}"
        )
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return
    
    # File upload
    pdf_file = st.file_uploader("Upload PDF", type="pdf")

    if pdf_file and pdf_file.name not in st.session_state.processed_files:
        # Process PDF
        processor = SimplePDFProcessor()
        with st.spinner("Processing PDF..."):
            try:
                # Extract text
                text = processor.read_pdf(pdf_file)
                # Create chunks
                chunks = processor.create_chunks(text, pdf_file)
                # Add to database
                if st.session_state.rag_system.add_documents(chunks):
                    st.session_state.processed_files.add(pdf_file.name)
                    st.success(f"Successfully processed {pdf_file.name}")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")

    # Query interface
    if st.session_state.processed_files:
        st.markdown("---")
        st.subheader("🔍 Query Your Documents")
        query = st.text_input("Ask a question:")

        if query:
            with st.spinner("Generating response..."):
                # Get relevant chunks
                results = st.session_state.rag_system.query_documents(query)
                if results and results["documents"]:
                    # Generate response
                    response = st.session_state.rag_system.generate_response(
                        query, results["documents"][0]
                    )

                    if response:
                        # Display results
                        st.markdown("### 📝 Answer:")
                        st.write(response)

                        with st.expander("View Source Passages"):
                            for idx, doc in enumerate(results["documents"][0], 1):
                                st.markdown(f"**Passage {idx}:**")
                                st.info(doc)
    else:
        st.info("👆 Please upload a PDF document to get started!")


if __name__ == "__main__":
    main()

        



        
