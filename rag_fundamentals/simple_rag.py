import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import pandas as pd
import ollama
from openai import OpenAI
import os

load_dotenv()

class EmbeddingModel:
    def __init__(self, model_type):
        self.model_type = model_type
        if model_type == 'openai':
            self.client = OpenAI(os.getenv('OPENAI_API_KEY'))
            self.embedding_fn = embedding_functions.OpenAIEmbedding(
                os.getenv('OPENAI_API_KEY'),
                model_name = 'text-embedding-3-small'
            )
        elif model_type == "chroma":
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        elif self.model_type == 'nomic':
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key = 'ollama',
                api_base="http://localhost:11434/v1",
                model_name="nomic-embed-text",
            )
        elif self.model_type == 'ollama':
            self.embedding_fn = embedding_functions.OllamaEmbeddingFunction(
                model_name = 'mxbai-embed-large'        # can view on Ollama to change
            )

class LLMModel:
    def __init__(self, model_type = 'openai'):
        if model_type == 'openai':
            self.client = OpenAI(
                api_key = os.getenv('OPENAI_API_KEY'),
                base_url = os.getenv('OPENAI_API_BASE_URL')
            )
            self.model = 'gpt-4o'
        else:
            self.client = OpenAI(
                api_key = 'ollama',
                base_url = 'http://localhost:11434/v1'
            )
            self.modll = 'llama3.2:latest'
    
    def generate_completion(self, messages):
        try:
            response = self.client.chat.completions.create(
                model = self.model,
                messages = messages,
                temperature = 0.0,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"     

def select_models():
    # Select LLM Model
    print("\nSelect LLM Model:")
    print("1. OpenAI GPT-4")
    print("2. Ollama Llama2")
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice in ["1", "2"]:
            llm_type = "openai" if choice == "1" else "ollama"
            break
        print("Please enter either 1 or 2")

    # Select Embedding Model
    print("\nSelect Embedding Model:")
    print("1. OpenAI Embeddings")
    print("2. Chroma Default")
    print("3. Nomic Embed Text (Ollama)")
    while True:
        choice = input("Enter choice (1, 2, or 3): ").strip()
        if choice in ["1", "2", "3"]:
            embedding_type = {"1": "openai", "2": "chroma", "3": "nomic"}[choice]
            break
        print("Please enter 1, 2, or 3")

    return llm_type, embedding_type

def generate_csv():
    facts = [
        {"id": 1, "fact": "The first human to orbit Earth was Yuri Gagarin in 1961."},
        {
            "id": 2,
            "fact": "The Apollo 11 mission landed the first humans on the Moon in 1969.",
        },
        {
            "id": 3,
            "fact": "The Hubble Space Telescope was launched in 1990 and has provided stunning images of the universe.",
        },
        {
            "id": 4,
            "fact": "Mars is the most explored planet in the solar system, with multiple rovers sent by NASA.",
        },
        {
            "id": 5,
            "fact": "The International Space Station (ISS) has been continuously occupied since November 2000.",
        },
        {
            "id": 6,
            "fact": "Voyager 1 is the farthest human-made object from Earth, launched in 1977.",
        },
        {
            "id": 7,
            "fact": "SpaceX, founded by Elon Musk, is the first private company to send humans to orbit.",
        },
        {
            "id": 8,
            "fact": "The James Webb Space Telescope, launched in 2021, is the successor to the Hubble Telescope.",
        },
        {   
            "id": 9, 
            "fact": "The Milky Way galaxy contains over 100 billion stars."},
        {
            "id": 10,
            "fact": "Black holes are regions of spacetime where gravity is so strong that nothing can escape.",
        },
    ]
        
    # write in csv
    df = pd.DataFrame(facts)
    facts_path = './rag_fundamentals/space_facts.csv'
    df.to_csv(facts_path, index = False)
    print(f"CSV file created successfully at {facts_path}")

def load_csv(csv_path= './rag_fundamentals/space_facts.csv'):
    df = pd.read_csv(csv_path)
    documents = df['fact'].tolist()
    print("\nLoaded documents: ")
    for doc in documents:
        print(f"- {doc}")
    return documents

def setup_chromadb(documents, embedding_model):
    client = chromadb.Client()

    try:
        # in case already have collection -> delete
        client.delete_collection("space_facts")
    except:
        pass

    # create a new collection
    collection = client.create_collection(
        name = 'space_facts',
        embedding_function = embedding_model.embedding_fn
    )

    # add documents
    for idx, doc in enumerate(documents):
        collection.add(documents = [doc], ids = [str(idx)])
    print("\nDocuments added to ChromaDB collection successfully!")
    return collection

def find_related_chunks(query, collection, top_k = 2):
    results = collection.query(
        query_texts = [query],
        n_results = top_k
    )

    print(f"Related chunks found:")
    for doc in results['documents'][0]:
        print(f"- {doc}")

    return list(
        zip(
            results['documents'][0], 
            (   # upgrade the priority for if - else
                results['metadatas'][0] 
                if results['metadatas'][0] 
                else [{}] * len(results['metadatas'][0])
            )
        )
    )

def augment_prompt(query, related_chunks):
    context = '\n'.join([chunk[0] for chunk in related_chunks])
    augmented_prompt = f"Context: \n{context} \n\nQuestion {query}"

    print(f"\nAugmented prompt: \n{augmented_prompt}")

    return augmented_prompt

def rag_pipeline(query, collection, llm_model: LLMModel, top_k = 2):
    print(f"Processing query: {query}")

    # find related chunks
    related_chunks = find_related_chunks(query, collection, top_k)
    # make augmented prompt
    augmented_prompt = augment_prompt(query, related_chunks)

    # get response
    response = llm_model.generate_completion(
        # messsages
        [
            {
                "role": "system",
                "content": "You are a helpful assistant who can answer questions about space but only answers questions that are directly related to the sources/ documents given"
            },
            {
                "role": "user",
                "content": augmented_prompt
            }
        ]
    )
    print(f"Generated response: {response}")
    references = [chunk[0] for chunk in related_chunks]
    return response, references

def main():
    print("Starting the RAG pipeline demo... ")

    # select models
    llm_type, embedding_type = select_models()

    # initialize models
    llm_model = LLMModel(llm_type)
    embedding_model = EmbeddingModel(embedding_type)

    print(f"\nUsing LLM: {llm_type.upper()}")
    print(f"Using Embeddings: {embedding_type.upper()}")

    # generate & load data
    generate_csv()
    documents = load_csv()

    # setup chroma db
    collection = setup_chromadb(documents, embedding_model)

    # run queries
    queries = [
        "What is the Hubble Space Telescope?",
        "Tell me about Mars exploration.",
    ]

    for query in queries:
        print("\n" + "=" * 50)
        print(f"Processing query: {query}")
        response, references = rag_pipeline(query, collection, llm_model)

        print("-" * 30)
        print(f"\nFinal Results:")
        print("- Response:", response)
        print("\n- References used:")
        for ref in references:
            print(f"  -- {ref}")
        print("=" * 50)

if __name__ == '__main__':
    main()



