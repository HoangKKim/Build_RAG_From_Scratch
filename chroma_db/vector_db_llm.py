import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import ollama
from openai import OpenAI

load_dotenv()
# openai_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'), 
                base_url = os.getenv('OPENAI_API_BASE_URL'))
embed_model = 'mxbai-embed-large'

db_path = './chroma_db/db/chroma_persist_storage'

def init_database(db_path):

    embed_function = embedding_functions.OllamaEmbeddingFunction(
        model_name = embed_model
    )

    chroma_client = chromadb.PersistentClient(path = db_path)
    collection_name = 'document_qa_collection'
    collection = chroma_client.get_or_create_collection(
        name = collection_name,
        embedding_function = embed_function
    )
    return collection

def load_documents_from_directory(dir_path):
    ''' load documents (articles from directory)'''
    print("=== Loading documents from directory ===")
    documents = []
    for filename in os.listdir(dir_path):
        if filename.endswith('.txt'):
            with open(file = os.path.join(dir_path, filename), mode = 'r', encoding='utf-8') as f:
                doc = f.read()
                documents.append({
                    "id": filename,
                    "text": doc
                })
    return documents

def split_text(text: str, chunk_size: int = 1000, chunk_ovelap: int = 20):
    '''split text into chunks'''
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_ovelap
    return chunks

def get_ollama_embedding(text, embed_model = 'mxbai-embed-large'):
    ''' generate embedding using Ollama API'''
    response = ollama.embeddings(
        model = embed_model,
        prompt = text
    )
    embedding = response['embedding']
    print("=== Generating embeddings ... ===")
    return embedding

def query_document(collection, question, n_results = 2):
    results = collection.query(
        query_texts = question,
        n_results = n_results
    )

    relevant_chunks = []
    for sublist in results['documents']:
        for doc in sublist:
            relevant_chunks.append(doc)
    print('=== Returning relevants chunks ===')
    return relevant_chunks

def generate_response(question, relevant_chunks, client):
    # client = OpenAI(api_key=openai_key)

    context = "\n\n".join(relevant_chunks)

    prompt = f"""You are an assistant for question-answering tasks. 
    Based on the retrived context to answer the question. 
    <Retrived-Context>
    {context}
    </Retrived-Context>
    <Question>
    {question}
    </Question>

    ### Task: 
    - Just only use the retrived context to answer the question.
    - If you don't know the answer, say that you don't know. DON'T give the hallucinated answer.
    - Use three sentences maximum and keep the answer concise.
    """

    response = client.chat.completions.create(
        model = 'gpt-4o',
        messages = [
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    answer = response.choices[0].message
    return answer


if __name__ == '__main__':
    dir_path = "./articles"

    # load documents from directory
    documents = load_documents_from_directory(dir_path)
    print(f"Loaded {len(documents)} documents")

    # initialize collection for database
    collection = init_database(db_path)

    # # split documents into chunks
    # chunked_documents = []
    # for doc in documents:
    #     chunks = split_text(doc['text'])
    #     print('=== Splitting docs into chunks ===')
    #     for i, chunk in enumerate(chunks):
    #         chunked_documents.append({
    #             "id": f"{doc['id']}_chunk{i+1}",
    #             "text": chunk
    #         })

    # # generate embeddings for the document chunks
    # for chunk in chunked_documents:
    #     print("=== Generating embeddings ... ===")
    #     chunk['embedding'] = get_ollama_embedding(chunk['text'])

    # # upsert documents with embedding into Chroma
    # for doc in chunked_documents:
    #     print("=== Inserting chunks into database ===")
    #     collection.upsert(
    #         ids = [doc['id']],
    #         documents = [doc['text']],
    #         embeddings = [doc['embedding']]
    #     )

    # question = 'tell me about AI replacing TV writers strike'
    # question = 'tell me about databricks acquisition of ai'
    question = "give me a brief overview of the articles. Be concise, please."

    relevant_chunks = query_document(collection, question)
    # print('=== Relevant chunks ===')
    # print(relevant_chunks)

    answer = generate_response(question, relevant_chunks, client)
    print("=== Answer ===")
    print(answer.content)





