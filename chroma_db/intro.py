import chromadb

# define text documents
documents = [
    {"id": "doc1", "text": "Hello, world!"},
    {"id": "doc2", "text": "How are you today?"},
    {"id": "doc3", "text": "Goodbye, see you later!"},
]

# define a query text
query_text = "Are you good today"

# --------------------------------------------------------------------------

# define chroma database
chroma_client = chromadb.Client()

# create collection
collection_name = 'test_collection'

collection = chroma_client.get_or_create_collection(name = collection_name)

# --------------------------------------------------------------------------
# add documents into collection (# switch `add` to `upsert` to avoid adding the same documents every time)
for doc in documents:
    collection.upsert(ids = doc['id'], 
                      documents = doc['text'])

# query the collection -> return `n` most similar results
results = collection.query(query_texts = [query_text], 
                           n_results = 3)

"""
print(results)

the result is printed in the terminal: 

>>> {   'ids': [['doc1', 'doc3', 'doc2']], 
        'embeddings': None, 
        'documents': [['Hello, world!', 'Goodbye, see you later!', 'How are you today?']], 
        'uris': None, 
        'included': ['metadatas', 'documents', 'distances'], 
        'data': None, 'metadatas': [[None, None, None]], 
        'distances': [[0.0, 1.1922799348831177, 1.294400930404663]] }
"""

# loop through the result
for idx, document in enumerate(results['documents'][0]):
    doc_id = results['ids'][0][idx]
    distance = results['distances'][0][idx]
    print(f"For the query: {query_text}, \nFound similar document {document} with ID {doc_id} - Distance: {distance}")
    print("-----------------------------")



