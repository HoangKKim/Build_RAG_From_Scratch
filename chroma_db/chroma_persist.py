'''
Bình thường, data sẽ được lưu tạm thời trên RAM -> khi app đóng/ restart máy ... -> mất dưx liệu
-> Solution: Persist data

**Persist data**: Dữ liệu được ghi xuống storage bền vững: HDD, SSD, Database hoặc cloud storage ...
'''

import chromadb
from chromadb.utils import embedding_functions

default_ef = embedding_functions.DefaultEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path= './chroma_db/db/chroma_persist')

collection = chroma_client.get_or_create_collection(name = "my_story", embedding_function = default_ef)

documents = [
    {"id": "doc1", "text": "Hello, world!"},
    {"id": "doc2", "text": "How are you today?"},
    {"id": "doc3", "text": "Goodbye, see you later!"},
    {"id": "doc4", "text": "Microsoft is a technology company that develops software."}
]

# define a query text
query_text = "Age of the Earth"

# add documents into collection
for doc in documents: 
    collection.upsert(ids = doc['id'], documents = doc['text'])

results = collection.query(
    query_texts = [query_text],
    n_results = 2
)

for idx, document in enumerate(results['documents'][0]):
    doc_id = results['ids'][0][idx]
    distance = results['distances'][0][idx]
    print(f"For the query: {query_text}, \nFound similar document {document} with ID {doc_id} - Distance: {distance}")
    print("-----------------------------")

