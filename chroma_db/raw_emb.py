# from openai import OpenAI
# import os
# from dotenv import load_dotenv

# load_dotenv()

# ''' With OpenAI API Key supports embedding model'''
# api_key = os.getenv('OPENAI_API_KEY')
# base_url = os.getenv('OPENAI_API_BASE_URL')

# client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'), 
#                 base_url = os.getenv('OPENAI_API_BASE_URL'))

# response = client.embeddings.create(
#     input = 'how are you',
#     model="text-embedding-3-small"
# )

# print(response)

'''In case, i dont have it, using ollama instead'''

import ollama
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

embed_model = "mxbai-embed-large"
# client = OpenAI(base_url = "http://localhost:11434/v1", api_key = 'ollama')

response = ollama.embed(
    input = 'Your text string goes here',
    model = embed_model
)

print(response)