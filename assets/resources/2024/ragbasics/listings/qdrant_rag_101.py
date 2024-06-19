# qdrant_rag_101.py
import os
from pathlib import Path
import pprint

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from ogbujipt.embedding.qdrant import collection

from mlx_lm import load, generate

chat_model, tokenizer = load('mlx-community/Hermes-2-Theta-Llama-3-8B-4bit')

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
# Needed to silence a Hugging Face tokenizers library warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

TEXT_SUFFIXES = ['.md', '.txt']
CONTENT_FOLDER = Path('assets/resources/2024/ragbasics/files')
DBPATH = '/tmp/qdrant_test'  # Set up disk storage location

assert Path(DBPATH).exists(), 'DB not found. You may need to run qdrant_build_db.py again'

QCLIENT = QdrantClient(path=DBPATH)

USER_PROMPT = 'How can I get a better understand what tokens are, and how they work in LLMs?'
SCORE_THRESHOLD = 0.2
MAX_CHUNKS = 4

# Set up to retrieve from previosly created content database named "ragbasics"
# Note: Here you have to match the embedding model with the one originally used in storage
qcoll = collection('ragbasics', embedding_model, db=QCLIENT)

results = qcoll.search(USER_PROMPT, limit=MAX_CHUNKS, score_threshold=SCORE_THRESHOLD)

top_match_text = results[0].payload['_text']  # Grabs the actual content
top_match_source = results[0].payload['src-file']  # Grabs the metadata stored alongside
print(f'Top matched chunk: {top_match_text}\n\nFrom file {top_match_source}')

gathered_chunks = '\n\n'.join(
    doc.payload['_text'] for doc in results if doc.payload)

sys_prompt = '''\
You are a helpful assistant who answers questions directly and as briefly as possible.
Consider the following context and answer the user\'s question.
If you cannot answer with the given context, just say you don't know.\n
'''

# Construct the input message struct from the system prompt, the gathered chunks, and the user prompt itself
messages = [
  {'role': 'system', 'content': sys_prompt},
  {'role': 'user', 'content': f'=== BEGIN CONTEXT\n\n{gathered_chunks}\n\n=== END CONTEXT'},
  {'role': 'user', 'content': f'Please use the context above to respond to the following:\n{USER_PROMPT}'}
  ]

pprint.pprint(messages, width=120)

chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
response = generate(chat_model, tokenizer, prompt=chat_prompt, verbose=True)

print('RAG-aided LLM response to the user prompt:\n', response)
