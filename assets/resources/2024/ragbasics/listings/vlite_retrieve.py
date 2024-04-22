# vlite_retrieve.py
from pathlib import Path

from vlite import VLite

COLLECTION_FPATH = Path('/tmp/ragbasics')

vdb = VLite(collection=COLLECTION_FPATH, device='mps')

# top_k=N means take the N closest matches
# return_scores=True adds the closeness scores to the return
results = vdb.retrieve('ChatML format has been converted using special, low-level LLM tokens', top_k=1, return_scores=True)
print(results[0])
