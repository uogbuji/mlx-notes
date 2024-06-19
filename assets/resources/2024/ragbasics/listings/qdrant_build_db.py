# qdrant_build_db.py
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from ogbujipt.text_helper import text_split
from ogbujipt.embedding.qdrant import collection

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
# Needed to silence a Hugging Face tokenizers library warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

TEXT_SUFFIXES = ['.md', '.txt']
CONTENT_FOLDER = Path('assets/resources/2024/ragbasics/files')
DBPATH = '/tmp/qdrant_test'  # Set up disk storage location
QCLIENT = QdrantClient(path=DBPATH)


def setup_db(files):
    # Create content database named "ragbasics", using the disk storage location set up above
    qcoll = collection('ragbasics', embedding_model, db=QCLIENT)

    for fname in files.iterdir():
        if fname.suffix in TEXT_SUFFIXES:
            fname = str(fname)
            print('Processing:', fname)
            with open(fname) as fp:
                # Governed by paragraph boundaries (\n\n), with a target chunk size of 100
                for chunk in text_split(fp.read(), chunk_size=100, separator='\n\n'):
                    # print(chunk, '\n¶')
                    # Probably more efficient to add in batches of chunks, but not bothering right now
                    # Metadata can be useful in many ways, including having the LLM cite sources in its response
                    qcoll.update(texts=[chunk], metas=[{'src-file': fname}])
        else:
            print('Skipping:', fname)
    return qcoll

vdb = setup_db(CONTENT_FOLDER)
results = vdb.search('How ChatML gets converted for use with the LLM', limit=1)

top_match_text = results[0].payload['_text']  # Grabs the actual content
top_match_source = results[0].payload['src-file']  # Grabs the metadata stored alongside
print(f'Matched chunk: {top_match_text}\n\nFrom file {top_match_source}')
