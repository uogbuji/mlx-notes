# vlite_custom_split_build_db.py
import os
from pathlib import Path

from vlite import VLite

from ogbujipt.text_helper import text_split

# Needed to silence a Hugging Face tokenizers library warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

TEXT_SUFFIXES = ['.md', '.txt']
# 
CONTENT_FOLDER = Path('assets/resources/2024/ragbasics/files')
# Path to a "CTX" file which is basically the vector DB in binary form
COLLECTION_FPATH = Path('/tmp/ragbasics')

def setup_db(files, collection):
    # Create database
    # If you don't specify a "collection" (basically a filename), vlite will create
    # a file, using the current timestamp. under "contexts" in the current dir
    # device='mps' uses Apple Metal acceleration for the embedding,
    # which is typically the most expensive stage
    vdb = VLite(collection=collection, device='mps')

    for fname in files.iterdir():
        if fname.suffix in TEXT_SUFFIXES:
            fname = str(fname)
            print('Processing:', fname)
            with open(fname) as fp:
                # Governed by paragraph boundaries (\n\n), with a target chunk size of 100
                for chunk in text_split(fp.read(), chunk_size=100, separator='\n\n'):
                    print(chunk, '\n¶')
                    vdb.add(chunk, metadata={'src-file': fname})
        else:
            print('Skipping:', fname)
    return vdb

vdb = setup_db(CONTENT_FOLDER, COLLECTION_FPATH)
vdb.save()  # Make sure the DB is up to date
