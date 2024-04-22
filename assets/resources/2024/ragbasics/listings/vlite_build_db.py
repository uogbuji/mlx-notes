import os
from pathlib import Path

from vlite import VLite
from vlite.utils import process_txt

# Needed to silence a Huggign Face tokenizers library warning
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
            print('Processing:', fname)
            vdb.add(process_txt(fname))
        else:
            print('Skipping:', fname)
    return vdb

vdb = setup_db(CONTENT_FOLDER, COLLECTION_FPATH)
vdb.save()  # Make sure the DB is up to date