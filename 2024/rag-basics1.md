![Google Gemini-generated cover image: Loading a truck with Appleicious content](../assets/images/2024/RAG-basics-1-cover.jpg)

# Retrieval augmentation with MLX: A bag full of RAG, part 1

_Author: [Uche Ogbuji](https://ucheog.carrd.co/)_

After the intial fun with LLMs, asking the capital of random countries and getting them to tell lousy jokes or write wack poems, the next area I've found people trying to figure out is how to "chat their documents". In other words, can you give an LLM access to some documents, databases, web pages, etc. and get them to use that context for more specialized discussion and applications? This is more formally called Retrieval Augmented Generation (RAG).

As usual I shan't spend too much time explaining fundamental AI principles in these articles which are focused on the MLX framework. For a very high level view of RAG, see [this synopsis from the Prompt Engineering Guide](https://www.promptingguide.ai/techniques/rag), or better yet, see [this full article from the same source](https://www.promptingguide.ai/research/rag). The latter is a long read, but really important if you're trying to advance from baby steps to almost any sophisticated use of LLMs. In any case, you'll need to understand at least the basics of RAG to get the most of this article.

### RAG application workflow (Source: [Gao et al.](https://arxiv.org/abs/2312.10997))

![RAG application workflow (Source: Gao et al. https://arxiv.org/abs/2312.10997)](../assets/images/2024/rag-process-gao-et-al.png)

In this article I'll show through code examples how you can start to build RAG apps to work with LLM generation on MLX. It's a big subtopic, even just to get through the basics, so I'll break it into two parts, the first of which focuses on the retrieval portion.

# Trying out a vector DBMS

So far in these articles the main benefit of MLX has been GenAI accelerated on Apple Silicon's Metal architecture. That's all about the "G" in RAG. It would be great to have the "R" part also taking some advantage of Metal, but that proves a bit tougher than I'd expected. Many of the best-known vector DBs (faiss, qdrant, etc.) use various techniques to accelerate embedding and perhaps lookup via GPU, but they focus on Nvidia (CUDA) and in some cases AMD (ROCm), with nothing for Metal. We need content to add to the database. I've made it easy by providing the markdown of articles in thei MLX notes series (including this article). You can [download them from Github](https://github.com/uogbuji/mlx-notes/tree/main/assets/resources/2024/ragbasics/files), the whole directory, or just the contained files, and put them in a location you can refer to in the code later.

PGVector is my usual go-to for vector DBMS, but it adds a few nuances and dependencies I wanted to avoid for this write-up. We'll just use [Qdrant](https://qdrant.tech/).

# Using the Qdrant vector DBMS via OgbujiPT

My OgbujiPT library includes tools to make it easy to use Qdrant or PostgreSQL/PGVector for vector database applicaitons such as RAG. Install the needed prerequisites.

```sh
pip install ogbujipt qdrant_client sentence_transformers
```

Listing 4 vectorizes the same markdown documents as before, and then does a sample retrieval, using Qdrant. With the `text_split` function, available in [OgbujiPT](https://github.com/OoriData/OgbujiPT), I split by Markdown paragraphs (`\n\n`), with a guideline that chunks should be kept under 100 characters where possible.

### Listing 4 (qdrant_build_db.py): Switch to Qdrant for content database from markdown files on disk

```py
# qdrant_build_db.py
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer  # ST docs: https://www.sbert.net/docs/
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
```

Output:

```
Matched chunk: `response` is the plain old string with the LLM completion/response. It will already have been streamed to the console thanks to `verbose=True`, right after the converted prompt, displayed so you can see how the ChatML format has been converted using special, low-level LLM tokens such as `<|im_start|>` & `<|im_end|>`. Having the system message in the chat prompting and all that definitely, by my quick impressions, made the interactions far more coherent.

From file assets/resources/2024/ragbasics/files/MLX-day-one.md
```

You might not be able to tell off the bat, but the embedding and indexing of text in the examples above is much slower that need be. let's look into an option for speeding it up.

# Using MLX-Embeddings for local, accelerated embedding generation

mlx-embeddings is a package designed to generate text and image embeddings locally on Apple Silicon using the MLX framework. It uses Apple's Metal acceleration for much faster computation than cross-platform libraries like sentence-transformers.

Install the package with:

```sh
pip install mlx-embeddings
```

Here's how you can use mlx-embeddings to generate embeddings for a list of texts:

```py
from mlx_embeddings import load, generate

# Load a model (e.g., MiniLM in MLX format)
model, processor = load("mlx-community/all-MiniLM-L6-v2-4bit")

# Generate normalized embeddings for a list of texts
output = generate(model, processor, texts=["I like grapes", "I like fruits"])
embeddings = output.text_embeds  # Normalized embeddings

# Example: Compute similarity matrix using MLX
import mlx.core as mx
similarity_matrix = mx.matmul(embeddings, embeddings.T)
print("Similarity matrix between texts:")
print(similarity_matrix)
```

This workflow is similar to sentence-transformers, but all computation runs natively on your Mac, taking full advantage of Apple Silicon hardware acceleration.

mlx-embeddings supports a growing set of popular models, including BERT and XLM-RoBERTa, with more being added. Vision models are also supported, making it suitable for multimodal RAG applications

You can use mlx-embeddings in place of sentence-transformers wherever you need to generate vector representations for indexing or retrieval. The embeddings can be stored in any vector database, including those mentioned in previous sections. For many usage scenarios MLX-based embedding and inference are significantly faster than running PyTorch-based models via CPU, and often faster than using non-native GPU backends on Mac.

# A word on tokens

Tokens have come up before in this series, and you might be wondering. "What are those, exactly?" Tokens are a really important concept with LLMs. When an LLM is dealing with language, it doesn't do so character by character, but it breaks down a given language into statistically useful groupings of characters, which are then identified with integer numbers. For example the characters "ing" occur pretty frequently, so a tokenizer might group those as a single token in many circumstances. It's sensitive to the surrounding character sequence, though, so the word "sing" might well be encoded as a single token of its own, regardless of containing "ing".

The best way to get a feel of LLM tokenization is to play around with sample text and see how it gets converted. Luckily there are many tools out there to help, including [the simple llama-tokenizer-js playground](https://belladoreai.github.io/llama-tokenizer-js/example-demo/build/) web app which allows you to enter text and see how the popular Llama LLMs would tokenize them.

![Quick exampls with the online Llama tokenizer](../assets/images/2024/tokenizer-examples.png)

The colors don't mean anything special in themselves. They're just visual tiling to separate the tokens. Notice how start of text is a special token `<s>`. You might remember we also encountered some other special tokens such as `<|im_start|>` (begin conversation turn) in previous articles. LLM pre-training and fine-tuning changes the way things are tokenized, as part of setting the entire model of language. Llama won't tokenize exactly as, say ChatGPT does, but the basic concepts stay the same.

The picture shows an example of how markup such as HTML can affect tokenization. There are models such as the commercial [Docugami](https://www.docugami.com/) which are trained towards efficient tokenization of markup. Code-specialized LLMs such as those used in programmer copilot tools would have efficient tokenizations of the sorts of constructs which are more common in programming code than in natural language.

## Creating more sensible chunks

As I mentioned, I used a simple text splitter from OgbujiPT above, but lately I've taken to use [Chonkie](https://github.com/chonkie-inc/chonkie), a library that offers a wide variety of flexible chunking options, chunking by tokens, and by LLM-guided heuristics.

In effect, the tokenization establishes the shape of language in a model, which is why using token boundaries in chunking text can help avoid weid boundary issues in vector lookups. There ae many other chunking tchniques you can try, as well. Just to cite one example, you can chunk each paragraph separately, say in a collection of articles. That way you have a coherent thread of meaning in each chunk which is more likely to align with expected search patterns.

# Generation next to come

Now that we can search a database of content in order to select items which seem relevant to an input, we're ready to turn our attention to the generation component of RAG, in part 2. Keep in mind always that RAG is trickier than one may think, and there are many considerations to designing an effective RAG pipeline.

# Recent Developments (mid 2024 - mid 2025)

As always, the MLX ecosystem is evolving rapidly, so readers should check for new models and tools regularly, here are some interesting tidbits dating from after I first wrote this article.

* **Model Format Conversions**: The MLX ecosystem now includes many converted models (e.g., MiniLM, BGE) in MLX format, available on Hugging Face, which can be loaded directly using mlx-embeddings
* **Alternative Embedding Packages**: Other projects like mlx_embedding_models and swift-embeddings enable running BERT- or RoBERTa-based embeddings natively on Mac, broadening the choices for local RAG workflows
* **Multimodal RAG**: With MLX now supporting vision models in addition to language, it is possible to build multimodal RAG systems (text + image retrieval) entirely on-device
* **Community Tools**: There is a growing ecosystem of RAG implementations optimized for MLX and Apple Silicon, including command-line tools and open-source projects for vector database integration

# Cultural accompaniment

While doing the final edits of this article I was enjoying the amazing groove of the Funmilayo Afrobeat Orquestra, plus Seun Kuti's Egypt 80, a song called Upside Down. I thought to myself (this groove is part of the article whether anyone knows it, so why not share?) Since I'm also a poet, DJ, etc. I think I'll start sharing with these articles some artistic snippet that accompanied me in the process, or maybe something that's been inspiring me lately.

[![Funmilayo Afrobeat Orquestra & Seun Kuti's Egypt 80 - Upside Down [Live Session]](https://img.youtube.com/vi/Gf8G3OhHW8I/0.jpg)](https://www.youtube.com/watch?v=Gf8G3OhHW8I)

<!--
<iframe width="560" height="315" src="https://www.youtube.com/embed/Gf8G3OhHW8I?si=ywIzDLuBnOeHMmTL" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
-->

I grew up on Afrobeat (not "Afrobeats, abeg, oh!"), back home in Nigeria, and I'm beyond delight to see how this magical, elemental music has found its way around the world and continues to flourish. Lovely to see Seun, my favorite contemporary exponent of his father, Fela's genre, OGs such as Kunle Justice on electric bass (a man who should be much better known!) and of course the dynamic, Brazil-based women of Funmilayo, named after Fela's mother. This one betta now! Make you enjoy!
