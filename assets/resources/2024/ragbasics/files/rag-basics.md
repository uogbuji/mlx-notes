![Google Gemini-generated cover image: ???](../assets/images/2024/xyz.png)

# Retrieval augmentation with MLX: A bag full of RAG

_Author: [Uche Ogbuji](https://ucheog.carrd.co/)_

After the intiial fun with LLMs asking the capital of random countries and getting them to tell lousy jokes or write wack poems, the next area I've found people trying to figure out is how to "chat their documents". In other words, can you give an LLM access to some documents, databases, web pages, etc. and get them to use that context for more specialized discussion and applications? This is more formally called Retrieval Augmented Generation (RAG).

As usual I shan't spend too much time explaining fundamental AI principles in these articles which are focused on the MLX framework. For a very high level view of RAG, see [this article at the Prompt Engineering Guide](https://www.promptingguide.ai/research/rag). It's a long read, but really important if you're trying to advance from babay steps to almost any sophisticated use of LLMs. In any case, you'll need to understand at least the basics of RAG to get the most of this article.

In this article I'll show through code examples how you can start to build RAG apps to work with LLM generation on MLX.

# Quick start

So far in these articles the main benefit of MLX has been GenAI accelerated on Apple Silicon's Metal architecture. That takes care of the "G" in RAG. It would be great to have the "RA" part also taking some advantage of Metal, but that proves a bit tougher than I'd expected. Many of the best-known vector DBs (faiss, qdrant, etc.) use various techniques to accelerate embedding and perhaps lookup via GPU, but they focus on Nvidia (CUDA) and in some cases AMD (ROCm), with nothing for Metal.

Following a hint from [Prince Canuma](https://huggingface.co/prince-canuma) I did find [vlite—"a simple and blazing fast vector database"](https://github.com/sdan/vlite) whose docs include the promising snippet:

> - `device` (optional): The device to use for embedding ('cpu', 'mps', or 'cuda'). Default is 'cpu'. 'mps' uses PyTorch's Metal Performance Shaders on M1 macs, 'cuda' uses a NVIDIA GPU for embedding generation.

```sh
pip install "vlite[ocr]"
```

This installs the vlite package as well as helper packages for including text from PDF documents in the database. In my case I ran into pip errors such as the following.

![pip errors trying to install vlite](../assets/images/2024/vlite-install-errors.png)

These sorts of conflicting dependencies are a common problem in the AI space, especially with widely-used and fast-evolving packages such as transformers, tokenizers, pytorch, pydantics and such.

If you don't need all the added PDF tools you can install just vlite by taking out the `[ocr]` modifier from the `pip` command.

```py
from vlite import VLite
from vlite.utils import process_txt

# Create database
vdb = VLite()
# vdb.add("hello world", metadata={"artist": "adele"})

vdb.add(process_txt(, use_ocr=True))

results = vdb.retrieve("how do transformers work?")
print(results)
```


# Going beyond

Now that you have a basic idea of how to use RAG in MLX, you're only limited by your imagination. You can load up your retrieval DB with your your company's knowledgebase to create a customer self-help bot. You can load it up with your financials to create an audit prep tool. You can load up with all your texts to you can remember whom to thank about that killer restaurant recommendation. Since you're using a locally-hosted LLM, you can run such apps entirely airgapped and have none of the privacy concerns as you would e.g. with OpenAI.

## Remember: The data is the magic

RAG techniques mean you are selecting information to include in the prompt you send to the LLM for completion/generation. Obviously the value of the information you put into this prompt correlates directly to the value of the responses you get from the LLM. Here's a dirty little secret about RAG projects: they tend to look extremely promising in prototype phase, and then run into massive engineering difficulties on the path towards full product status.

One of the biggest reasons for this disconnect is data quality.

