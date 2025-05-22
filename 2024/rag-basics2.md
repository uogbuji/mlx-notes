![Google Gemini-generated cover image: Truck full of Appleicious content; ready to roll](../assets/images/2024/RAG-basics-2-cover.jpg)

# Retrieval augmentation with MLX: A bag full of RAG, part 2
14 June 2024. Versions: mlx: 0.15.0 | mlx-lm: 0.14.3
Updated: 11 May 2025. Versions: mlx: 0.22.0 | mlx-lm: 0.20.6
_Author: [Uche Ogbuji](https://ucheog.carrd.co/)_

[In the first part of this article](https://github.com/uogbuji/mlx-notes/blob/main/2024/rag-basics1.md) I made a basic introduction to Retrieval Augmented Generation (RAG), a technique for integrating content retrieved from databases or other sources into prompts for LLM. In the first part I showed how you might construct such a context database (retrieval), and in this part we'll see how the content can be stuffed into the prompt for the LLM in the generation phase. You'll want to read part 1 before proceeding.

# Back to the land of LLMs

While fiddling with the vector database we haven't got around yet to using the G (Generation) part of RAG. The results from vector DB lookup are exact raw chunks of content. What you usually want in such scenarios, is for the LLM to take this raw content and work it into a coherent response to the user. A next step is to stuff the retrieved text into the prompt, as context, along with some instructions (generally placed in a system prompt). If all goes well, the LLM's response proves useful, and is anchored by the facts retrieved from the vector DB, lowering the LLM's tendency to hallucinate.

_Aside: Hallucination is one of the most misunderstood topics in GenAI. It's always important to remember what LLMs are trained to do: they are trained to complete the text provided in the prompt. They are just predicting tokens and generating language. This means that they will sometimes generate language whose meaning is confusing, false or misleading, which we call hallucinations, but in doing so, they are merely following their training._

_A part of the solution is to include in the prompt facts and instructions which are carefully constructed (i.e. prompt engineered) according to an understanding of the LLM's statistical tendencies. This reduces the likelihood of hallucinations, but it may not be possible to completely eliminate that tendency. Some LLMs are trained or fine-tuned to be especially "obedient" to the context, and these are good choices for RAG. Picking the right LLM is another part of the solution; using multi-stage pipelines with verification by other LLMs or even people (perhaps from a random or heuristically selected sample of transcripts) is another part of the solution. RAG is a simple concept, but getting consistently great results with it involves complex considerations_

## Prompt stuffing 101

In the previous article, [Listing 4 (qdrant_build_db.py)](https://github.com/uogbuji/mlx-notes/tree/main/assets/resources/2024/ragbasics/listings) created a Qdrant vector database from the markdown of articles in this series. We can now use that database to retrieve likely chunks of content and stuff these in the prompt for the generation phase of RAG. Listing 1, below, is a simple example of this process, using the MLX generation interface explored in previous articles.

The code first queries the vector database for chunks of content semantically similar to the user question or prompt, which is hard-coded for simplicity. It then pulls the chunks into a template to construct an overall prompt, which is sent to the LLM for completion.

### Listing 1 (qdrant_rag_101.py)

_Note: [You can find all code listings on GitHub.](https://github.com/uogbuji/mlx-notes/tree/main/assets/resources/2024/ragbasics/listings)_

```py
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
```

The pretty-printed `messages` structure comes out as follows:

```python
[{'content': 'You are a helpful assistant who answers questions directly and as briefly as possible.\n'
             "Consider the following context and answer the user's question.\n"
             "If you cannot answer with the given context, just say you don't know.\n"
             '\n',
  'role': 'system'},
 {'content': '=== BEGIN CONTEXT\n'
             '\n'
             'Tokens have come up before in this series, and you might be wondering. "What are those, exactly?" Tokens '
             "are a really important concept with LLMs. When an LLM is dealing with language, it doesn't do so "
             'character by character, but it breaks down a given language into statistically useful groupings of '
             'characters, which are then identified with integer numbers. For example the characters "ing" occur '
             "pretty frequently, so a tokenizer might group those as a single token in many circumstances. It's "
             'sensitive to the surrounding character sequence, though, so the word "sing" might well be encoded as a '
             'single token of its own, regardless of containing "ing".\n'
             '\n'
             'The best way to get a feel of LLM tokenization is to play around with sample text and see how it gets '
             'converted. Luckily there are many tools out there to help, including [the simple llama-tokenizer-js '
             'playground](https://belladoreai.github.io/llama-tokenizer-js/example-demo/build/) web app which allows '
             'you to enter text and see how the popular Llama LLMs would tokenize them.\n'
             '\n'
             '## Wait, what are tokens again?\n'
             '\n'
             "The colors don't mean anything special in themselves. They're just visual tiling to separate the tokens. "
             'Notice how start of text is a special token `<s>`. You might remember we also encountered some other '
             'special tokens such as `<|im_start|>` (begin conversation turn) in previous articles. LLM pre-training '
             'and fine-tuning changes the way things are tokenized, as part of setting the entire model of language. '
             "Llama won't tokenize exactly as, say ChatGPT does, but the basic concepts stay the same.\n"
             '\n'
             '=== END CONTEXT',
  'role': 'user'},
 {'content': 'Please use the context above to respond to the following:\n'
             'How can I get a better understand what tokens are, and how they work in LLMs?',
  'role': 'user'}]
  ```

Output (the LLM's response):

> According to the context, the best way to get a better understanding of tokens in LLMs is to play around with sample text and see how it gets converted. You can use the simple llama-tokenizer-js playground web app, which allows you to enter text and see how popular LLMs would tokenize it. Additionally, you can also remember that tokens are a way for LLMs to break down a given language into statistically useful groupings of characters, identified with integer numbers.

### Faster prompt processing

One detail that popped out to my eye, from an MLX perspective, was the generation speed:

```
Prompt: 443.319 tokens-per-sec
Generation: 44.225 tokens-per-sec
```

Back in April I was seeing the following report (same 2021 Apple M1 Max MacBook Pro with 64GB RAM):

```
Prompt: 84.037 tokens-per-sec
Generation: 104.326 tokens-per-sec
```

The generation speed looks slower now, but the prompt processing speed is some 5X faster, and in RAG applications, whre the prompt gets stuffed with retrieved data, this is an important figure. That said, this is a completely different model from the `h2o-danube2-1.8b-chat-MLX-4bit` from the earlier article, and many aspects of the model itself can affect prompt processing and generation speeds.

The model I've used in the code above is my new favorite, general-purpose, open-source model, `Hermes-2-Theta-Llama-3-8B`, and in particular [a 4 bit quant I converted to MLX and contributed to the community myself](https://huggingface.co/mlx-community/Hermes-2-Theta-Llama-3-8B-4bit), using techniques from my previous article in this series, ["Converting models from Hugging Face to MLX format, and sharing"](https://github.com/uogbuji/mlx-notes/blob/main/2024/conversion-etc.md).

# Best Practices: Chunk Size and Embedding Model Selection

## Optimizing Chunk Size for RAG

Chunk size plays a critical role in the effectiveness and efficiency of Retrieval-Augmented Generation (RAG) systems. The right chunk size balances the need for detailed, relevant retrieval with the speed and faithfulness of generated responses.

- **Precision vs. Context:** Smaller chunks (e.g., 250–256 tokens) enable more precise retrieval, as each chunk is focused on a narrow context. However, if chunks are too small, important context may be lost, leading to fragmented or incomplete answers.
- **Larger Chunks:** Larger chunks (e.g., 512 tokens or a paragraph) provide more context, reducing the risk of missing relevant details, but can dilute the representation if multiple topics are included, potentially lowering retrieval precision and slowing response generation.
- **Experimentation is Key:** There is no universal optimal chunk size. Start with sizes between 250 and 512 tokens and adjust based on your data and use case. Monitor both retrieval accuracy and system latency to find the best balance.
- **Semantic Chunking:** Advanced strategies, such as semantically informed chunking (e.g., the SPLICE method), can further improve retrieval by aligning chunk boundaries with natural topic or section breaks, preserving meaning and context.

## Choosing the Right Embedding Model

The choice of embedding model directly impacts retrieval quality, system performance, and scalability:

- **Model Benchmarks:** Use benchmarks like the Massive Text Embedding Benchmark (MTEB) to compare models on tasks relevant to your application, such as retrieval, semantic similarity, and reranking
- **Dense vs. Sparse vs. Hybrid:** Dense models (e.g., E5, MiniLM) excel at semantic search, while sparse models (e.g., BM25) are better for keyword matching. Hybrid approaches often yield the best results, especially for heterogeneous or domain-specific data
- **Model Context Window:** Ensure the model’s maximum token limit aligns with your chosen chunk size. For most RAG applications, models supporting 512 tokens per embedding are sufficient, but longer context windows may be needed for larger documents
- **Efficiency and Domain Fit:** Consider inference speed, memory requirements, and how well the model handles your domain’s language and structure. Test multiple models and measure performance on your actual data to guide selection

## Summary: Chunk Size and Embedding Model tips

| Aspect                   | Recommendation                                     |
|--------------------------|----------------------------------------------------|
| Chunk Size               | Start with 250–512 tokens; adjust as needed        |
| Chunking Strategy        | Prefer semantic or paragraph-based chunking        |
| Embedding Model          | Use MTEB or real-world benchmarks for selection    |
| Model Type               | Dense for semantics; hybrid for complex datasets   |
| Context Window           | Ensure model supports your chunk size              |
| Evaluation               | Test for faithfulness, relevance, and efficiency   |

By carefully tuning chunk size and embedding model choice, you can significantly improve both the precision and responsiveness of your RAG system.

# Going beyond

These are the basic bones of RAG. Using just the code so far, you already have a lot of basis for experimentation. You can change the chunk size of the data stored in the vector DB—an adjustment which might surprise you in the degree of its effects. You can play with `SCORE_THRESHOLD` and `MAX_CHUNKS` to dial up or down what gets stuffed into the prompt for generation.

That's just scratching the surface. There are a dizzying array of techniques and variations to RAG. Just to name a selection, you can:

* use overlap with the chunking, so that you're less likely to chop apart or orphan the context of each chunk
* have multiple levels of chunking, e.g. chunking document section headers as well as their contents, sometimes called hierarchical RAG
* base the retrieval on more basic SQL or other traditional database query rather than vector search, perhaps even using a coding LLM to generate the SQL (yes, there are special security implications to this)
* use text matching rather than semantic vector search
* take retrieved chunks and re-summarize them using an LLM before sending them for generation (contextual compression), or re-assess their relevance (reranking)
* retrieve and stuff with structured knowledge graphs rather than loose text
* use an LLM to rewrite the user's prompt to better suit the context (while maintaining fidelity to the original)
* structure the stuffing of the prompts into a format to match the training of a context obedient generation LLM

Of course you can mix and match all the above, and so much more. RAG is really just an onramp to engineering, rather than its destination. As I continue this article series, I'll probably end up touching on many other advanced RAG techniques.

For now, you have a basic idea of how to use RAG in MLX, and you're mostly limited by your imagination. Load up your retrieval DB with your your company's knowledgebase to create a customer self-help bot. Load it up with your financials to create a prep tool for investor reporting. Load up with all your instant messages so you can remember whom to thank about that killer restaurant recommendation once you get around to trying it. Since you're using a locally-hosted LLM, courtesy MLX, you can run such apps entirely airgapped and have few of the privacy concerns from using e.g. OpenAI, Anthropic or Google.

# Its data all the way down

At the heart of AI has always been high quality data at high volume. RAG, if anything makes this connection far more obvious. If you want to gain its benefits, you have to be ready to commit to sound data architecture and management. We all know that garbage in leads to garbage out, but it's especially pernicious to deal with garbage out that's been given a spit shine by an eager LLM during generation.

There is a lot of energy around RAG projects, but they hide a dirty little secret: they tend to look extremely promising in prototype phases, and then run into massive engineering difficulties on the path towards full product status. A lot of this is because, to be frank, organizations have often spent so much time cutting corners in their data engineering that they just don't have the right fuel for RAG, and they might not even realize where their pipelines are falling short.

RAG is essentially the main grown-up LLM technique we have right now. It's at the heart of many product initiatives, including many of my own ones. Don't ever think, however, that it's a cure-all for the various issues in GenAI, such as hallucination and unpredictable behavior. In addition to making sure you have your overall data engineering house in order, be ready to implement sound AI Ops, with a lot of testing and ongoing metrics. There's no magic escape from this if you want to take the benefits of AI at scale.

<!-- 
# The Nuisance is in the nuances…

## …But the dazzle is in the data

# A final word of caution
 -->

# Cultural accompaniment

This time I'm going with some Indian/African musical syncretism right in Chocolate City, USA. It's a DJ set by Priyanka, who put on one of my favorite online DJ sets ever mashing up Amapiano grooves with hits from India. The live vibe is…damn! I mean the crowd starts by rhythmically chanting "DeeeeeeJaaaaay, we wanna paaaaartaaaaay", and not three minutes in a dude from the crowd jumps in with a saxophone. It's a great way to set a creative mood while puzzling through your RAG content chunking strategy.

[![Amapiano Live Mix | AmapianoDMV x PRIYANKA (of Indiamapiano fame)](https://i.ytimg.com/vi/8f3e8aMNDf0/hqdefault.jpg)](https://www.youtube.com/watch?v=8f3e8aMNDf0)
