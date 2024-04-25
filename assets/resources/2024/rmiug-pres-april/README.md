# Hands-on Intro to MLX April 2024

via RMAIIG, Boulder, CO

Files:


## Downloading & running an LLM

```py
from mlx_lm import load, generate
model, tokenizer = load('mlx-community/Phi-3-mini-4k-instruct-4bit')
```

## Doing the generation

```py
p = 'Can a Rocky Mountain High take me to the sky?'
resp = generate(model, tokenizer, prompt=p, verbose=True)
```

## Tokens & tokenizer

Relevant link: [llama-tokenizer-js playground](https://belladoreai.github.io/llama-tokenizer-js/example-demo/build/), a simple web app which allows you to enter text and see how the popular Llama LLMs would tokenize them

## Let’s chat about chat

```py
sysmsg = 'You\'re a friendly, helpful chatbot'
messages = [
  {'role': 'system', 'content': sysmsg},
  {'role': 'user', 'content': 'How are you today?'}]
chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
response = generate(model, tokenizer, prompt=chat_prompt, verbose=True)
```


