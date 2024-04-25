# Hands-on Intro to MLX April 2024

via RMAIIG, Boulder, CO

[Presentation slides](https://docs.google.com/presentation/d/1IUNuFcS3YrkPFn7Oxw7uoB94TMnIK_wD1Pxu3-tzTjo/edit?usp=sharing)

## A few key pointers

* GitHub: [ml-explore/mlx](https://github.com/ml-explore/mlx)—core framework
  * [Also via PyPI: mlx](https://pypi.org/project/mlx/)
* GH: [ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples)—quick code; various use cases
  * [Also via PyPI (language AI bits): mlx-lm](https://pypi.org/project/mlx-lm/)
* HuggingFace: [mlx-community](https://huggingface.co/mlx-community)
* GH: [uogbuji/mlx-notes](https://github.com/uogbuji/mlx-notes/) articles from which this presentation originated

## Python setup 

```sh
pip install mlx mlx-lm
```

```sh
git clone https://github.com/ml-explore/mlx-examples.git
cd mlx-examples/llms
pip install -U .
```

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

## Converting your own models

```sh
python -m mlx_lm.convert
--hf-path h2oai/h2o-danube2-1.8b-chat
--mlx-path ./mlx/h2o-danube2-1.8b-chat -q
```

## OK enough words; what about images?

```sh
cd mlx-examples/stable_diffusion
python txt2image.py -v --model sd --n_images 4 --n_rows 2 --cfg 8.0 --steps 50 --output test.png "Boulder flatirons"
```
