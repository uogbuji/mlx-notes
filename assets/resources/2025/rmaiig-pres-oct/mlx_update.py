# Main installs
# pip install mlx mlx_lm
# Or if you use uv (recommended):
# uv pip install mlx mlx_lm

# Basic inference
from mlx_lm import load, generate
# Took e.g. to download the ~12GB model weights
# model, tokenizer = load('mlx-community/gpt-oss-20b-MXFP4-Q4')
model, tokenizer = load('mlx-community/Llama-3.2-3B-Instruct-4bit')

response = generate(model, tokenizer, prompt="A fun limerick about Boulder, Colorado is:", verbose=True)
print(response)

messages = [
  {'role': 'system', 'content': 'You are a friendly chatbot who always responds in the style of a talk show host'},
  {'role': 'user', 'content': 'Do you have any advice for the CU Buffs quarterback to help them win the game?'}]
chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
response = generate(model, tokenizer, prompt=chat_prompt, verbose=True)

# VLM
# pip install mlx_vlm torch torchvision torchaudio  # Not sure why torchvision is needed
# from mlx_vlm import load as vlm_load, generate as vlm_generate
# vlm_model, vlm_tokenizer = vlm_load('mlx-community/Qwen2.5-VL-3B-Instruct-4bit')

MLX_VLM_CLI = '''
python -m mlx_vlm.generate \
--model mlx-community/Qwen2.5-VL-3B-Instruct-4bit \
--max-tokens 1024 \
--temp 0.0 \
--image assets/images/2024/black-panther-hulk-cover-open-source-llm-800x500.jpg \
--prompt "Describe image in detail; include all text"
'''

# https://d1yei2z3i6k35z.cloudfront.net/2779827/6877f5f879110_RMAIROUND-white.png
# https://secure.meetupstatic.com/photos/event/2/0/d/2/clean_517328402.webp

# Audio

MLX_VLM_AUDIO_CLI = '''
mlx_vlm.generate --model mlx-community/gemma-3n-E2B-it-4bit \
    --max-tokens 1024
    --prompt "Describe what you hear"
    --audio '/Users/uche/Downloads/Voicy_PSST I SEE DEAD PEOPLE.mp3'
'''

# Imagegen

# Using MFLUX https://github.com/filipstrand/mflux

# pip install mflux
# Note: You need a Hugging Face account and to be logged in with `huggingface-cli login`
# The first time you run this you will be required to accept a license agreement on HF

MFLUX_CLI = '''
mflux-generate --model schnell --prompt "Luxury food photograph" --steps 2 --seed 2 -q 4
'''

# ---

# Servers: hosting models for inference

# mlx-omni-server https://github.com/madroidmaq/mlx-omni-server
# "MLX Omni Server is a local inference server powered by Apple's MLX framework, specifically designed for Apple Silicon (M-series) chips. It implements OpenAI-compatible API endpoints, enabling seamless integration with existing OpenAI SDK clients while leveraging the power of local ML inference."

# Note: Use with caution, and probably an isolated venv because it to pins many old dependency versions
# pip install mlx-omni-server

# mlx-openai-server https://github.com/cubist38/mlx-openai-server
# "OpenAI-compatible endpoints for MLX models [using] Python and [FastAPI], it provides an efficient, scalable, and user-friendly solution for running MLX-based multimodal models locally with an OpenAI-compatible interface. The server supports text, vision, audio processing, and image generation capabilities with enhanced Flux-series model support."

# Note: Use with caution, and probably an isolated venv because it to pins many old dependency versions
# pip install mlx-openai-server

# A more basic option: MLX server
MLX_SERVER_CLI = '''
mlx serve --model mlx-community/Llama-3.2-3B-Instruct-4bit --port 8080
'''
