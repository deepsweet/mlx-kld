# MLX KLD

Set of scripts to measure KLD, made specifically for MLX format.

## Kullback-Leibler (KL) Divergence

>KLD measures the "distance" between two probability distributions. It doesn't just look at one token; it compares the entire vector of probabilities from the original model to the quantized model. It asks, "How much information is lost when we use the quantized distribution to approximate the original one?"

Read ["Why Maybe We're Measuring LLM Compression Wrong"](https://huggingface.co/blog/rishiraj/kld-guided-quantization) and ["oQ: oMLX Universal Dynamic Quantization"](https://github.com/jundot/omlx/blob/main/docs/oQ_Quantization.md) for details.

![Qwen3.6-35B-A3B KLD/RAM chart](./results/Qwen3.6-35B-A3B.png)

See [Qwen3.6-35B-A3B.ipynb](./results/Qwen3.6-35B-A3B.ipynb) for the above chart example.

## Usage

```sh
# clone the repo
git clone git@github.com:deepsweet/mlx-kld.git
cd mlx-kld/

# install dependencies
uv venv
uv pip install -r requirements.txt

# download an original reference model
uv tool install huggingface-hub
hf download Qwen/Qwen3.6-35B-A3B

# convert it into MLX
uv tool install mlx-lm
# add `--dtype float16` if needed
mlx_lm.convert \
  --hf-path ~/.cache/huggingface/models/Qwen/Qwen3.6-35B-A3B \
  --mlx-path /path/to/Qwen3.6-35B-A3B-MLX

# prepare a diverse prompt at least 8192 tokens long
# for example, "The Fall of the House of Usher" by Edgar Allan Poe would suffice
curl -sL "https://www.gutenberg.org/cache/epub/932/pg932.txt" | tail -n +42 > prompt.txt

# load, calculate and save reference model log-probabilities
# reference.py <max_tokens> <reference_model_path>
uv run reference.py 16384 /path/to/Qwen3.6-35B-A3B-MLX

# compare a target quantized model against it
# compare_target.py <max_tokens> <target_model_path>
uv run compare.py 16384 /path/to/Qwen3.6-35B-A3B-MLX-oQ8

# cleanup when finished
rm prompt.npy reference.npy
```