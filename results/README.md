# KLD and PPL evaluation of various MLX quantizations

>Quantization aims to reduce the precision of a model's parameter from higher bit-widths to lower bit-widths.

From ["A Visual Guide to Quantization"](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization).

Kullback-Leibler divergence (KLD):

>KLD measures the "distance" between two probability distributions. It doesn't just look at one token; it compares the entire vector of probabilities from the original model to the quantized model. It asks, "How much information is lost when we use the quantized distribution to approximate the original one?"

From ["Why Maybe We're Measuring LLM Compression Wrong"](https://huggingface.co/blog/rishiraj/kld-guided-quantization).

Perplexity (PPL):

>Perplexity measures the model’s uncertainty when predicting the next token.

KLD + PPL:

>The relationship between them reveals deeper insights. High PPL with low KLD suggests the model is fundamentally sound but might need fine-tuning. High KLD with low PPL? You’ve got a distribution mismatch that’s probably going to bite you in production.

>Use PPL as a sanity check, but trust KLD for structural fidelity.

From ["The 'Q4_K_M' Illusion: Why KL Divergence and Perplexity Are Your Only Friends in the GGUF Wild West"](https://www.banandre.com/blog/quantization-fidelity-benchmarking-kld-and-ppl-as-metrics-for-gguf-model-selection).

## Qwen3.6-35B-A3B

![Qwen3.6-35B-A3B KLD/RAM chart](./Qwen3.6-35B-A3B.svg?20250430)

### Reference

- model: [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- tool: [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) v0.4.4
- multimodal: Vision-Language
- data type: bfloat16
- context: single 16384-token prompt using Aes Sedai's [combined_all_micro.txt](https://huggingface.co/AesSedai/GLM-4.5-GGUF/raw/main/combined_all_micro.txt)
- PPL mean: 5.906250

### oQ

- ["oQ: oMLX Universal Dynamic Quantization"](https://github.com/jundot/omlx/blob/main/docs/oQ_Quantization.md)
- https://huggingface.co/collections/deepsweet/qwen36-35b-a3b
- tool: [oMLX](https://github.com/jundot/omlx) v0.3.6
- sensitivity model:
  - tool: [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) v0.4.4
  - quantization: Q8
  - mode: affine
  - group size: default, omitted
  - data type: bfloat16
- text-only: no
- data type: bfloat16

| Quantization | KLD mean (nats) | KLD p95 (nats) | KLD p99 (nats) | PPL mean | PPL delta | RAM (GiB) | 
|--------------|----------------:|---------------:|---------------:|---------:|----------:|----------:|
| oQ2          |        0.277344 |       1.062500 |       2.642969 | 7.000000 | +1.093750 |     11.40 |
| oQ3          |        0.175781 |       0.644531 |       1.438672 | 6.562500 | +0.656250 |     14.77 |
| oQ3.5        |        0.170898 |       0.636719 |       1.407422 | 6.531250 | +0.625000 |     16.00 |
| oQ4          |        0.028076 |       0.132812 |       0.263672 | 5.968750 | +0.062500 |     18.83 |
| oQ5          |        0.010254 |       0.096191 |       0.132812 | 5.937500 | +0.031250 |     22.76 |
| oQ6          |        0.008057 |       0.089355 |       0.125000 | 5.906250 | +0.000000 |     26.51 |
| oQ8          |        0.005219 |       0.081543 |       0.122070 | 5.906250 | +0.000000 |     34.27 |

### Q

- tool: [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) v0.4.4
- mode: affine
- group size: default, omitted
- data type: bfloat16

| Quantization | KLD mean (nats) | KLD p95 (nats) | KLD p99 (nats) |   PPL mean |   PPL delta | RAM (GiB) | 
|--------------|----------------:|---------------:|---------------:|-----------:|------------:|----------:|
| Q2*          |        3.093750 |       7.687500 |      10.375000 | 112.000000 | +106.093750 |     10.10 |
| Q3           |        0.230469 |       0.808594 |       1.704297 |   6.781250 |   +0.875000 |     14.14 |
| Q4           |        0.062500 |       0.230469 |       0.492188 |   6.062500 |   +0.156250 |     18.17 |
| Q5           |        0.019531 |       0.113281 |       0.188477 |   5.906250 |   +0.000000 |     22.20 |
| Q6           |        0.009094 |       0.092773 |       0.126953 |   5.906250 |   +0.000000 |     26.23 |
| Q8           |        0.005402 |       0.082031 |       0.120679 |   5.906250 |   +0.000000 |     34.30 |

<sup>*Q2 is off the chart</sup>

### MXFP

- tool: [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) v0.4.4
- mode: mxfp4 / mxfp8
- group size: default, omitted
- data type: bfloat16

| Quantization | KLD mean (nats) | KLD p95 (nats) | KLD p99 (nats) | PPL mean | PPL delta | RAM (GiB) | 
|--------------|----------------:|---------------:|---------------:|---------:|----------:|----------:|
| MXFP4        |        0.111328 |       0.382812 |       0.804688 | 6.218750 | +0.312500 |     17.16 |
| MXFP8        |        0.041992 |       0.164062 |       0.287402 | 6.031250 | +0.125000 |     33.29 |

### UD

- ["MLX Dynamic Quants"](https://unsloth.ai/docs/models/qwen3.6#mlx-dynamic-quants)
- https://huggingface.co/unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit
- https://huggingface.co/unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit

| Quantization | KLD mean (nats) | KLD p95 (nats) | KLD p99 (nats) | PPL mean | PPL delta | RAM (GiB) | 
|--------------|----------------:|---------------:|---------------:|---------:|----------:|----------:|
| 3-bit        |        0.048584 |       0.193359 |       0.453125 | 6.062500 | +0.156250 |     15.35 |
| 4-bit        |        0.016357 |       0.108887 |       0.169922 | 5.937500 | +0.031250 |     19.32 |

## Qwen3.6-27B

### Reference

- model: [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)
- tool: [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) v0.4.4
- multimodal: Vision-Language
- data type: bfloat16
- context: single 16384-token prompt using Aes Sedai's [combined_all_micro.txt](https://huggingface.co/AesSedai/GLM-4.5-GGUF/raw/main/combined_all_micro.txt)
- PPL mean: 16.875000

>[!CAUTION]
>I've come to a very interesting conclusion regarding the Qwen3.6-27B evaluation.
>
>It gets completely blown away by the very diverse and unpredictable [combined_all_micro.txt](https://huggingface.co/AesSedai/GLM-4.5-GGUF/raw/main/combined_all_micro.txt) prompt:
>
>- Reference mean PPL is 16.875000 (expected <5). The model really struggles to predict the correct next tokens.
>- Affine Q4 has a high KLD 0.456142 (expected <0.1), _lower_ mean PPL and a _negative_ PPL delta of -3.937500. This suggests that the noise introduced by quantization acts as a "regularization filter", actually helping the model on this specific prompt.
>- This highlights and stresses out the generalization limitations of the dense 27B architecture: the model likely wasn't trained on such a diverse mixture of domains simultaneously.
>- In contrast, the 35B‑A3B MoE feels robust. Its expert architecture seems to allow it to switch and route between multiple domains easily.
>- The evaluation itself is correct, because when I switch to the "community‑standard" but rather monotonous WikiText‑2, reference mean PPL becomes 4.593750, Q4 mean KLD 0.036133 and PPL delta +0.062500 (expected degradation). Also, on Edgar Poe's prose, mean PPL drops even lower to 1.359375.
>
>My intention is to keep this flaw exposed as is, because a true evaluation should not aim to please its target. One could try to get nicer and more community-aligned numbers by using [wikitext-2-raw-v1 test split](https://huggingface.co/datasets/Salesforce/wikitext/blob/main/wikitext-2-raw-v1/test-00000-of-00001.parquet).

### oQ

- ["oQ: oMLX Universal Dynamic Quantization"](https://github.com/jundot/omlx/blob/main/docs/oQ_Quantization.md)
- https://huggingface.co/collections/deepsweet/qwen36-27b
- tool: [oMLX](https://github.com/jundot/omlx) v0.3.8
- sensitivity model:
  - tool: [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) v0.4.4
  - quantization: Q8
  - mode: affine
  - group size: default, omitted
  - data type: bfloat16
- text-only: no
- data type: bfloat16

| Quantization | KLD mean (nats) | KLD p95 (nats) | KLD p99 (nats) |  PPL mean | PPL delta | RAM (GiB) | 
|--------------|----------------:|---------------:|---------------:|----------:|----------:|----------:|
| oQ4          |        0.425781 |       0.932617 |      14.000000 | 14.250000 | -2.625000 |     14.72 |

### Q

- tool: [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) v0.4.4
- mode: affine
- group size: default, omitted
- data type: bfloat16

| Quantization | KLD mean (nats) | KLD p95 (nats) | KLD p99 (nats) |   PPL mean |   PPL delta | RAM (GiB) | 
|--------------|----------------:|---------------:|---------------:|-----------:|------------:|----------:|
| Q2*          |        2.687500 |       8.812500 |      17.000000 |  31.625000 |  +14.750000 |      7.83 |
| Q3           |        0.914062 |       4.781250 |      17.375000 |  14.437500 |   -2.437500 |     10.96 |
| Q4           |        0.457031 |       1.154297 |      14.009375 |  12.937500 |   -3.937500 |     14.09 |
| Q5           |        0.332031 |       0.588867 |      11.153125 |  14.250000 |   -2.625000 |     17.23 |
| Q6           |        0.180664 |       0.128906 |       5.295312 |  17.500000 |   +0.625000 |     26.23 |
| Q8           |        0.088379 |       0.095703 |       1.183203 |  16.875000 |   +0.000000 |     20.36 |

<sup>*Q2 is off the chart</sup>

### MXFP

- tool: [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) v0.4.4
- mode: mxfp4 / mxfp8
- group size: default, omitted
- data type: bfloat16

| Quantization | KLD mean (nats) | KLD p95 (nats) | KLD p99 (nats) | PPL mean | PPL delta | RAM (GiB) | 
|--------------|----------------:|---------------:|---------------:|---------:|----------:|----------:|

### UD

- ["MLX Dynamic Quants"](https://unsloth.ai/docs/models/qwen3.6#mlx-dynamic-quants)
- https://huggingface.co/unsloth/Qwen3.6-27B-UD-MLX-3bit
- https://huggingface.co/unsloth/Qwen3.6-27B-UD-MLX-4bit
- https://huggingface.co/unsloth/Qwen3.6-27B-UD-MLX-6bit

| Quantization | KLD mean (nats) | KLD p95 (nats) | KLD p99 (nats) |  PPL mean | PPL delta | RAM (GiB) | 
|--------------|----------------:|---------------:|---------------:|----------:|----------:|----------:|
| 3-bit        |        0.511719 |       1.593750 |      14.321875 | 14.250000 | -2.625000 |     21.54 |
| 4-bit        |        0.271484 |       0.384766 |       9.509375 | 14.437500 | -2.437500 |     23.53 |
| 6-bit        |        0.103516 |       0.105957 |       1.869531 | 16.875000 | +0.000000 |     27.55 |
