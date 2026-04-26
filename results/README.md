## Qwen3.6-35B-A3B

- source model: [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- oQ sensitivity model: affine Q8 bfloat16
- data type: float16
- prompt: 16384 tokens of Aes Sedai's [combined_all_micro.txt](https://huggingface.co/AesSedai/GLM-4.5-GGUF/raw/main/combined_all_micro.txt)

<img src="./Qwen3.6-35B-A3B.png" width="760" height="820" alt="Qwen3.6-35B-A3B KLD/RAM chart"/>

### oQ

| Quantization | KL Divergence (nats) | RAM (GiB) |
|--------------|----------------------|-----------|
| oQ2          | 0.246338             | 11.40     |
| oQ3          | 0.134155             | 14.96     |
| oQ3.5        | 0.126221             | 16.17     |
| oQ4          | 0.016815             | 18.98     |
| oQ5          | 0.008270             | 22.76     |
| oQ6          | 0.006718             | 26.51     |
| oQ8          | 0.002277             | 34.27     |

### Q

| Quantization | KL Divergence (nats) | RAM (GiB) |
|--------------|----------------------|-----------|
| Q2*          | 3.042969             | 10.10     |
| Q3           | 0.206299             | 14.14     |
| Q4           | 0.054230             | 18.17     |
| Q5           | 0.015419             | 22.20     |
| Q6           | 0.007050             | 26.23     |
| Q8           | 0.000926             | 34.30     |
| MXFP4        | 0.097501             | 17.16     |
| MXFP8        | 0.038293             | 33.29     |

<sup>*Q2 is off the chart</sup>