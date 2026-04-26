## Qwen3.6-35B-A3B

- source model: [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- oQ sensitivity model: affine Q8 bfloat16
- oQ dtype: float16
- prompt: 8192 tokens

![Qwen3.6-35B-A3B KLD/RAM chart](./Qwen3.6-35B-A3B.png)

### oQ

| Quantization | KL Divergence (nats) | RAM (GiB) |
|--------------|----------------------|-----------|
| oQ2          | 1.887695             | 11.40     |
| oQ3          | 0.778320             | 14.96     |
| oQ3.5        | 0.719727             | 16.17     |
| oQ4          | 0.109985             | 18.98     |
| oQ5          | 0.038055             | 22.76     |
| oQ6          | 0.024841             | 26.51     |
| oQ8          | 0.010155             | 34.27     |

### Q

| Quantization | KL Divergence (nats) | RAM (GiB) |
|--------------|----------------------|-----------|
| Q2*          | 4.687500             | 10.10     |
| Q3           | 1.103516             | 14.14     |
| Q4           | 0.211914             | 18.17     |
| Q5           | 0.063660             | 22.20     |
| Q6           | 0.026718             | 26.23     |
| Q8           | 0.014196             | 34.30     |
| MXFP4        | 0.383822             | 17.16     |
| MXFP8        | 0.130360             | 33.29     |