import sys
import mlx.core
import mlx.nn
import mlx_lm
import numpy

if len(sys.argv) != 2:
    print("Usage: reference.py <reference_model_path>")
    sys.exit(1)

ref_model_path = sys.argv[1]

MAX_TOKENS = 8192
INPUT_PROMPT_FILE = "prompt.txt"
OUTPUT_LOG_PROBS_FILE = "reference.npy"
OUTPUT_PROMPT_FILE = "prompt.npy"

def load_full_prompt(file_path, tokenizer, max_tokens):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    token_ids = tokenizer.encode(text)
    token_len = len(token_ids)

    if token_len < max_tokens:
        raise ValueError(f"File too short: {token_len} < {max_tokens}")
        
    if token_len > max_tokens:
        print(f"Truncating prompt from {token_len} to {max_tokens} tokens...")
        token_ids = token_ids[:max_tokens]

    return mlx.core.array(token_ids)[None]

mlx.core.clear_cache()

print("Loading reference model...")
ref_model, ref_tokenizer = mlx_lm.load(ref_model_path)

print("Loading prompt...")
fixed_input = load_full_prompt(INPUT_PROMPT_FILE, ref_tokenizer, MAX_TOKENS)

print("Calculating log-probabilities...")
ref_logits = ref_model(fixed_input)
ref_log_probs = mlx.nn.log_softmax(ref_logits, axis=-1)

print("Saving artifacts...")
if ref_log_probs.dtype == mlx.core.bfloat16:
    ref_log_probs_arr = numpy.array(ref_log_probs.astype(mlx.core.float32))
else:
    ref_log_probs_arr = numpy.array(ref_log_probs)

fixed_input_arr = numpy.array(fixed_input)

numpy.save(OUTPUT_LOG_PROBS_FILE, ref_log_probs_arr)
numpy.save(OUTPUT_PROMPT_FILE, fixed_input_arr)