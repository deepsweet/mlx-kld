import sys

import mlx.core
import mlx.nn
import mlx_lm
import numpy


def main():
    if len(sys.argv) != 3:
        print("Usage: reference.py <reference_model_path> <max_tokens>")
        sys.exit(1)

    ref_model_path = sys.argv[1]
    max_tokens = int(sys.argv[2])

    INPUT_PROMPT_FILE = "prompt.txt"
    OUTPUT_LOG_PROBS_FILE = "reference.npy"
    OUTPUT_PROMPT_FILE = "prompt.npy"

    mlx.core.clear_cache()

    print("Loading reference model...")
    ref_model, ref_tokenizer, ref_config = mlx_lm.load(ref_model_path, return_config=True)

    ref_model_context = ref_config.get("max_position_embeddings")
    if ref_model_context is None:
        text_config = ref_config.get("text_config", {})
        ref_model_context = text_config.get("max_position_embeddings")
    if ref_model_context is None:
        raise ValueError("Could not determine model maximum context length")

    if max_tokens > ref_model_context:
        raise ValueError(f"max_tokens {max_tokens} > model max context {ref_model_context}")

    print("Loading prompt...")
    with open(INPUT_PROMPT_FILE, encoding="utf-8") as f:
        text = f.read()

    token_ids = ref_tokenizer.encode(text, truncation=True, max_length=max_tokens)
    token_len = len(token_ids)

    if token_len < max_tokens:
        raise ValueError(f"Prompt {token_len} < max_tokens {max_tokens}")

    fixed_input = mlx.core.array(token_ids)[None]

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

if __name__ == "__main__":
    main()
