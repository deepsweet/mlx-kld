import sys

import mlx.core
import mlx.nn
import mlx_lm

from . import const


def main():
    if len(sys.argv) != 3:
        print("Usage: reference.py <reference_model_path> <max_tokens>")
        sys.exit(1)

    ref_model_path = sys.argv[1]
    max_tokens = int(sys.argv[2])

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
    with open(const.SOURCE_PROMPT_FILE, encoding="utf-8") as f:
        prompt_text = f.read()

    # tokenize prompt and truncate to max_tokens with no padding
    token_ids = ref_tokenizer.encode(prompt_text, truncation=True, max_length=max_tokens)
    token_len = len(token_ids)

    if token_len < max_tokens:
        raise ValueError(f"Prompt {token_len} < max_tokens {max_tokens}")

    # add batch dimension (batch_size, max_tokens)
    prompt = mlx.core.array(token_ids)[None]

    print("Calculating log-probabilities...")
    # raw logits per token from forward pass over vocabulary (batch_size, max_tokens, vocab_size)
    ref_logits = ref_model(prompt)

    del ref_model
    mlx.core.clear_cache()

    # convert logits to numerically stable log-probabilities along the vocabulary axis
    ref_log_probs = mlx.nn.log_softmax(ref_logits, axis=-1)

    print("Calculating perplexity...")
    # drop last token because there is no "next token" to predict
    shift_logits = ref_logits[:, :-1, :]
    # drop first token because there is no previous token to use as context for prediction
    shift_prompt = prompt[:, 1:]
    # cross-entropy loss between the predicted logits and target tokens
    cross_entropy = mlx.nn.losses.cross_entropy(shift_logits, shift_prompt, reduction="mean")
    # convert cross-entropy to perplexity
    ref_ppl = mlx.core.exp(cross_entropy).item()

    print("Saving artifacts...")
    mlx.core.save(const.LOGPROBS_FILE, ref_log_probs)
    mlx.core.save(const.TOKENIZED_PROMPT_FILE, prompt)
    mlx.core.save(const.PERPLEXITY_FILE, mlx.core.array(ref_ppl))

    print(f"\nPPL mean: {ref_ppl:.6f}")

if __name__ == "__main__":
    main()
