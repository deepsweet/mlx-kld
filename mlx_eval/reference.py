import gc
import pathlib
import sys

import mlx.core
import mlx.nn
import mlx_lm

from . import const


def run_reference(ref_model_path, max_tokens, source_prompt_path):
    """
    Run a reference model on the prompt and return log-probabilities,
    tokenized prompt, and reference perplexity as an MLX array.
    """

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
    prompt_text = pathlib.Path(source_prompt_path).read_text(encoding="utf-8")

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
    gc.collect()
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

    return {
        "log_probs": ref_log_probs,
        "prompt": prompt,
        "perplexity": mlx.core.array(ref_ppl),
    }


def main():
    if len(sys.argv) != 3:
        print("Usage: mlx_eval.reference <reference_model_path> <max_tokens>")
        sys.exit(1)

    ref_model_path = sys.argv[1]
    max_tokens = int(sys.argv[2])

    result = run_reference(
        ref_model_path=ref_model_path,
        max_tokens=max_tokens,
        source_prompt_path=const.SOURCE_PROMPT_PATH,
    )

    print("Saving artifacts...")
    mlx.core.save(const.REF_LOG_PROBS_PATH, result["log_probs"])
    mlx.core.save(const.TOKENIZED_PROMPT_PATH, result["prompt"])
    mlx.core.save(const.REF_PERPLEXITY_PATH, result["perplexity"])

    print(f"\nPPL mean: {result["perplexity"].item():.6f}")


if __name__ == "__main__":
    main()
