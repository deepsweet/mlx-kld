import gc
import statistics
import sys

import mlx.core
import mlx.nn
import mlx_lm

from . import const


def run_compare(target_model_path, tokenized_prompt_path, ref_log_probs_path, ref_perplexity_path):
    """
    Run a target model comparison using reference log-probabilities and tokenized prompt,
    and return KL divergence metrics, perplexity, and memory usage.
    """

    mlx.core.clear_cache()

    print("Loading target model...")
    target_model, _ = mlx_lm.load(target_model_path)
    target_memory = mlx.core.get_active_memory()

    print("Loading reference log-probabilities...")
    ref_log_probs = mlx.core.load(ref_log_probs_path)

    print("Loading reference prompt...")
    prompt = mlx.core.load(tokenized_prompt_path)

    print("Loading reference perplexity...")
    ref_ppl = mlx.core.load(ref_perplexity_path).item()

    print("Calculating log-probabilities...")
    # raw logits per token from forward pass over vocabulary (batch_size, max_tokens, vocab_size)
    target_logits = target_model(prompt)

    del target_model
    gc.collect()
    mlx.core.clear_cache()

    # convert logits to numerically stable log-probabilities along the vocabulary axis
    target_log_probs = mlx.nn.log_softmax(target_logits, axis=-1)

    print("Calculating KL divergence...")
    # per-token KL Divergence summed over vocabulary (batch_size, max_tokens)
    kld_none = mlx.nn.losses.kl_div_loss(target_log_probs, ref_log_probs, reduction="none")
    kld_mean = mlx.core.mean(kld_none).item()
    kld_list = kld_none.flatten().tolist()
    kld_p95 = statistics.quantiles(kld_list, n=100)[-5]
    kld_p99 = statistics.quantiles(kld_list, n=100)[-1]

    print("Calculating perplexity...")
    # drop last token because there is no "next token" to predict
    shift_logits = target_logits[:, :-1, :]
    # drop first token because there is no previous token to use as context for prediction
    shift_prompt = prompt[:, 1:]
    # cross-entropy loss between the predicted logits and target tokens
    cross_entropy = mlx.nn.losses.cross_entropy(shift_logits, shift_prompt, reduction="mean")
    # convert cross-entropy to perplexity
    target_ppl = mlx.core.exp(cross_entropy).item()
    ppl_delta = target_ppl - ref_ppl

    return {
        "kld_mean": kld_mean,
        "kld_p95": kld_p95,
        "kld_p99": kld_p99,
        "ppl_mean": target_ppl,
        "ppl_delta": ppl_delta,
        "memory": target_memory,
    }


def main():
    if len(sys.argv) != 2:
        print("Usage: mlx_eval.compare <target_model_path>")
        sys.exit(1)

    target_model_path = sys.argv[1]

    result = run_compare(
        target_model_path=target_model_path,
        tokenized_prompt_path=const.TOKENIZED_PROMPT_PATH,
        ref_log_probs_path=const.REF_LOG_PROBS_PATH,
        ref_perplexity_path=const.REF_PERPLEXITY_PATH,
    )

    print(f"\nKLD mean: {result["kld_mean"]:.6f}")
    print(f"KLD p95: {result["kld_p95"]:.6f}")
    print(f"KLD p99: {result["kld_p99"]:.6f}")
    print(f"PPL mean: {result["ppl_mean"]:.6f}")
    print(f"PPL delta: {result["ppl_delta"]:+.6f}")

    target_model_memory_gib = result["memory"] / (1024**3)
    print(f"RAM: {target_model_memory_gib:.2f}")


if __name__ == "__main__":
    main()
