import statistics
import sys

import mlx.core
import mlx.nn
import mlx_lm

from . import const


def main():
    if len(sys.argv) != 2:
        print("Usage: mlx_eval.compare <target_model_path>")
        sys.exit(1)

    target_model_path = sys.argv[1]

    mlx.core.clear_cache()

    print("Loading target model...")
    target_model, _ = mlx_lm.load(target_model_path)
    target_memory = mlx.core.get_active_memory()

    print("Loading reference log-probabilities...")
    ref_log_probs = mlx.core.load(const.LOGPROBS_FILE)

    print("Loading reference prompt...")
    prompt = mlx.core.load(const.TOKENIZED_PROMPT_FILE)

    print("Loading reference perplexity...")
    ref_ppl = mlx.core.load(const.PERPLEXITY_FILE).item()

    print("Calculating log-probabilities...")
    # raw logits per token from forward pass over vocabulary (batch_size, max_tokens, vocab_size)
    target_logits = target_model(prompt)

    del target_model
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

    print(f"\nKLD mean: {kld_mean:.6f}")
    print(f"KLD p95: {kld_p95:.6f}")
    print(f"KLD p99: {kld_p99:.6f}")
    print(f"PPL mean: {target_ppl:.6f}")
    print(f"PPL delta: {ppl_delta:+.6f}")

    target_model_memory_gib = target_memory / (1024**3)
    print(f"RAM: {target_model_memory_gib:.2f}")

if __name__ == "__main__":
    main()
