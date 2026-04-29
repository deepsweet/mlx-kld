import statistics
import sys

import mlx.core
import mlx.nn
import mlx_lm

from . import const


def main():
    if len(sys.argv) != 2:
        print("Usage: compare_target.py <target_model_path>")
        sys.exit(1)

    target_model_path = sys.argv[1]

    mlx.core.clear_cache()

    print("Loading target model...")
    target_model, _ = mlx_lm.load(target_model_path)
    target_memory = mlx.core.get_active_memory()

    print("Loading reference log-probabilities...")
    ref_log_probs = mlx.core.load(const.LOGPROBS_FILE)

    print("Loading prompt...")
    prompt = mlx.core.load(const.TOKENIZED_PROMPT_FILE)

    print("Calculating log-probabilities...")
    target_logits = target_model(prompt)
    target_log_probs = mlx.nn.log_softmax(target_logits, axis=-1)

    del target_model
    mlx.core.clear_cache()

    print("Calculating KL Divergence...")
    kld_none = mlx.nn.losses.kl_div_loss(target_log_probs, ref_log_probs, reduction="none")
    kld_mean = mlx.core.mean(kld_none).item()
    kld_list = kld_none.flatten().tolist()
    kld_p95 = statistics.quantiles(kld_list, n=100)[-5]
    kld_p99 = statistics.quantiles(kld_list, n=100)[-1]

    print(f"\nKLD mean: {kld_mean:.6f}")
    print(f"KLD p95: {kld_p95:.6f}")
    print(f"KLD p99: {kld_p99:.6f}")

    target_model_memory_gib = target_memory / (1024**3)
    print(f"RAM: {target_model_memory_gib:.2f}")

if __name__ == "__main__":
    main()
