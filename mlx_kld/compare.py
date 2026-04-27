import sys

import mlx.core
import mlx.nn
import mlx_lm


def main():
    if len(sys.argv) != 2:
        print("Usage: compare_target.py <target_model_path>")
        sys.exit(1)

    target_model_path = sys.argv[1]

    INPUT_LOG_PROBS_FILE = "reference.npy"
    INPUT_PROMPT_FILE = "prompt.npy"

    mlx.core.clear_cache()

    print("Loading target model...")
    target_model, target_tokenizer = mlx_lm.load(target_model_path)
    target_model_memory = mlx.core.get_active_memory()

    print("Loading reference log-probabilities...")
    ref_log_probs = mlx.core.load(INPUT_LOG_PROBS_FILE)

    print("Loading prompt...")
    fixed_input = mlx.core.load(INPUT_PROMPT_FILE)

    print("Calculating log-probabilities...")
    target_logits = target_model(fixed_input)
    target_log_probs = mlx.nn.log_softmax(target_logits, axis=-1)

    print("Calculating KL Divergence...")
    kld = mlx.nn.losses.kl_div_loss(target_log_probs, ref_log_probs, reduction="mean").item()

    print(f"\nModel memory: {target_model_memory / (1024**3):.2f} GiB")
    print(f"KL Divergence: {kld:.6f}")

if __name__ == "__main__":
    main()
