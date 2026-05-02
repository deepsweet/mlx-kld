import math
import unittest.mock

import mlx.core
import mlx.nn
import pytest

import mlx_eval

EXPECTED_PROMPT = [[0, 1, 2, 3]]
LOG_PROB_VALUE = -math.log(10)
# (4 sequences x 10 vocab)
EXPECTED_LOGPROBS_ARR = [[[LOG_PROB_VALUE] * 10] * 4]
EXPECTED_LOGPROBS = mlx.core.array(EXPECTED_LOGPROBS_ARR)
EXPECTED_PPL = 10

class UniformLogitModel(mlx.nn.Module):
    def __init__(self, vocab_size=10, embed_dim=4):
        super().__init__()
        self.embed = mlx.nn.Embedding(vocab_size, embed_dim)
        self.lm_head = mlx.nn.Linear(embed_dim, vocab_size)

    def __call__(self, x):
        x = self.embed(x)
        return self.lm_head(x)

def set_uniform_weights(model):
    model.embed.weight = mlx.core.zeros_like(model.embed.weight)
    model.lm_head.weight = mlx.core.zeros_like(model.lm_head.weight)
    model.lm_head.bias = mlx.core.zeros_like(model.lm_head.bias)

class FixedTokenizer:
    def encode(self, text, truncation=True, max_length=None): # noqa: ARG002
        return ([0, 1, 2, 3] * (max_length // 4 + 1))[:max_length]

def test_run_reference(tmp_path):
    model = UniformLogitModel(vocab_size=10, embed_dim=4)
    set_uniform_weights(model)
    tokenizer = FixedTokenizer()

    prompt_text = "a b c d"
    max_tokens = 4

    src_file = tmp_path / "prompt.txt"
    src_file.write_text(prompt_text)

    with unittest.mock.patch("mlx_eval.reference.mlx_lm.load") as mock_load:
        mock_load.return_value = (model, tokenizer, {"max_position_embeddings": 1000})

        ref_data = mlx_eval.reference.run_reference(
            ref_model_path="dummy",
            max_tokens=max_tokens,
            source_prompt_file=str(src_file),
            verbose=False,
        )

    assert ref_data["prompt"].tolist() == EXPECTED_PROMPT
    assert mlx.core.allclose(ref_data["log_probs"], EXPECTED_LOGPROBS, atol=1e-6)
    assert ref_data["perplexity"].item() == pytest.approx(EXPECTED_PPL, abs=1e-6)
