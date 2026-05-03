import math
import unittest.mock

import mlx.core
import pytest
import utils

import mlx_eval

EXPECTED_PROMPT = [[0, 1, 2, 3]]
LOG_PROB_VALUE = -math.log(10)
EXPECTED_LOGPROBS_ARR = [[[LOG_PROB_VALUE] * 10] * 4]
EXPECTED_LOGPROBS = mlx.core.array(EXPECTED_LOGPROBS_ARR)
EXPECTED_PPL = 10

EXPECTED_VARYING_KLD_MEAN = 3.391503923921846e-05
EXPECTED_VARYING_KLD_P95 = 9.675684850662947e-05
EXPECTED_VARYING_KLD_P99 = 0.00010317994747310877
EXPECTED_VARYING_PPL_MEAN = 10.020231246948242
EXPECTED_VARYING_PPL_DELTA = 0.020231246948242188


def test_run_compare_self(tmp_path):
    model = utils.UniformLogitModel(vocab_size=10, dims=4)
    tokenizer = utils.FixedTokenizer()

    log_probs_path = tmp_path / "logprobs.npy"
    prompt_path = tmp_path / "prompt.npy"
    ppl_path = tmp_path / "ref_ppl.npy"
    mlx.core.save(log_probs_path, EXPECTED_LOGPROBS)
    mlx.core.save(prompt_path, mlx.core.array(EXPECTED_PROMPT))
    mlx.core.save(ppl_path, mlx.core.array(EXPECTED_PPL))

    with unittest.mock.patch("mlx_eval.compare.mlx_lm.load") as mock_load:
        mock_load.return_value = (model, tokenizer)
        result = mlx_eval.compare.run_compare(
            target_model_path="dummy",
            tokenized_prompt_path=str(prompt_path),
            ref_log_probs_path=str(log_probs_path),
            ref_perplexity_path=str(ppl_path),
        )

    assert result["kld_mean"] == 0
    assert result["kld_p95"] == 0
    assert result["kld_p99"] == 0
    assert result["ppl_mean"] == pytest.approx(EXPECTED_PPL, abs=1e-6)
    assert result["ppl_delta"] == pytest.approx(0, abs=1e-6)
    assert result["memory"] > 0


def test_run_compare_varying_perturbation(tmp_path):
    prompt_tensor = mlx.core.array(EXPECTED_PROMPT)

    log_probs_path = tmp_path / "logprobs.npy"
    prompt_path = tmp_path / "prompt.npy"
    ppl_path = tmp_path / "ref_ppl.npy"
    mlx.core.save(log_probs_path, EXPECTED_LOGPROBS)
    mlx.core.save(prompt_path, prompt_tensor)
    mlx.core.save(ppl_path, mlx.core.array(EXPECTED_PPL))

    target_model = utils.PositionDependentModel(vocab_size=10, dims=4, base_perturbation=0.01)
    tokenizer = utils.FixedTokenizer()

    with unittest.mock.patch("mlx_eval.compare.mlx_lm.load") as mock_load:
        mock_load.return_value = (target_model, tokenizer)
        result = mlx_eval.compare.run_compare(
            target_model_path="dummy",
            tokenized_prompt_path=str(prompt_path),
            ref_log_probs_path=str(log_probs_path),
            ref_perplexity_path=str(ppl_path),
        )

    assert result["kld_mean"] == EXPECTED_VARYING_KLD_MEAN
    assert result["kld_p95"] == EXPECTED_VARYING_KLD_P95
    assert result["kld_p99"] == EXPECTED_VARYING_KLD_P99
    assert result["ppl_mean"] == EXPECTED_VARYING_PPL_MEAN
    assert result["ppl_delta"] == EXPECTED_VARYING_PPL_DELTA
    assert result["memory"] > 0
