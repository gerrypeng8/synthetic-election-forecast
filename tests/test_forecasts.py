"""Forecast checks using known examples and deliberately broken input data."""

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from scripts.forecast import make_forecasts
from scripts.simulate_data import simulate_data


def tiny_inputs():
    frame = pl.DataFrame({"race_id": ["test", "test"], "cell_id": ["a", "b"],
                          "population": [80, 20], "true_share_a": [0.0, 1.0]})
    responses = pl.DataFrame({"response_id": [0, 1, 2], "race_id": ["test"] * 3,
                              "cell_id": ["a", "b", "b"], "model": ["test_model"] * 3,
                              "vote_a": [0, 1, None]})
    return frame, responses


def test_known_weighted_result_and_missing_answer():
    frame, responses = tiny_inputs()
    summary, _ = make_forecasts(frame, responses)
    row = summary.row(0, named=True)
    assert row["weighted_share_a"] == pytest.approx(0.2)
    assert row["raw_share_a"] == pytest.approx(0.5)
    assert row["valid_answers"] == 2
    assert row["missing_answers"] == 1


def test_forecast_coverage_bounds_and_repeatability():
    frame, responses = simulate_data()
    summary, draws = make_forecasts(frame, responses)
    other_summary, other_draws = make_forecasts(frame, responses)
    assert_frame_equal(summary, other_summary)
    assert_frame_equal(draws, other_draws)
    assert summary.height == 6
    assert draws.height == 6 * 2000
    assert summary.select("race_id", "model").unique().height == 6
    for column in ["weighted_share_a", "weighted_share_b", "win_probability_a",
                   "win_probability_b", "lower_95_a", "upper_95_a"]:
        assert summary[column].is_finite().all()
        assert summary[column].is_between(0, 1).all()
    assert (summary["lower_95_a"] < summary["upper_95_a"]).all()
    assert np.allclose(summary["weighted_share_a"] + summary["weighted_share_b"], 1)
    assert np.allclose(summary["win_probability_a"] + summary["win_probability_b"], 1)
    assert draws["share_a"].is_between(0, 1).all()
    assert np.allclose(draws["share_a"] + draws["share_b"], 1)


def test_weighting_reduces_known_sampling_bias():
    frame, responses = simulate_data()
    summary, _ = make_forecasts(frame, responses)
    # Only model 1 follows the known population probabilities without extra model bias.
    unbiased_model = summary.filter(pl.col("model") == "example_model_1")
    raw_error = (unbiased_model["raw_share_a"] - unbiased_model["true_population_share_a"]).abs()
    weighted_error = (unbiased_model["weighted_share_a"] - unbiased_model["true_population_share_a"]).abs()
    assert (weighted_error < raw_error).all()
    assert (weighted_error < 0.04).all()


def test_negative_population_is_rejected():
    frame, responses = tiny_inputs()
    frame = frame.with_columns(pl.lit(-1).alias("population"))
    with pytest.raises(ValueError, match="Population counts"):
        make_forecasts(frame, responses)


def test_duplicate_population_cell_is_rejected():
    frame, responses = tiny_inputs()
    frame = pl.concat([frame, frame.head(1)])
    with pytest.raises(ValueError, match="cell IDs"):
        make_forecasts(frame, responses)


def test_invalid_vote_is_rejected():
    frame, responses = tiny_inputs()
    responses = responses.with_columns(pl.lit(2).alias("vote_a"))
    with pytest.raises(ValueError, match="Votes must"):
        make_forecasts(frame, responses)


def test_unknown_cell_is_rejected():
    frame, responses = tiny_inputs()
    responses = responses.with_columns(pl.lit("unknown").alias("cell_id"))
    with pytest.raises(ValueError, match="unknown race or cell"):
        make_forecasts(frame, responses)


def test_group_without_answers_is_rejected():
    frame, responses = tiny_inputs()
    responses = responses.filter(pl.col("cell_id") == "a")
    with pytest.raises(ValueError, match="No valid answers"):
        make_forecasts(frame, responses)


def test_duplicate_response_is_rejected():
    frame, responses = tiny_inputs()
    responses = pl.concat([responses, responses.head(1)])
    with pytest.raises(ValueError, match="Response IDs"):
        make_forecasts(frame, responses)
