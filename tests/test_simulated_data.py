"""Checks on fictional inputs, independent of the forecast calculations."""

import polars as pl
from polars.testing import assert_frame_equal

from scripts.simulate_data import simulate_data


def test_simulation_is_reproducible():
    frame, responses = simulate_data()
    other_frame, other_responses = simulate_data()
    assert_frame_equal(frame, other_frame)
    assert_frame_equal(responses, other_responses)


def test_population_and_response_structure():
    frame, responses = simulate_data()
    assert frame.height == 18  # Three races, three ages, two education groups.
    assert frame["cell_id"].n_unique() == 18
    assert (frame["population"] > 0).all()
    assert frame["true_share_a"].is_between(0, 1).all()
    totals = frame.group_by("race_id").agg(pl.col("population").sum())
    assert totals["population"].to_list() == [100000] * 3
    assert responses["response_id"].n_unique() == responses.height
    assert responses["model"].n_unique() == 2
    assert responses["vote_a"].null_count() > 0
    assert responses["vote_a"].drop_nulls().is_in([0, 1]).all()
    assert responses.join(frame.select("cell_id"), on="cell_id", how="anti").is_empty()
    assert frame["simulated"].all() and responses["simulated"].all()


def test_demographics_and_support_are_related():
    frame, _ = simulate_data()
    # Education composition varies by age, and vote probabilities vary across cells.
    assert frame["population"].n_unique() > 2
    assert frame["true_share_a"].n_unique() > 6
