"""Weight fictional responses to population cells and save illustrative forecasts."""

from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]


def validate_inputs(frame, responses):
    """Fail clearly rather than silently dropping bad records or population cells."""
    if frame.is_empty() or responses.is_empty():
        raise ValueError("Inputs must not be empty.")
    for name in ["race_id", "cell_id", "population"]:
        if frame[name].null_count():
            raise ValueError(f"Missing frame values in {name}.")
    if frame["cell_id"].n_unique() != frame.height:
        raise ValueError("Population cell IDs must be unique.")
    if (frame["population"] <= 0).any():
        raise ValueError("Population counts must be positive.")
    for name in ["response_id", "race_id", "cell_id", "model"]:
        if responses[name].null_count():
            raise ValueError(f"Missing response values in {name}.")
    if responses["response_id"].n_unique() != responses.height:
        raise ValueError("Response IDs must be unique.")
    if not responses["vote_a"].drop_nulls().is_in([0, 1]).all():
        raise ValueError("Votes must be 0, 1, or missing.")
    unknown = responses.join(frame.select("race_id", "cell_id"),
                             on=["race_id", "cell_id"], how="anti")
    if unknown.height:
        raise ValueError("Responses contain an unknown race or cell.")


def make_forecasts(frame, responses, seed=2453, n_draws=2000):
    """Return summaries and draws for each fictional race and model separately.

    Bootstrap the observed answers within each cell, then population-weight them.
    Weighted draws give an illustrative distribution over two-candidate outcomes.
    This assumes independent Bernoulli answers and does NOT measure real-world
    LLM bias, turnout uncertainty, or future election changes.
    """
    validate_inputs(frame, responses)
    if n_draws < 2:
        raise ValueError("Use at least two forecast draws.")
    rng = np.random.default_rng(seed)
    summaries = []
    draws_tables = []

    for race in frame["race_id"].unique().sort().to_list():
        cells = frame.filter(pl.col("race_id") == race).sort("cell_id")
        total_population = cells["population"].sum()
        for model in responses["model"].unique().sort().to_list():
            sample = responses.filter((pl.col("race_id") == race) & (pl.col("model") == model))
            weighted_draws = np.zeros(n_draws)
            weighted_share = 0.0
            for cell in cells.iter_rows(named=True):
                votes = sample.filter(pl.col("cell_id") == cell["cell_id"])["vote_a"].drop_nulls()
                if len(votes) == 0:
                    raise ValueError(f"No valid answers for {race}, {model}, {cell['cell_id']}.")
                successes = votes.sum()
                weight = cell["population"] / total_population
                # The point estimate is the population-weighted observed share.
                cell_share = successes / len(votes)
                weighted_share += weight * cell_share
                # For 0/1 answers, a binomial draw is equivalent to resampling the
                # answers with replacement and counting the votes for A.
                cell_draws = rng.binomial(len(votes), cell_share, size=n_draws) / len(votes)
                weighted_draws += weight * cell_draws

            lo, hi = np.quantile(weighted_draws, [0.025, 0.975])
            # In this fictional two-candidate setting, half the vote is the win threshold.
            # Split exact ties evenly so probabilities sum to one.
            win_a = float(np.mean(weighted_draws > 0.5) + 0.5 * np.mean(weighted_draws == 0.5))
            truth = float((cells["true_share_a"] * cells["population"]).sum() / total_population)
            summaries.append({
                "race_id": race, "model": model, "simulated": True,
                "valid_answers": sample["vote_a"].len() - sample["vote_a"].null_count(),
                "missing_answers": sample["vote_a"].null_count(),
                "raw_share_a": sample["vote_a"].mean(), "weighted_share_a": weighted_share,
                "weighted_share_b": 1 - weighted_share,
                "lower_95_a": float(lo), "upper_95_a": float(hi),
                "win_probability_a": win_a, "win_probability_b": 1 - win_a,
                "true_population_share_a": truth,
            })
            draws_tables.append(pl.DataFrame({
                "race_id": [race] * n_draws, "model": [model] * n_draws,
                "draw_id": np.arange(n_draws), "share_a": weighted_draws,
                "share_b": 1 - weighted_draws, "simulated": [True] * n_draws,
            }))

    return pl.DataFrame(summaries), pl.concat(draws_tables)


def main():
    source = ROOT / "data" / "00_simulated_data"
    if not (source / "frame.parquet").exists() or not (source / "responses.parquet").exists():
        raise SystemExit("First run: uv run python scripts/simulate_data.py")
    frame = pl.read_parquet(source / "frame.parquet")
    responses = pl.read_parquet(source / "responses.parquet")
    summary, draws = make_forecasts(frame, responses)
    output = ROOT / "outputs" / "simulated"
    output.mkdir(parents=True, exist_ok=True)
    summary.write_parquet(output / "forecasts.parquet")
    summary.write_csv(output / "forecasts.csv")
    draws.write_parquet(output / "forecast_draws.parquet")
    # A small readable table for terminal inspection; saved outputs retain full precision.
    display = summary.select("race_id", "model", "raw_share_a", "weighted_share_a",
                             "true_population_share_a", "lower_95_a", "upper_95_a")
    print("FICTIONAL DATA ONLY. Shares are proportions (0.55 means 55%).")
    print(display)
    print(f"Saved summaries and {draws.height:,} outcome draws to {output}")


if __name__ == "__main__":
    main()
