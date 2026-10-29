"""Create fictional election inputs. Run before forecast.py. No API calls."""

from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]


def simulate_data(seed=2453):
    """Return a population frame and deliberately unrepresentative responses."""
    rng = np.random.default_rng(seed)
    frame_rows = []
    response_rows = []

    # These are fictional two-candidate races, not actual Senate predictions.
    races = {"example_1": -0.45, "example_2": 0.0, "example_3": 0.45}
    ages = {"18-34": 0.6, "35-64": 0.0, "65+": -0.5}
    populations = {"18-34": 25000, "35-64": 50000, "65+": 25000}

    for race, race_effect in races.items():
        for age, age_effect in ages.items():
            for education in ["no_degree", "degree"]:
                degree = education == "degree"
                cell_id = f"{race}_{age}_{education}"
                # Age and education are related; the youngest group has fewer degrees.
                degree_share = {"18-34": 0.25, "35-64": 0.45, "65+": 0.35}[age]
                population = int(populations[age] * (degree_share if degree else 1 - degree_share))

                # Candidate A support depends on race, age, education, and their interaction.
                log_odds = race_effect + age_effect + 0.7 * degree
                log_odds += 0.3 * (age == "18-34") * degree
                true_share = float(1 / (1 + np.exp(-log_odds)))
                frame_rows.append({
                    "race_id": race, "cell_id": cell_id, "age": age,
                    "education": education, "population": population,
                    "true_share_a": true_share, "simulated": True,
                })

                # Oversample young people and degree holders on purpose.
                sample_size = (300 if age == "18-34" else 100) * (3 if degree else 1)
                for model, model_bias in [("example_model_1", 0.0), ("example_model_2", -0.35)]:
                    probability = float(1 / (1 + np.exp(-(log_odds + model_bias))))
                    votes = rng.binomial(1, probability, size=sample_size)
                    missing = rng.random(sample_size) < 0.02
                    for i in range(sample_size):
                        response_rows.append({
                            "response_id": len(response_rows), "race_id": race,
                            "cell_id": cell_id, "model": model,
                            "vote_a": None if missing[i] else int(votes[i]),
                            "simulated": True,
                        })

    return pl.DataFrame(frame_rows), pl.DataFrame(response_rows)


def main():
    output = ROOT / "data" / "00_simulated_data"
    output.mkdir(parents=True, exist_ok=True)
    frame, responses = simulate_data()
    frame.write_parquet(output / "frame.parquet")
    responses.write_parquet(output / "responses.parquet")
    print(f"Saved {frame.height} population cells and {responses.height:,} fictional responses.")
    print(f"Missing answers: {responses['vote_a'].null_count()}")
    print(f"Files: {output}")


if __name__ == "__main__":
    main()
