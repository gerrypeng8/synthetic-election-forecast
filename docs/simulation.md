# Running and understanding the simulation

This stage uses three fictional two-candidate races and two fictional models.
It makes no API calls and is not a forecast of any real election.

From the project root:

```sh
uv sync
uv run python scripts/simulate_data.py
uv run python scripts/forecast.py
uv run pytest -q
```

## Inputs

`data/00_simulated_data/frame.parquet` has one row per race, age group, and education
group. Each race has 100,000 fictional adults across six demographic cells.
`population` is a count; `true_share_a` is the known population probability of
supporting Candidate A. This truth is used for comparison only, not to estimate support.

`responses.parquet` has one row per fictional answer. `vote_a` is 1 for Candidate A,
0 for Candidate B, and missing for a failed answer. Young people and degree holders
are overrepresented. Their support differs from other groups, creating sampling bias.
The second model also has a systematic shift toward Candidate B.

The same default seed recreates identical inputs and outputs.

## Forecast calculation

For each race and model separately:

1. Remove missing answers within each cell, counting them in the summary.
2. Calculate the fraction choosing A within each cell.
3. Multiply each fraction by that cell's population share and add the results.

For example, if a cell contains 20% of the population and 60% of its valid answers
choose A, it contributes 0.20 * 0.60 = 0.12 to A's overall vote share.

For illustrative uncertainty, bootstrap the answers within each cell: sample the
observed answers with replacement, keeping the number of valid answers fixed.
For binary answers this is equivalent to drawing A's count from a binomial
distribution with that sample size and the observed share. Divide by the sample
size to obtain a share, then population-weight the cells. Repeat 2,000 times.
The 2.5th and 97.5th percentiles form a 95% percentile bootstrap interval.
The proportion above 0.5 is an illustrative measure of how often A wins across
bootstrap draws in this two-candidate example. Exact ties split evenly.
This bootstrap win frequency is not a calibrated real-election probability.

These draws assume independent answers and fixed population weights. They do not
capture systematic model bias, correlated answers, turnout, or changing preferences.
If every answer in a cell is identical, its bootstrap share never changes; this
method cannot infer unobserved variation in such a group.
They are not yet a defensible uncertainty model for a real election. More generated
answers can narrow these intervals without fixing bias. The two models are shown
separately; this stage does not choose an ensemble method.

## Inspect the outputs

Open `outputs/simulated/forecasts.csv` in VS Code or a spreadsheet application.
Shares are proportions: 0.55 means 55%, not 0.55%.

| Column | Meaning |
| --- | --- |
| race_id / model | Fictional race and response generator |
| simulated | Always true in this stage |
| valid_answers / missing_answers | Counts used or excluded |
| raw_share_a | A's share of valid answers before weighting |
| weighted_share_a / weighted_share_b | Population-weighted observed support |
| lower_95_a / upper_95_a | Conditional 95% percentile bootstrap interval for A |
| win_probability_a / win_probability_b | Fraction of modeled wins |
| true_population_share_a | Known simulated truth, used for diagnostics |

`forecasts.parquet` contains the same summary in Parquet format.
`forecast_draws.parquet` contains 2,000 draws per model per race, 12,000 rows total.
To inspect inputs and draws in the terminal:

```sh
uv run python -c 'import polars as pl; print(pl.read_parquet("data/00_simulated_data/frame.parquet"))'
uv run python -c 'import polars as pl; print(pl.read_parquet("data/00_simulated_data/responses.parquet").head(10))'
uv run python -c 'import polars as pl; print(pl.read_parquet("outputs/simulated/forecast_draws.parquet").head(10))'
```

Compare raw and weighted support against known truth for example_model_1: weighting
should reduce the deliberately introduced sampling bias. Model 2 demonstrates that
correct demographic weights cannot necessarily remove model bias.

Tests check repeatability, known weighted results, missing answers, race coverage,
probability bounds, reduced sampling bias, and rejection of broken inputs. Separate
tests for actual data and live API responses will be added when those stages exist.
