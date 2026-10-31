"""Download the CES common content from 2025 into data/01_raw_data."""

from pathlib import Path

import polars as pl
from pyDataverse.api import DataAccessApi

OUTPUT_DIR = Path("data/01_raw_data")

# CES 2025 Common Content, doi:10.7910/DVN/DFUGJR
CES_FILE_ID = 13600926
# Respondent ID, survey weight, state, birth year, gender, education, race, Hispanic origin,
# party ID, ideology, and voter registration.

CES_COLUMNS = ["caseid", "commonweight", "inputstate", "birthyr", "gender4", "educ", "race",
               "hispanic", "pid7", "ideo5", "votereg"]


def main():
    api = DataAccessApi("https://dataverse.harvard.edu")
    response = api.get_datafile(CES_FILE_ID, is_pid=False)
    response.raise_for_status()

    ces = pl.read_csv(response.content, separator="\t", columns=CES_COLUMNS, infer_schema_length=0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ces.write_parquet(OUTPUT_DIR / "ces_common_2025.parquet")


if __name__ == "__main__":
    main()
