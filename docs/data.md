# Getting the raw data

`scripts/download_data.py` downloads the CES into `data/01_raw_data/` as a Parquet file. 
It needs no API key. 

From the project root:

```sh
uv sync
uv run python scripts/download_data.py
```

## Why the raw file is not committed

The file stays out of the repo due to size, and because the script can recreate it.
