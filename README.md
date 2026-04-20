# Metro route analysis (Streamlit)

## What this does

`eda.py` is a Streamlit dashboard that visualizes **passenger occupancy between each adjacent station** over time, grouped by **day of week**.

It reads trip records from CSVs under `inputs/inputs/` (columns: `start__station`, `end__station`, `start__time`, `end__time`), and uses:

- `terminal_name.txt` for the **ordered station list** (and code→name labels)
- `time_table.json` for **between-station travel-time weights** per day-of-week and direction

Occupancy values are reported as **average passengers in each time bin** (15 or 30 minutes).

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run eda.py
```

## Data notes

- Trip timestamps in the CSV are parsed as UTC (the `Z` suffix) and then converted to local time for binning.
- By default, the app uses `Asia/Ho_Chi_Minh` as the local timezone.
