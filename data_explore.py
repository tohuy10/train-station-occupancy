"""Streamlit app for exploratory trip-duration analysis."""

# Import path utilities for folder scanning.
from pathlib import Path

# Import NumPy for fast histogram counting.
import numpy as np
# Import Pandas for CSV and datetime operations.
import pandas as pd
# Import Streamlit for the interactive app.
import streamlit as st
# Import Altair for charting.
import altair as alt


# Define the default folder containing trip CSV files.
DEFAULT_TRIPS_DIR = Path("inputs") / "inputs"


# Configure Streamlit page settings.
st.set_page_config(page_title="Trip duration explorer", layout="wide")


def _bucket_labels() -> list[str]:
    """Build duration labels with 30-minute splits for first 2 hours."""
    # Create finer buckets for first 2 hours, then hourly buckets onward.
    labels = ["0-0.5h", "0.5-1h", "1-1.5h", "1.5-2h"] + [f"{i}-{i + 1}h" for i in range(2, 24)]
    # Append an overflow bucket label.
    labels.append("24h+")
    # Return full label list.
    return labels


def _bucket_edges() -> np.ndarray:
    """Build numeric edges (hours) for all non-overflow buckets."""
    # 0, 0.5, 1, 1.5, 2, then 3..24.
    return np.array([0.0, 0.5, 1.0, 1.5] + [float(x) for x in range(2, 25)], dtype=np.float64)


@st.cache_data(show_spinner=False)
def compute_duration_buckets(csv_files: tuple[str, ...], chunksize: int = 200_000) -> dict[str, object]:
    """
    Compute trip counts by duration buckets and same-station subset.

    Buckets:
      - 0-1h, 1-2h, ..., 23-24h (left inclusive, right exclusive)
      - 24h+ (single overflow bucket)
    """
    # Build labels and base edges once.
    labels = _bucket_labels()
    # Build edges for non-overflow buckets.
    edges = _bucket_edges()
    # Initialize counters for all buckets including overflow.
    total_counts = np.zeros(len(labels), dtype=np.int64)
    # Initialize same-station counters for all buckets including overflow.
    same_station_counts = np.zeros(len(labels), dtype=np.int64)

    # Track total input rows scanned.
    total_rows_scanned = 0
    # Track valid trips: duration <= 1.5h and different start/end station.
    valid_trips_within_1_5h_non_same_station = 0
    # Track invalid trips as complement of the valid rule among usable durations.
    invalid_trips = 0

    # Iterate through selected CSV files.
    for csv_file in csv_files:
        # Read each file in chunks to keep memory usage stable.
        for chunk in pd.read_csv(
            csv_file,
            usecols=["start__station", "end__station", "start__time", "end__time"],
            chunksize=chunksize,
        ):
            # Update scanned row count.
            total_rows_scanned += int(len(chunk))

            # Parse start timestamps as UTC.
            start_dt = pd.to_datetime(chunk["start__time"], utc=True, errors="coerce")
            # Parse end timestamps as UTC.
            end_dt = pd.to_datetime(chunk["end__time"], utc=True, errors="coerce")

            # Compute duration in hours.
            duration_hours = (end_dt - start_dt).dt.total_seconds() / 3600.0

            # Keep finite and non-negative durations for robust bucketing.
            usable_mask = duration_hours.notna() & (duration_hours >= 0)
            # If no usable rows in this chunk, continue.
            if not usable_mask.any():
                continue

            # Filter durations to usable values as NumPy.
            usable_durations = duration_hours.loc[usable_mask].to_numpy(dtype=np.float64)
            # Build same-station mask for usable rows.
            same_station_mask = (
                chunk.loc[usable_mask, "start__station"].astype(str).to_numpy()
                == chunk.loc[usable_mask, "end__station"].astype(str).to_numpy()
            )
            # Count valid trips with duration <= 1.5h and non-same-station.
            valid_trips_within_1_5h_non_same_station += int(
                ((usable_durations <= 1.5) & (~same_station_mask)).sum()
            )
            # Count invalid trips (duration > 1.5h or same-station).
            invalid_trips += int(
                ((usable_durations > 1.5) | same_station_mask).sum()
            )

            # Use only durations below 24h for base buckets.
            base_durations = usable_durations[usable_durations < 24]
            # Count buckets for durations in [0, 24) using custom edges.
            base_counts, _ = np.histogram(base_durations, bins=edges)
            # Add to total base bucket counts.
            total_counts[:-1] += base_counts
            # Count overflow bucket (24h+).
            total_counts[-1] += int((usable_durations >= 24).sum())

            # Extract same-station durations only.
            same_durations = usable_durations[same_station_mask]
            # Use only same-station durations below 24h for base buckets.
            same_base_durations = same_durations[same_durations < 24]
            # Count same-station buckets for durations in [0, 24) using custom edges.
            same_base_counts, _ = np.histogram(same_base_durations, bins=edges)
            # Add to same-station base bucket counts.
            same_station_counts[:-1] += same_base_counts
            # Count same-station overflow bucket (24h+).
            same_station_counts[-1] += int((same_durations >= 24).sum())

    # Build output table for plotting and display.
    df = pd.DataFrame(
        {
            "duration_bucket": labels,
            "trip_count": total_counts,
            "same_station_trip_count": same_station_counts,
        }
    )

    # Compute same-station share percentage per bucket.
    df["same_station_pct"] = np.where(
        df["trip_count"] > 0,
        (df["same_station_trip_count"] / df["trip_count"]) * 100.0,
        0.0,
    )

    # Return both table and scan metadata.
    return {
        "table": df,
        "total_rows_scanned": int(total_rows_scanned),
        "valid_trips_within_1_5h_non_same_station": int(valid_trips_within_1_5h_non_same_station),
        "invalid_trips": int(invalid_trips),
    }


# Render app title.
st.title("Trip Duration Explorer")
# Render short app description.
st.caption(
    "Scans all trip CSV files and summarizes counts by duration buckets: "
    "`0-0.5h`, `0.5-1h`, `1-1.5h`, `1.5-2h`, ... , `23-24h`, and `24h+` "
    "(table shows all buckets; chart hides sub-1h buckets)."
)

# Discover all CSV files under the input folder.
default_csv_files = sorted(str(p) for p in DEFAULT_TRIPS_DIR.glob("*.csv"))

# # Sidebar: data source controls.
# st.sidebar.header("Data source")
# # Sidebar: show folder path.
# st.sidebar.write(f"Folder: `{DEFAULT_TRIPS_DIR.as_posix()}`")
# # Sidebar: allow file selection.
# selected_files = st.sidebar.multiselect(
#     "CSV files to include",
#     options=default_csv_files,
#     default=default_csv_files,
# )

# If no file is selected, stop with guidance.
# if not selected_files:
#     st.warning("Select at least one CSV file to run the analysis.")
#     st.stop()

# Run computation with fixed chunk size.
with st.spinner("Scanning files and computing duration buckets..."):
    #result = compute_duration_buckets(tuple(selected_files))
    result = compute_duration_buckets(tuple(default_csv_files))
# Extract result table.
summary_df = result["table"]
# Use all buckets in the table display.
display_df = summary_df.copy()
# Build a total row from all table buckets.
total_trip_count = int(summary_df["trip_count"].sum())
# Sum same-station counts across all table buckets.
total_same_station = int(summary_df["same_station_trip_count"].sum())
# Compute same-station percent for the total row.
total_same_station_pct = (total_same_station / total_trip_count * 100.0) if total_trip_count > 0 else 0.0
# Create a fixed one-row total table to keep it pinned at the bottom.
total_df = pd.DataFrame(
    [
        {
            "duration_bucket": "Total",
            "trip_count": total_trip_count,
            "same_station_trip_count": total_same_station,
            "same_station_pct": total_same_station_pct,
        }
    ]
)

# Show overall scan metrics.
metric_col1, metric_col2, metric_col3 = st.columns(3)
# Show scanned row count.
metric_col1.metric("Rows scanned", f"{result['total_rows_scanned']:,}")
# Show valid trips (within 1.5 hours, non-same-station) count.
metric_col2.metric(
    "Valid trips (<= 1.5h, non-same-station)",
    f"{result['valid_trips_within_1_5h_non_same_station']:,}",
)
# Show invalid trips count.
metric_col3.metric("Invalid trips", f"{result['invalid_trips']:,}")

# Show tabular summary section header.
st.subheader("Duration summary table")

# Build a styled table to highlight same-station counts.
styled = display_df.style.format(
    {
        "trip_count": "{:,.0f}",
        "same_station_trip_count": "{:,.0f}",
        "same_station_pct": "{:.2f}%",
    }
).background_gradient(subset=["same_station_trip_count"], cmap="Oranges")

# Render styled table.
st.dataframe(styled, width="stretch", hide_index=True)

# Show a fixed total row below the sortable table.
total_styled = total_df.style.format(
    {
        "trip_count": "{:,.0f}",
        "same_station_trip_count": "{:,.0f}",
        "same_station_pct": "{:.2f}%",
    }
).hide(axis="columns")
st.dataframe(total_styled, width="stretch", hide_index=True)

# Show chart section header.
st.subheader("Trip counts by duration bucket (>1 hour)")

# Convert table to long format for grouped bar chart.
chart_df = display_df.loc[~display_df["duration_bucket"].isin(["0-0.5h", "0.5-1h"])].melt(
    id_vars=["duration_bucket"],
    value_vars=["trip_count", "same_station_trip_count"],
    var_name="series",
    value_name="count",
)

# Build grouped bar chart.
bar_chart = (
    alt.Chart(chart_df)
    .mark_bar()
    .encode(
        x=alt.X(
            "duration_bucket:N",
            title="Duration bucket",
            sort=[x for x in _bucket_labels() if x not in ["0-0.5h", "0.5-1h"]],
        ),
        y=alt.Y("count:Q", title="Number of trips"),
        color=alt.Color(
            "series:N",
            title="Series",
            scale=alt.Scale(
                domain=["trip_count", "same_station_trip_count"],
                range=["#4C78A8", "#F58518"],
            ),
        ),
        xOffset="series:N",
        tooltip=[
            alt.Tooltip("duration_bucket:N", title="Bucket"),
            alt.Tooltip("series:N", title="Series"),
            alt.Tooltip("count:Q", title="Trips", format=",.0f"),
        ],
    )
    .properties(height=420)
)

# Render chart.
st.altair_chart(bar_chart, width="stretch")
