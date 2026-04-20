"""
Streamlit dashboard: passenger occupancy (counts) between stations over time.

Run:
  streamlit run eda.py
"""

# Import the standard library JSON parser.
import json
# Import monotonic timer for progress logging.
import time
# Import the standard library path utilities.
from pathlib import Path

# Import NumPy for fast numeric arrays.
import numpy as np
# Import Pandas for CSV loading and timestamp parsing.
import pandas as pd
# Import Streamlit for the interactive UI.
import streamlit as st
# Import Altair for plotting (heatmap + line charts).
import altair as alt


# Define the IANA timezone for HCMC (Vietnam) to convert '...Z' timestamps into local time-of-day.
DEFAULT_TZ = "Asia/Ho_Chi_Minh"

# Define the default folder that contains the trip CSV files.
DEFAULT_TRIPS_DIR = Path("inputs") / "inputs"

# Define the default timetable file path.
DEFAULT_TIMETABLE_PATH = Path("time_table.json")

# Define the default station code->name mapping file path.
DEFAULT_TERMINAL_NAMES_PATH = Path("terminal_name.txt")


# Build the Streamlit page configuration early (before other Streamlit calls).
st.set_page_config(page_title="Metro route passenger occupancy", layout="wide")


def _parse_hhmmss_to_seconds(hhmmss: str) -> int:
    """Convert a 'HH:MM:SS' string to seconds."""
    # Split the string by ':' to get hours, minutes, seconds.
    hh, mm, ss = hhmmss.split(":")
    # Convert each part to int and compute total seconds.
    return int(hh) * 3600 + int(mm) * 60 + int(ss)


def _parse_hhmm_to_seconds(hhmm: str) -> int:
    """Convert a 'HH:MM' string to seconds (supports '24:00')."""
    # Handle end-of-day marker explicitly.
    if hhmm == "24:00":
        return 24 * 3600
    # Split the string by ':' to get hours and minutes.
    hh, mm = hhmm.split(":")
    # Convert each part to int and compute total seconds.
    return int(hh) * 3600 + int(mm) * 60


def _load_terminal_names(path: Path) -> tuple[list[str], dict[str, str]]:
    """Load station stop order + mapping from `terminal_name.txt`."""
    # Read the full file as text.
    text = path.read_text(encoding="utf-8")
    # Initialize the ordered stop list.
    stop_order: list[str] = []
    # Initialize the code->human name mapping.
    code_to_name: dict[str, str] = {}
    # Iterate each line in the file.
    for raw_line in text.splitlines():
        # Strip whitespace.
        line = raw_line.strip()
        # Skip empty lines.
        if not line:
            continue
        # Split "CODE: Name" into its parts.
        code, name = line.split(":", 1)
        # Normalize the code.
        code = code.strip()
        # Normalize the name.
        name = name.strip()
        # Append to the ordered stop list.
        stop_order.append(code)
        # Store in the mapping.
        code_to_name[code] = name
    # Return both the order and mapping.
    return stop_order, code_to_name


def _load_timetable(path: Path) -> dict[tuple[int, int], list[int]]:
    """
    Load `time_table.json` and return per-(day_of_week,varId) segment travel times in seconds.

    Output:
      - key: (dow, varId) where dow in 0..6, varId in {1,2}
      - value: list[int] of length (n_stops - 1): seconds to next stop for that direction
    """
    # Read the JSON text from disk.
    raw = path.read_text(encoding="utf-8")
    # Parse JSON into Python objects.
    items = json.loads(raw)
    # Prepare mapping for (dow,varId) -> list of segment seconds.
    out: dict[tuple[int, int], list[int]] = {}
    # Iterate over each timetable entry.
    for entry in items:
        # Get the direction id (1 forward, 2 reverse).
        var_id = int(entry["varId"])
        # Convert each "HH:MM:SS" segment duration to seconds.
        seg_seconds = [_parse_hhmmss_to_seconds(x) for x in entry["time_to_next_stop"]]
        # Map this entry to all days listed (0=Mon, ..., 6=Sun).
        for dow in entry["days"]:
            # Store; if duplicates exist, last one wins.
            out[(int(dow), var_id)] = seg_seconds
    # Return the mapping.
    return out


def _load_departures(path: Path) -> dict[tuple[int, int], list[int]]:
    """
    Load `time_table.json` and return per-(day_of_week,varId) departure times from the first stop (in seconds).

    Output:
      - key: (dow, varId) where dow in 0..6, varId in {1,2}
      - value: sorted list[int] of seconds since midnight when a vehicle departs the first stop
    """
    # Read the JSON text from disk.
    raw = path.read_text(encoding="utf-8")
    # Parse JSON into Python objects.
    items = json.loads(raw)
    # Prepare mapping for (dow,varId) -> list of departure seconds.
    out: dict[tuple[int, int], list[int]] = {}
    # Iterate over each timetable entry.
    for entry in items:
        # Get the direction id (1 forward, 2 reverse).
        var_id = int(entry["varId"])
        # Convert each "HH:MM:SS" departure time to seconds since midnight.
        dep_seconds = [_parse_hhmmss_to_seconds(x) for x in entry.get("StartTime", [])]
        # Map this entry to all days listed (0=Mon, ..., 6=Sun).
        for dow in entry["days"]:
            # Initialize the list if needed.
            out.setdefault((int(dow), var_id), [])
            # Extend with this entry's departures.
            out[(int(dow), var_id)].extend(dep_seconds)
    # Sort and de-duplicate departures for each day+direction.
    for k in list(out.keys()):
        # Sort unique seconds.
        out[k] = sorted(set(out[k]))
    # Return the mapping.
    return out


def _precompute_paths(
    stop_order: list[str],
    timetable_seg_seconds: dict[tuple[int, int], list[int]],
) -> dict[tuple[int, int, int], tuple[list[int], np.ndarray, np.ndarray]]:
    """
    Precompute segment paths + segment time fractions for every (dow, start_idx, end_idx).

    Output mapping:
      key: (dow, start_idx, end_idx)
      value: (segment_indices, start_fracs, end_fracs)
        - segment_indices: list[int] positions in the direction-specific adjacency list
        - start_fracs/end_fracs: arrays of floats in [0,1] giving per-segment time window within trip duration
    """
    # Get number of stations (stops) on the line.
    n_stops = len(stop_order)
    # Initialize the output dictionary.
    out: dict[tuple[int, int, int], tuple[list[int], np.ndarray, np.ndarray]] = {}
    # Iterate all days of week.
    for dow in range(7):
        # Iterate all start indices.
        for s in range(n_stops):
            # Iterate all end indices.
            for e in range(n_stops):
                # Skip same-stop trips.
                if s == e:
                    continue
                # Determine direction from index ordering.
                var_id = 1 if s < e else 2
                # Get segment travel times (seconds) for this day+direction.
                seg_seconds_all = timetable_seg_seconds.get((dow, var_id))
                # If timetable missing, skip precompute for this combo.
                if seg_seconds_all is None:
                    continue
                # For direction 1, segment indices are 0..(n_stops-2) in forward order.
                if var_id == 1:
                    # Segment indices used by this trip are s..(e-1).
                    seg_idxs = list(range(s, e))
                else:
                    # For direction 2, build adjacency in reverse order (stop_order reversed).
                    # Segment index k (0-based) corresponds to moving from rev[k] -> rev[k+1].
                    # If s>e in forward indexing, then in reverse indexing those positions are:
                    #   s_rev = (n_stops - 1 - s), e_rev = (n_stops - 1 - e)
                    s_rev = (n_stops - 1 - s)
                    e_rev = (n_stops - 1 - e)
                    # Segment indices used by this trip are s_rev..(e_rev-1) in reverse adjacency.
                    seg_idxs = list(range(s_rev, e_rev))
                # Pull only the segment seconds on the path.
                path_seconds = np.asarray([seg_seconds_all[i] for i in seg_idxs], dtype=np.float64)
                # Compute total scheduled seconds across the path.
                total = float(path_seconds.sum())
                # If total is zero (should not happen), skip.
                if total <= 0:
                    continue
                # Compute cumulative fractions at segment boundaries.
                cum = np.cumsum(path_seconds) / total
                # Start fraction for each segment is the previous cumulative boundary.
                start_fracs = np.concatenate(([0.0], cum[:-1]))
                # End fraction for each segment is the cumulative boundary after that segment.
                end_fracs = cum
                # Store the precomputed path.
                out[(dow, s, e)] = (seg_idxs, start_fracs, end_fracs)
    # Return the precomputed mapping.
    return out


def _format_segment_label(from_code: str, to_code: str, code_to_name: dict[str, str]) -> str:
    """Create a readable segment label, including station names if available."""
    # Look up human names (fallback to code).
    from_name = code_to_name.get(from_code, from_code)
    # Look up human names (fallback to code).
    to_name = code_to_name.get(to_code, to_code)
    # Return a combined label.
    return f"{from_code} → {to_code}  ({from_name} → {to_name})"


@st.cache_data(show_spinner=False)
def compute_occupancy_all_days(
    trip_files: tuple[str, ...],
    bin_minutes: int,
    tz_name: str,
    stop_order: tuple[str, ...],
    timetable_seg_seconds_items: tuple[tuple[tuple[int, int], tuple[int, ...]], ...],
) -> dict[str, object]:
    """
    Compute passenger counts between stations and at stations for all days (0..6).

    Returns a dict with:
      - "bin_minutes": int
      - "bin_starts": np.ndarray of seconds since midnight (bin start times)
      - "segments_fwd": list[(from,to)] in forward direction order
      - "segments_rev": list[(from,to)] in reverse direction order
      - "occ": np.ndarray shape (7, 2, n_segments, n_bins)  (direction axis: 0=fwd(var1), 1=rev(var2))
      - "station_occ": np.ndarray shape (7, 2, n_stations, n_bins) (people at stations)
      - "stations_fwd": list[str] station codes in forward order
      - "stations_rev": list[str] station codes in reverse order
    """
    # Convert the immutable inputs back to mutable structures.
    stop_order_list = list(stop_order)
    # Build code->index mapping for fast lookup.
    code_to_idx = {code: i for i, code in enumerate(stop_order_list)}
    # Rebuild timetable mapping from cache-safe tuples.
    timetable_seg_seconds: dict[tuple[int, int], list[int]] = {
        k: list(v) for (k, v) in timetable_seg_seconds_items
    }
    # Precompute paths for all (dow, start_idx, end_idx).
    paths = _precompute_paths(stop_order_list, timetable_seg_seconds)
    # Compute number of bins in a full day.
    bin_sec = int(bin_minutes) * 60
    # Build bin start times for 00:00..24:00.
    bin_starts = np.arange(0, 24 * 3600, bin_sec, dtype=np.int32)
    # Compute bin count.
    n_bins = int(len(bin_starts))
    # Build forward adjacency segments.
    segments_fwd = [(stop_order_list[i], stop_order_list[i + 1]) for i in range(len(stop_order_list) - 1)]
    # Build reverse adjacency segments (reverse stop order).
    stop_order_rev = list(reversed(stop_order_list))
    # Build reverse adjacency segments.
    segments_rev = [(stop_order_rev[i], stop_order_rev[i + 1]) for i in range(len(stop_order_rev) - 1)]
    # Number of segments between adjacent stops.
    n_seg = len(segments_fwd)
    # Number of stations.
    n_sta = len(stop_order_list)
    # Allocate occupancy array: (dow, direction, segment, bin).
    occ = np.zeros((7, 2, n_seg, n_bins), dtype=np.float32)
    # Allocate at-station occupancy array: (dow, direction, station, bin).
    station_occ = np.zeros((7, 2, n_sta, n_bins), dtype=np.float32)
    # Track total rows read for progress logs.
    total_rows_read = 0
    # Track total valid rows used for computation.
    total_rows_valid = 0
    # Report progress every 200k input rows.
    report_every = 200_000
    # Track next report threshold.
    next_report = report_every
    # Start timer for elapsed seconds.
    started_at = time.perf_counter()

    def _add_interval_counts_1d(target: np.ndarray, a: float, b: float) -> None:
        """Add +1 to every bin overlapped by [a,b), clipped to a day."""
        # Skip non-overlapping full-day range quickly.
        if b <= 0.0 or a >= 24.0 * 3600.0:
            return
        # Clip to valid day bounds.
        a = max(0.0, a)
        b = min(24.0 * 3600.0, b)
        if b <= a:
            return
        # Compute overlapped bin range.
        i0 = int(a // bin_sec)
        i1 = int((b - 1e-9) // bin_sec)
        # Clamp indices to valid array bounds.
        i0 = max(0, min(i0, n_bins - 1))
        i1 = max(0, min(i1, n_bins - 1))
        # Add +1 for all overlapped bins.
        for bi in range(i0, i1 + 1):
            target[bi] += 1.0

    # Iterate each CSV file path.
    for file_path in trip_files:
        # Track per-file rows for progress logs.
        file_rows_read = 0
        # Log file start.
        print(f"[precompute] starting file: {file_path}", flush=True)
        # Read the CSV in chunks to control memory.
        for chunk in pd.read_csv(
            file_path,
            usecols=["start__station", "end__station", "start__time", "end__time"],
            chunksize=200_000,
        ):
            # Count input rows in this chunk.
            chunk_rows = int(len(chunk))
            # Update row counters.
            total_rows_read += chunk_rows
            # Update per-file row counter.
            file_rows_read += chunk_rows


            # Parse timestamps as UTC first.
            start_dt = pd.to_datetime(chunk["start__time"], utc=True, errors="coerce")
            # Parse timestamps as UTC first.
            end_dt = pd.to_datetime(chunk["end__time"], utc=True, errors="coerce")
            # Convert to local timezone for day-of-week and time-of-day binning.
            start_dt = start_dt.dt.tz_convert(tz_name)
            # Convert to local timezone for day-of-week and time-of-day binning.
            end_dt = end_dt.dt.tz_convert(tz_name)

            # Compute trip duration in seconds.
            duration = (end_dt - start_dt).dt.total_seconds()

            start_s = chunk["start__station"].astype(str)
            end_s   = chunk["end__station"].astype(str)
            # for i,di in enumerate(duration):
            #     if di >= 90*60:
                    #print("ABNORMAL DURATION ", start_s[i], " ", end_s[i],start_dt.iloc[i], " ", end_dt.iloc[i])
            # Drop rows with missing timestamps or non-positive durations.
            #valid_mask = duration.notna() & (duration > 0)
            
            valid_mask = (duration <= 90*60) & (start_s != end_s)
            # Apply mask to stations and timestamps.
            #print(chunk[~valid_mask])
            chunk = chunk.loc[valid_mask].copy()
            # Apply mask to start time series.
            start_dt = start_dt.loc[valid_mask]
            # Apply mask to end time series.
            end_dt = end_dt.loc[valid_mask]
            # Apply mask to duration.
            duration = duration.loc[valid_mask]
            # Update valid-row counter.
            total_rows_valid += int(len(chunk))

            # Print progress per 100k rows read.
            while total_rows_read >= next_report:
                # Compute elapsed seconds.
                elapsed = time.perf_counter() - started_at
                # Emit a compact progress message.
                print(
                    f"[precompute] checkpoint: {next_report:,} rows | "
                    f"actual rows read: {total_rows_read:,} | "
                    f"actual valid rows: {total_rows_valid:,} | "
                    f"current file rows: {file_rows_read:,} | elapsed: {elapsed:.1f}s",
                    flush=True,
                )
                # Move threshold to next 200k.
                next_report += report_every

            # Compute day-of-week (Monday=0..Sunday=6) based on local start time.
            dow = start_dt.dt.dayofweek.astype(np.int8)
            # Compute seconds since midnight for local start time.

            start_sec = (
                start_dt.dt.hour*3600
                + start_dt.dt.minute*60
                + start_dt.dt.second
            ).astype(np.int32)
            # Compute seconds since midnight for local end time.
            end_sec = (
                end_dt.dt.hour * 3600
                + end_dt.dt.minute * 60
                + end_dt.dt.second
            ).astype(np.int32)

            elapsed = time.perf_counter() - started_at
            

            # Iterate rows efficiently using itertuples.
            for s_code, e_code, d, ssec, esec, dsec in zip(
                chunk["start__station"].astype(str).to_numpy(),
                chunk["end__station"].astype(str).to_numpy(),
                dow.to_numpy(),
                start_sec.to_numpy(),
                end_sec.to_numpy(),
                duration.to_numpy(),
            ):
                # Skip unknown station codes.
                # if s_code not in code_to_idx or e_code not in code_to_idx:
                #     continue
                # Convert station codes to indices along the line.
                s_idx = code_to_idx[s_code]
                # Convert station codes to indices along the line.
                e_idx = code_to_idx[e_code]
                # Skip same-stop trips.
                # if s_idx == e_idx:
                #     continue
                # Determine direction from index ordering.
                if s_idx < e_idx:
                    dir_axis = 0  # forward (varId=1)
                else:
                    dir_axis = 1  # reverse (varId=2)

                # Compute station indices in the chosen direction axis.
                if dir_axis == 0:
                    s_sta_idx = int(s_idx)
                    e_sta_idx = int(e_idx)
                else:
                    s_sta_idx = int((n_sta - 1) - s_idx)
                    e_sta_idx = int((n_sta - 1) - e_idx)

                # Add at-station occupancy windows:
                # - 3 minutes after tap-in at origin station.
                _add_interval_counts_1d(
                    station_occ[int(d), dir_axis, s_sta_idx, :],
                    float(ssec),
                    float(ssec) + 180.0,
                )
                # - 3 minutes before tap-off at destination station.
                _add_interval_counts_1d(
                    station_occ[int(d), dir_axis, e_sta_idx, :],
                    float(esec) - 180.0,
                    float(esec),
                )
                # Get the precomputed path for this trip.
                key = (int(d), int(s_idx), int(e_idx))
                # Skip if path missing (should be rare).
                # if key not in paths:
                #     continue
                # Unpack the precomputed path data.
                seg_idxs, start_fracs, end_fracs = paths[key]
                
                # For each segment in the trip path, add this passenger count to overlapped bin(s).
                for local_seg_pos, seg_idx in enumerate(seg_idxs):
                    # Compute the segment start second-of-day using proportional allocation.
                    a = float(ssec) + float(dsec) * float(start_fracs[local_seg_pos])
                    # Compute the segment end second-of-day using proportional allocation.
                    b = float(ssec) + float(dsec) * float(end_fracs[local_seg_pos])
                    # Clip to the 0..86400 range.
                    # if b <= 0 or a >= 24 * 3600:
                    #     continue
                    # Clip the interval to a valid day range.
                    a = max(0.0, a)
                    # # Clip the interval to a valid day range.
                    if (a>=float(23*3600) and a<float(24*3600)) and (b>float(24 * 3600)):
                        b = float(24 * 3600)
                    # if b>=float(24 * 3600 - 15 * 60):
                    #     print("OVERFLOW", a," ",b, " ", ssec, " ", dsec)
                    #b = min(float(24 * 3600), b)
                    # # If interval is empty after clipping, skip.
                    # if b <= a:
                    #     continue
                    # Compute the first bin index overlapped by [a,b).
                    i0 = int(a // bin_sec)
                    # Compute the last bin index overlapped by [a,b).
                    i1 = int((b - 1e-9) // bin_sec)
                    # If the segment stays within one bin, add one passenger count.
                    if i0 == i1:
                        # Count this passenger once in that bin.
                        occ[int(d), dir_axis, int(seg_idx), i0] += 1.0
                    else:
                        # Count this passenger once in every overlapped bin.
                        for bi in range(i0, i1 + 1):
                            occ[int(d), dir_axis, int(seg_idx), bi] += 1.0
                
                
        

        # Log end-of-file summary.
        print(f"[precompute] finished file: {file_path} | rows read: {file_rows_read:,}", flush=True)

    # Log final summary after all files.
    total_elapsed = time.perf_counter() - started_at
    print(
        f"[precompute] done | total rows read: {total_rows_read:,} | "
        f"total valid rows: {total_rows_valid:,} | elapsed: {total_elapsed:.1f}s",
        flush=True,
    )

    # Return everything needed for visualization.
    return {
        "bin_minutes": int(bin_minutes),
        "bin_starts": bin_starts,
        "segments_fwd": segments_fwd,
        "segments_rev": segments_rev,
        "stations_fwd": stop_order_list,
        "stations_rev": stop_order_rev,
        "occ": occ,
        "station_occ": station_occ,
    }


# Render the dashboard title.
st.title("Passenger occupancy between stations (time-binned)")

# Show a short description for context.
st.caption(
    "This dashboard estimates **passenger counts on each between-station segment** per time bin, "
    "grouped by **day-of-week**. Times are interpreted in local timezone."
)

# Load station stop order + human names.
stop_order, code_to_name = _load_terminal_names(DEFAULT_TERMINAL_NAMES_PATH)

# Load timetable segment seconds per day + direction.
timetable_seg_seconds = _load_timetable(DEFAULT_TIMETABLE_PATH)

# Load departure times per day + direction.
departures_by_day_var = _load_departures(DEFAULT_TIMETABLE_PATH)

# Validate that timetable segment counts match the number of adjacent segments.
expected_segments = len(stop_order) - 1
# Compute a set of lengths present in timetable for quick validation.
lengths_present = {len(v) for v in timetable_seg_seconds.values()}
# Warn if timetable doesn't match stop count.
if expected_segments not in lengths_present:
    st.warning(
        f"Timetable segment count mismatch: expected {expected_segments} segments (from `terminal_name.txt`), "
        f"but found timetable segment lengths {sorted(lengths_present)} in `time_table.json`."
    )

# Find trip CSV files automatically (include all CSV files in the folder).
default_trip_files = sorted(str(p) for p in DEFAULT_TRIPS_DIR.glob("*.csv"))

# Sidebar: file selection.
# st.sidebar.header("Data sources")
# # Sidebar: show trips directory.
# st.sidebar.write(f"Trips folder: `{DEFAULT_TRIPS_DIR.as_posix()}`")
# # Sidebar: trip file multi-select.
# trip_files = st.sidebar.multiselect(
#     "Trip CSV files to include",
#     options=default_trip_files,
#     default=default_trip_files,
#     disabled=True
# )

# Sidebar: analysis settings.
st.sidebar.header("Analysis settings")
# Sidebar: bin size selection.
bin_minutes = st.sidebar.radio("Bin size (minutes)", options=[15, 30], index=0, horizontal=True)

# Main: time-range selection.
st.subheader("Filters")
# Build clock-time options in 15-minute steps, plus an explicit 24:00 end marker.
time_options = [f"{m // 60:02d}:{m % 60:02d}" for m in range(0, 24 * 60, 15)] + ["24:00"]
# Create a time range slider in HH:MM clock format.
start_label, end_label = st.select_slider(
    "Displayed time range (local time-of-day)",
    options=time_options,
    value=("05:00", "24:00"),
)

# Convert selected HH:MM values to seconds for slicing.
display_start_sec = _parse_hhmm_to_seconds(start_label)
# Convert selected HH:MM values to seconds for slicing.
display_end_sec = _parse_hhmm_to_seconds(end_label)

# If the user selected an invalid range, show an error.
if display_end_sec <= display_start_sec:
    st.error("Please choose an end time that is after the start time.")
    st.stop()

# If no files selected, stop.
# if not trip_files:
#     st.error("Please select at least one trip CSV file in the sidebar.")
#     st.stop()

# Build a cache-safe tuple of trip files.
#trip_files_key = tuple(trip_files)
trip_files_key = tuple(default_trip_files)
# Build a cache-safe timetable representation.
timetable_items_key = tuple(sorted(((k[0], k[1]), tuple(v)) for k, v in timetable_seg_seconds.items()))
# Build a cache-safe departures representation.
departures_items_key = tuple(sorted(((k[0], k[1]), tuple(v)) for k, v in departures_by_day_var.items()))

# Compute occupancy for all days (cached).
with st.spinner("Computing occupancy (cached)… this can take a while the first time."):
    results = compute_occupancy_all_days(
        trip_files=trip_files_key,
        bin_minutes=int(bin_minutes),
        tz_name=str(DEFAULT_TZ),
        stop_order=tuple(stop_order),
        timetable_seg_seconds_items=timetable_items_key,
    )

# Extract outputs.
bin_starts = results["bin_starts"]
# Extract outputs.
segments_fwd = results["segments_fwd"]
# Extract outputs.
segments_rev = results["segments_rev"]
# Extract outputs.
stations_fwd = results["stations_fwd"]
# Extract outputs.
stations_rev = results["stations_rev"]
# Extract outputs.
occ = results["occ"]
# Extract outputs.
station_occ = results["station_occ"]

# Compute bin seconds.
bin_sec = int(bin_minutes) * 60

# Compute display bin indices.
disp_i0 = int(display_start_sec // bin_sec)
# Compute display bin indices.
disp_i1 = int((display_end_sec - 1) // bin_sec) + 1

# Build time labels for displayed bins.
time_labels = []
# Build time labels for displayed bins.
for s in bin_starts[disp_i0:disp_i1]:
    # Convert seconds to HH:MM.
    hh = int(s) // 3600
    # Convert seconds to HH:MM.
    mm = (int(s) % 3600) // 60
    # Append formatted label.
    time_labels.append(f"{hh:02d}:{mm:02d}")


def _make_heatmap_df(day: int, dir_axis: int) -> pd.DataFrame:
    """Create a long-form DataFrame for Altair heatmap for one day + direction."""
    # Pick direction-specific segments.
    segs = segments_fwd if dir_axis == 0 else segments_rev
    # Slice the occupancy for the day+direction and displayed time window.
    mat = occ[int(day), int(dir_axis), :, disp_i0:disp_i1]
    # Build readable segment labels.
    seg_labels = [
        _format_segment_label(a, b, code_to_name)
        for (a, b) in segs
    ]
    # Flatten into long format.
    rows = []
    # Iterate segments.
    for si, seg_label in enumerate(seg_labels):
        # Iterate displayed bins.
        for ti, t_label in enumerate(time_labels):
            # Append one record.
            rows.append(
                {
                    "segment": seg_label,
                    "segment_order": int(si),
                    "time": t_label,
                    "passenger_count": float(mat[si, ti]),
                }
            )
    # Return as DataFrame.
    return pd.DataFrame(rows)


def _make_in_transit_df(day: int, dir_axis: int) -> pd.DataFrame:
    """Create a DataFrame of trains in transit per displayed time bin for one day + direction."""
    # Map dir axis to varId.
    var_id = 1 if int(dir_axis) == 0 else 2
    # Get segment seconds and compute whole-journey runtime for this day+direction.
    seg_seconds = timetable_seg_seconds.get((int(day), int(var_id)), [])
    journey_sec = float(sum(seg_seconds))
    # Get departures (seconds since midnight) for this day+direction.
    deps = departures_by_day_var.get((int(day), int(var_id)), [])
    # Initialize in-transit train counts per displayed bin.
    counts = np.zeros(len(time_labels), dtype=np.int32)
    # If timetable is missing, return zeros.
    if journey_sec <= 0:
        return pd.DataFrame({"time": time_labels, "trains_in_transit": counts.astype(int)})
    # Count trains that overlap each displayed time bin.
    for dsec in deps:
        # Train active interval [a,b) from departure to end of one-way run.
        a = float(dsec)
        b = float(dsec) + journey_sec
        # Skip intervals outside display window.
        if b <= display_start_sec or a >= display_end_sec:
            continue
        # Clip to display window.
        a = max(a, float(display_start_sec))
        b = min(b, float(display_end_sec))
        if b <= a:
            continue
        # Find overlapped bin indices relative to display window.
        i0 = int((a - float(display_start_sec)) // float(bin_sec))
        i1 = int(((b - float(display_start_sec)) - 1e-9) // float(bin_sec))
        # Add this train to every overlapped bin.
        for bi in range(max(i0, 0), min(i1, len(counts) - 1) + 1):
            counts[bi] += 1
    # Build DataFrame.
    return pd.DataFrame({"time": time_labels, "trains_in_transit": counts.astype(int)})


def _make_avg_per_bus_df(df_hm: pd.DataFrame, df_transit: pd.DataFrame) -> pd.DataFrame:
    """Create per-segment per-time average passenger values."""
    # Join passenger counts with in-transit train counts by time.
    merged = df_hm.merge(df_transit, on="time", how="left")
    # Fill missing in-transit counts with zero before division safety handling.
    merged["trains_in_transit"] = merged["trains_in_transit"].fillna(0).astype(float)
    # Compute effective denominator for average:
    # - If trains > 0, use trains count.
    # - If trains == 0 and passengers > 0, assume 1 special train.
    # - If trains == 0 and passengers == 0, keep denominator at 1 so average becomes 0.
    merged["effective_trains_for_avg"] = np.where(
        merged["trains_in_transit"] > 0,
        merged["trains_in_transit"],
        1.0,
    )
    # Build display trains count for tooltip:
    # - real trains count when > 0
    # - 1 when passenger_count > 0 but trains count is 0 (special train assumption)
    # - 0 when both passenger_count and trains count are 0
    merged["trains_in_transit_display"] = np.where(
        merged["trains_in_transit"] > 0,
        merged["trains_in_transit"],
        np.where(merged["passenger_count"] > 0, 1.0, 0.0),
    )
    # Compute average passenger with the above fallback rule.
    merged["avg_passengers_per_bus"] = merged["passenger_count"] / merged["effective_trains_for_avg"]
    # Return plotting frame.
    return merged


def _make_station_heatmap_df(day: int, dir_axis: int) -> pd.DataFrame:
    """Create a long-form DataFrame for people-at-station heatmap."""
    # Pick direction-specific station order.
    stations = stations_fwd if int(dir_axis) == 0 else stations_rev
    # Slice station occupancy for day+direction and display window.
    mat = station_occ[int(day), int(dir_axis), :, disp_i0:disp_i1]
    # Build long-form rows.
    rows = []
    for si, sta_code in enumerate(stations):
        # Build full-name station label for axis readability.
        sta_name = code_to_name.get(sta_code, sta_code)
        sta_label = f"{sta_code} ({sta_name})"
        for ti, t_label in enumerate(time_labels):
            rows.append(
                {
                    "station_code": sta_code,
                    "station_name": sta_name,
                    "station_label": sta_label,
                    "station_order": int(si),
                    "time": t_label,
                    "station_people_count": float(mat[si, ti]),
                }
            )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def compute_weekly_summary_cached(
    occ: np.ndarray,
    bin_starts: np.ndarray,
    segments_fwd: tuple[tuple[str, str], ...],
    segments_rev: tuple[tuple[str, str], ...],
    code_to_name_items: tuple[tuple[str, str], ...],
    day_name_map: tuple[str, ...],
    departures_items: tuple[tuple[tuple[int, int], tuple[int, ...]], ...],
    timetable_items: tuple[tuple[tuple[int, int], tuple[int, ...]], ...],
) -> pd.DataFrame:
    """Build cached full-day weekly summary frame from bin-level cells (all days/directions)."""
    # Rebuild small mappings from cache-safe tuples.
    code_to_name_local = {k: v for k, v in code_to_name_items}
    departures_local = {k: list(v) for (k, v) in departures_items}
    timetable_local = {k: list(v) for (k, v) in timetable_items}

    # Compute bin size and use full-day window for cache stability.
    bin_sec = int(bin_starts[1] - bin_starts[0]) if len(bin_starts) > 1 else 900
    display_start_sec = 0
    display_end_sec = 24 * 3600

    # Build full-day time labels.
    time_labels_local = []
    for s in bin_starts:
        hh = int(s) // 3600
        mm = (int(s) % 3600) // 60
        time_labels_local.append(f"{hh:02d}:{mm:02d}")

    def _transit_counts(day_idx: int, dir_axis: int) -> np.ndarray:
        """Compute trains-in-transit counts for displayed bins."""
        var_id = 1 if int(dir_axis) == 0 else 2
        journey_sec = float(sum(timetable_local.get((int(day_idx), int(var_id)), [])))
        deps = departures_local.get((int(day_idx), int(var_id)), [])
        counts = np.zeros(len(time_labels_local), dtype=np.float64)
        if journey_sec <= 0:
            return counts
        for dsec in deps:
            a = float(dsec)
            b = float(dsec) + journey_sec
            if b <= display_start_sec or a >= display_end_sec:
                continue
            a = max(a, float(display_start_sec))
            b = min(b, float(display_end_sec))
            if b <= a:
                continue
            i0 = int((a - float(display_start_sec)) // float(bin_sec))
            i1 = int(((b - float(display_start_sec)) - 1e-9) // float(bin_sec))
            for bi in range(max(i0, 0), min(i1, len(counts) - 1) + 1):
                counts[bi] += 1.0
        return counts

    rows = []
    for d in range(7):
        for dir_axis in (0, 1):
            segs = segments_fwd if int(dir_axis) == 0 else segments_rev
            mat = occ[int(d), int(dir_axis), :, :]
            transit = _transit_counts(day_idx=int(d), dir_axis=int(dir_axis))
            effective = np.where(transit > 0, transit, 1.0)
            avg_mat = mat / effective[np.newaxis, :]
            for si, (a, b) in enumerate(segs):
                seg_label = _format_segment_label(a, b, code_to_name_local)
                for ti, t_label in enumerate(time_labels_local):
                    rows.append(
                        {
                            "day": day_name_map[int(d)],
                            "direction": "Forward" if int(dir_axis) == 0 else "Reverse",
                            "segment": seg_label,
                            "time": t_label,
                            "time_sec": int(bin_starts[ti]),
                            "passenger_count": float(mat[si, ti]),
                            "avg_passengers_per_bus": float(avg_mat[si, ti]),
                        }
                    )
    return pd.DataFrame(rows)


# Always show all day tabs (Mon..Sun).
days_selected_sorted = [0, 1, 2, 3, 4, 5, 6]
day_name_map = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Always show both directions as graph sub-tabs.
dir_axes: list[int] = [0, 1]

# Create tabs per selected day-of-week.
day_tabs = st.tabs([day_name_map[d] for d in days_selected_sorted])

# Render one heatmap per day (and per direction if needed).
for tab, d in zip(day_tabs, days_selected_sorted):
    # Switch into the tab.
    with tab:
        # Show which day is being displayed.
        st.subheader(f"Day: {day_name_map[d]}")
        # Create sub-tabs if both directions selected.
        if len(dir_axes) == 2:
            # Create direction sub-tabs.
            dir_tabs = st.tabs(["Forward (BTN → STT)", "Reverse (STT → BTN)"])
            # Render each direction heatmap.
            for dir_tab, dir_axis in zip(dir_tabs, dir_axes):
                # Switch into the direction tab.
                with dir_tab:
                    # Build and render trains-in-transit chart.
                    df_transit = _make_in_transit_df(day=int(d), dir_axis=int(dir_axis))
                    # Build in-transit bar chart.
                    dep_chart = (
                        alt.Chart(df_transit)
                        .mark_bar()
                        .encode(
                            x=alt.X("time:N", title="Time (local)", sort=time_labels),
                            y=alt.Y("trains_in_transit:Q", title="Trains in transit"),
                            tooltip=[alt.Tooltip("time:N"), alt.Tooltip("trains_in_transit:Q")],
                        )
                        .properties(height=140)
                    )
                    # Show the chart.
                    st.altair_chart(dep_chart, width="stretch")

                    # Build the heatmap dataframe.
                    df_hm = _make_heatmap_df(day=int(d), dir_axis=int(dir_axis))
                    # Build the heatmap chart.
                    chart = (
                        alt.Chart(df_hm)
                        .mark_rect()
                        .encode(
                            x=alt.X("time:N", title="Time (local)", sort=time_labels),
                            y=alt.Y(
                                "segment:N",
                                title="Between-station segment",
                                sort=alt.SortField(field="segment_order", order="ascending"),
                            ),
                            color=alt.Color("passenger_count:Q", title="Passenger count", scale=alt.Scale(scheme="viridis")),
                            tooltip=[
                                alt.Tooltip("segment:N"),
                                alt.Tooltip("time:N"),
                                alt.Tooltip("passenger_count:Q", format=".0f"),
                            ],
                        )
                        .properties(height=520, title="Between-station passenger count heatmap")
                    )
                    # Render the chart.
                    st.altair_chart(chart, width="stretch")

                    # Build average-passenger frame and chart.
                    df_avg = _make_avg_per_bus_df(df_hm=df_hm, df_transit=df_transit)
                    avg_chart = (
                        alt.Chart(df_avg)
                        .mark_rect()
                        .encode(
                            x=alt.X("time:N", title="Time (local)", sort=time_labels),
                            y=alt.Y(
                                "segment:N",
                                title="Between-station segment",
                                sort=alt.SortField(field="segment_order", order="ascending"),
                            ),
                            color=alt.Color(
                                "avg_passengers_per_bus:Q",
                                title="Avg passenger",
                                scale=alt.Scale(scheme="plasma"),
                            ),
                            tooltip=[
                                alt.Tooltip("segment:N"),
                                alt.Tooltip("time:N"),
                                alt.Tooltip("trains_in_transit_display:Q", title="trains_in_transit", format=".0f"),
                                alt.Tooltip("avg_passengers_per_bus:Q", title="avg_passenger", format=".2f"),
                            ],
                        )
                        .properties(height=520, title="Between-station average passenger heatmap")
                    )
                    st.altair_chart(avg_chart, width="stretch")

                    # Build and render at-station people heatmap.
                    df_station = _make_station_heatmap_df(day=int(d), dir_axis=int(dir_axis))
                    station_chart = (
                        alt.Chart(df_station)
                        .mark_rect()
                        .encode(
                            x=alt.X("time:N", title="Time (local)", sort=time_labels),
                            y=alt.Y(
                                "station_label:N",
                                title="Station",
                                sort=alt.SortField(field="station_order", order="ascending"),
                            ),
                            color=alt.Color(
                                "station_people_count:Q",
                                title="Passenger count",
                                scale=alt.Scale(scheme="magma"),
                            ),
                            tooltip=[
                                alt.Tooltip("station_code:N", title="station_code"),
                                alt.Tooltip("station_name:N", title="station_name"),
                                alt.Tooltip("time:N"),
                                alt.Tooltip("station_people_count:Q", format=".0f"),
                            ],
                        )
                        .properties(height=520, title="At-station people count heatmap")
                    )
                    st.altair_chart(station_chart, width="stretch")
        else:
            # Build and render trains-in-transit chart.
            df_transit = _make_in_transit_df(day=int(d), dir_axis=int(dir_axes[0]))
            # Build in-transit bar chart.
            dep_chart = (
                alt.Chart(df_transit)
                .mark_bar()
                .encode(
                    x=alt.X("time:N", title="Time (local)", sort=time_labels),
                    y=alt.Y("trains_in_transit:Q", title="Trains in transit"),
                    tooltip=[alt.Tooltip("time:N"), alt.Tooltip("trains_in_transit:Q")],
                )
                .properties(height=140)
            )
            # Show the chart.
            st.altair_chart(dep_chart, width="stretch")

            # Build the heatmap dataframe.
            df_hm = _make_heatmap_df(day=int(d), dir_axis=int(dir_axes[0]))
            # Build the heatmap chart.
            chart = (
                alt.Chart(df_hm)
                .mark_rect()
                .encode(
                    x=alt.X("time:N", title="Time (local)", sort=time_labels),
                    y=alt.Y(
                        "segment:N",
                        title="Between-station segment",
                        sort=alt.SortField(field="segment_order", order="ascending"),
                    ),
                    color=alt.Color("passenger_count:Q", title="Passenger count", scale=alt.Scale(scheme="viridis")),
                    tooltip=[
                        alt.Tooltip("segment:N"),
                        alt.Tooltip("time:N"),
                        alt.Tooltip("passenger_count:Q", format=".0f"),
                    ],
                )
                .properties(height=520, title="Between-station passenger count heatmap")
            )
            # Render the chart.
            st.altair_chart(chart, width="stretch")

            # Build average-passenger frame and chart.
            df_avg = _make_avg_per_bus_df(df_hm=df_hm, df_transit=df_transit)
            avg_chart = (
                alt.Chart(df_avg)
                .mark_rect()
                .encode(
                    x=alt.X("time:N", title="Time (local)", sort=time_labels),
                    y=alt.Y(
                        "segment:N",
                        title="Between-station segment",
                        sort=alt.SortField(field="segment_order", order="ascending"),
                    ),
                    color=alt.Color(
                        "avg_passengers_per_bus:Q",
                        title="Avg passenger",
                        scale=alt.Scale(scheme="plasma"),
                    ),
                    tooltip=[
                        alt.Tooltip("segment:N"),
                        alt.Tooltip("time:N"),
                        alt.Tooltip("trains_in_transit_display:Q", title="trains_in_transit", format=".0f"),
                        alt.Tooltip("avg_passengers_per_bus:Q", title="avg_passenger", format=".2f"),
                    ],
                )
                .properties(height=520, title="Between-station average passenger heatmap")
            )
            st.altair_chart(avg_chart, width="stretch")

            # Build and render at-station people heatmap.
            df_station = _make_station_heatmap_df(day=int(d), dir_axis=int(dir_axes[0]))
            station_chart = (
                alt.Chart(df_station)
                .mark_rect()
                .encode(
                    x=alt.X("time:N", title="Time (local)", sort=time_labels),
                    y=alt.Y(
                        "station_label:N",
                        title="Station",
                        sort=alt.SortField(field="station_order", order="ascending"),
                    ),
                    color=alt.Color(
                        "station_people_count:Q",
                        title="People at station",
                        scale=alt.Scale(scheme="magma"),
                    ),
                    tooltip=[
                        alt.Tooltip("station_code:N", title="station_code"),
                        alt.Tooltip("station_name:N", title="station_name"),
                        alt.Tooltip("time:N"),
                        alt.Tooltip("station_people_count:Q", format=".0f"),
                    ],
                )
                .properties(height=520, title="At-station people count heatmap")
            )
            st.altair_chart(station_chart, width="stretch")

# Build weekly summary frame from cache.
weekly_df = compute_weekly_summary_cached(
    occ=occ,
    bin_starts=bin_starts,
    segments_fwd=tuple(segments_fwd),
    segments_rev=tuple(segments_rev),
    code_to_name_items=tuple(sorted(code_to_name.items())),
    day_name_map=tuple(day_name_map),
    departures_items=departures_items_key,
    timetable_items=timetable_items_key,
)
# Apply slider window after cache so range changes do not invalidate precompute.
# weekly_df = weekly_df.loc[
#     (weekly_df["time_sec"] >= int(display_start_sec))
#     & (weekly_df["time_sec"] < int(display_end_sec))
# ].copy()

# Render weekly summary charts from cached data.
# Build per-day max rows with associated segment/time/direction context.
max_count_rows = weekly_df.loc[
    weekly_df.groupby("day")["passenger_count"].idxmax(),
    ["day", "passenger_count", "time", "segment", "direction"],
].rename(
    columns={
        "passenger_count": "max_passenger_count",
        "time": "max_passenger_time",
        "segment": "max_passenger_segment",
        "direction": "max_passenger_direction",
    }
)
max_avg_rows = weekly_df.loc[
    weekly_df.groupby("day")["avg_passengers_per_bus"].idxmax(),
    ["day", "avg_passengers_per_bus", "time", "segment", "direction"],
].rename(
    columns={
        "avg_passengers_per_bus": "max_avg_passengers_per_bus",
        "time": "max_avg_time",
        "segment": "max_avg_segment",
        "direction": "max_avg_direction",
    }
)
daily_max_df = max_count_rows.merge(max_avg_rows, on="day", how="inner")
# Keep day order stable.
daily_max_df["day"] = pd.Categorical(daily_max_df["day"], categories=day_name_map, ordered=True)
daily_max_df = daily_max_df.sort_values("day")

st.subheader("Weekly summary from bins")
st.caption("Max values are derived from bin-level segment cells across both directions.")

# Plot max passenger count by day with labels.
max_count_base = alt.Chart(daily_max_df)
max_count_bars = (
    max_count_base
    .mark_bar(color="#4C78A8")
    .encode(
        x=alt.X("day:N", sort=day_name_map, title="Day of week", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("max_passenger_count:Q", title="Max passenger count"),
        tooltip=[
            alt.Tooltip("day:N"),
            alt.Tooltip("max_passenger_count:Q", format=".0f"),
            alt.Tooltip("max_passenger_time:N", title="timeframe"),
            alt.Tooltip("max_passenger_segment:N", title="station"),
            alt.Tooltip("max_passenger_direction:N", title="direction"),
        ],
    )
)
max_count_labels = (
    max_count_base
    .mark_text(dy=-3, color="#1F2A44")
    .encode(
        x=alt.X("day:N", sort=day_name_map, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("max_passenger_count:Q"),
        text=alt.Text("max_passenger_count:Q", format=".0f"),
    )
)
max_count_chart = (max_count_bars + max_count_labels).properties(
    height=250,
    padding={"top": 24, "left": 8, "right": 8, "bottom": 8},
    title=alt.TitleParams("Max passenger count by day", anchor="start", orient="top", offset=16),
)
st.altair_chart(max_count_chart, width="stretch")

# Plot max average passenger by day with labels.
max_avg_base = alt.Chart(daily_max_df)
max_avg_bars = (
    max_avg_base
    .mark_bar(color="#F58518")
    .encode(
        x=alt.X("day:N", sort=day_name_map, title="Day of week", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("max_avg_passengers_per_bus:Q", title="Max avg passenger"),
        tooltip=[
            alt.Tooltip("day:N"),
            alt.Tooltip("max_avg_passengers_per_bus:Q", format=".2f"),
            alt.Tooltip("max_avg_time:N", title="timeframe"),
            alt.Tooltip("max_avg_segment:N", title="station"),
            alt.Tooltip("max_avg_direction:N", title="direction"),
        ],
    )
)
max_avg_labels = (
    max_avg_base
    .mark_text(dy=-3, color="#5A2B00")
    .encode(
        x=alt.X("day:N", sort=day_name_map, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("max_avg_passengers_per_bus:Q"),
        text=alt.Text("max_avg_passengers_per_bus:Q", format=".2f"),
    )
)
max_avg_chart = (max_avg_bars + max_avg_labels).properties(
    height=250,
    padding={"top": 24, "left": 8, "right": 8, "bottom": 8},
    title=alt.TitleParams("Max average passenger by day", anchor="start", orient="top", offset=16),
)
st.altair_chart(max_avg_chart, width="stretch")

# Build max-by-time for each weekday.
max_time_df = (
    weekly_df.groupby(["day", "time"], as_index=False)
    .agg(
        max_passenger_count=("passenger_count", "max"),
        max_avg_passenger=("avg_passengers_per_bus", "max"),
    )
)
max_time_df["day"] = pd.Categorical(max_time_df["day"], categories=day_name_map, ordered=True)

st.subheader("Max trends by time")

# Max passenger count line chart by time.
max_count_time_chart = (
    alt.Chart(max_time_df)
    .mark_line(point=False)
    .encode(
        x=alt.X("time:N", sort=time_labels, title="Time (local)"),
        y=alt.Y("max_passenger_count:Q", title="Max passenger count"),
        color=alt.Color("day:N", title="Day", sort=day_name_map),
        tooltip=[
            alt.Tooltip("day:N"),
            alt.Tooltip("time:N"),
            alt.Tooltip("max_passenger_count:Q", format=".2f"),
        ],
    )
    .properties(height=280, title="Max passenger count by time")
)
st.altair_chart(max_count_time_chart, width="stretch")

# Max average passenger line chart by time.
max_avg_time_chart = (
    alt.Chart(max_time_df)
    .mark_line(point=False)
    .encode(
        x=alt.X("time:N", sort=time_labels, title="Time (local)"),
        y=alt.Y("max_avg_passenger:Q", title="Max avg passenger"),
        color=alt.Color("day:N", title="Day", sort=day_name_map),
        tooltip=[
            alt.Tooltip("day:N"),
            alt.Tooltip("time:N"),
            alt.Tooltip("max_avg_passenger:Q", format=".2f"),
        ],
    )
    .properties(height=280, title="Max average passenger by time")
)
st.altair_chart(max_avg_time_chart, width="stretch")

# Show a small note about interpretation.
# st.info(
#     "Interpretation: values are **passenger counts per segment bin**. "
#     "Each passenger contributes +1 to every time bin overlapped by the segment interval."
# )
