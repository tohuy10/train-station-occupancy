"""Calibrate segment travel times from trip tap-in/tap-out data.

This script estimates realistic between-station travel times by combining:
- Scheduled segment times (`time_to_next_stop`)
- Scheduled departures (`StartTime`) to estimate passenger wait time
- Observed tap-in/tap-out duration from trip CSV files

Output:
- Printed table of final estimated time between adjacent stations
- CSV file: `estimated_time_between_stations.csv`
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# Input locations.
TRIPS_DIR = Path("inputs") / "inputs"
TIMETABLE_PATH = Path("time_table.json")
TERMINAL_NAME_PATH = Path("terminal_name.txt")

# Timezone used by your current project.
LOCAL_TZ = "Asia/Ho_Chi_Minh"

# Training filters (seconds).
MIN_OBS_DURATION_SEC = 30.0
MAX_OBS_DURATION_SEC = 1.5 * 3600.0

# Ridge regularization strength for stable fitting.
RIDGE_LAMBDA = 1e-2

# Service-group definition aligned with timetable structure.
SERVICE_GROUPS = [
    ("Mon-Thu", [0, 1, 2, 3]),
    ("Fri", [4]),
    ("Sat-Sun", [5, 6]),
]


def parse_hhmmss_to_seconds(hhmmss: str) -> int:
    """Convert HH:MM:SS string to integer seconds from midnight."""
    hh, mm, ss = hhmmss.split(":")
    return int(hh) * 3600 + int(mm) * 60 + int(ss)


def load_terminal_names(path: Path) -> tuple[list[str], dict[str, str]]:
    """Load stop order and code->station-name map from terminal_name.txt."""
    stop_order: list[str] = []
    code_to_name: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        code, name = line.split(":", 1)
        code = code.strip()
        name = name.strip()
        stop_order.append(code)
        code_to_name[code] = name
    return stop_order, code_to_name


def load_timetable(path: Path) -> tuple[dict[tuple[int, int], list[int]], dict[tuple[int, int], list[int]]]:
    """Load per-day per-direction segment seconds and departure seconds."""
    data = json.loads(path.read_text(encoding="utf-8"))
    seg_seconds: dict[tuple[int, int], list[int]] = {}
    departures: dict[tuple[int, int], list[int]] = {}

    for item in data:
        var_id = int(item["varId"])
        seg = [parse_hhmmss_to_seconds(x) for x in item["time_to_next_stop"]]
        dep = sorted(parse_hhmmss_to_seconds(x) for x in item.get("StartTime", []))
        for dow in item["days"]:
            key = (int(dow), var_id)
            seg_seconds[key] = seg
            departures[key] = dep
    return seg_seconds, departures


def estimate_wait_seconds(tap_sec: float, dep_list: list[int]) -> float:
    """Estimate wait as time from tap-in to next scheduled departure."""
    if not dep_list:
        return np.nan
    idx = int(np.searchsorted(dep_list, tap_sec, side="left"))
    if idx < len(dep_list):
        return float(dep_list[idx] - tap_sec)
    # After last departure, wait until first departure next day.
    return float((24 * 3600 - tap_sec) + dep_list[0])


def service_group_of_dow(dow: int) -> int:
    """Map day-of-week (0=Mon..6=Sun) to service-group index."""
    if dow in (0, 1, 2, 3):
        return 0
    if dow == 4:
        return 1
    return 2


def fit_segment_factors(
    stop_order: list[str],
    seg_seconds_by_day_var: dict[tuple[int, int], list[int]],
    departures_by_day_var: dict[tuple[int, int], list[int]],
    trip_files: list[Path],
) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], float]]:
    """Fit segment factors and station-travel constants per service-group and direction."""
    n_stops = len(stop_order)
    n_seg = n_stops - 1
    n_feat = n_seg + 1
    bias_idx = n_seg
    code_to_idx = {code: i for i, code in enumerate(stop_order)}

    # Normal-equation accumulators keyed by (service_group_idx, dir_axis).
    xtx: dict[tuple[int, int], np.ndarray] = {}
    xtb: dict[tuple[int, int], np.ndarray] = {}
    used_samples: dict[tuple[int, int], int] = {}
    for group_idx in range(len(SERVICE_GROUPS)):
        for dir_axis in (0, 1):
            key = (group_idx, dir_axis)
            xtx[key] = np.zeros((n_feat, n_feat), dtype=np.float64)
            xtb[key] = np.zeros(n_feat, dtype=np.float64)
            used_samples[key] = 0

    for file_path in trip_files:
        for chunk in pd.read_csv(
            file_path,
            usecols=["start__station", "end__station", "start__time", "end__time"],
            chunksize=200_000,
        ):
            start_dt = pd.to_datetime(chunk["start__time"], utc=True, errors="coerce").dt.tz_convert(LOCAL_TZ)
            end_dt = pd.to_datetime(chunk["end__time"], utc=True, errors="coerce").dt.tz_convert(LOCAL_TZ)
            duration = (end_dt - start_dt).dt.total_seconds()

            # Keep plausible observed durations for fitting.
            valid = duration.notna() & (duration >= MIN_OBS_DURATION_SEC) & (duration <= MAX_OBS_DURATION_SEC)
            if not valid.any():
                continue

            chunk = chunk.loc[valid]
            start_dt = start_dt.loc[valid]
            end_dt = end_dt.loc[valid]
            duration = duration.loc[valid]

            dow = start_dt.dt.dayofweek.to_numpy(dtype=np.int8)
            tap_sec = (
                start_dt.dt.hour.to_numpy(dtype=np.int32) * 3600
                + start_dt.dt.minute.to_numpy(dtype=np.int32) * 60
                + start_dt.dt.second.to_numpy(dtype=np.int32)
            ).astype(np.float64)

            s_codes = chunk["start__station"].astype(str).to_numpy()
            e_codes = chunk["end__station"].astype(str).to_numpy()
            dur = duration.to_numpy(dtype=np.float64)

            for i in range(len(chunk)):
                s_code = s_codes[i]
                e_code = e_codes[i]
                if s_code not in code_to_idx or e_code not in code_to_idx:
                    continue
                s_idx = code_to_idx[s_code]
                e_idx = code_to_idx[e_code]
                if s_idx == e_idx:
                    continue

                var_id = 1 if s_idx < e_idx else 2
                dir_axis = 0 if var_id == 1 else 1
                dow_i = int(dow[i])
                day_key = (dow_i, var_id)

                if day_key not in seg_seconds_by_day_var or day_key not in departures_by_day_var:
                    continue

                seg_all = seg_seconds_by_day_var[day_key]
                dep_list = departures_by_day_var[day_key]
                group_idx = service_group_of_dow(dow_i)
                group_dir_key = (group_idx, dir_axis)

                if var_id == 1:
                    seg_idxs = list(range(s_idx, e_idx))
                else:
                    s_rev = (n_stops - 1 - s_idx)
                    e_rev = (n_stops - 1 - e_idx)
                    seg_idxs = list(range(s_rev, e_rev))

                if not seg_idxs:
                    continue

                x_vals = np.asarray([float(seg_all[j]) for j in seg_idxs], dtype=np.float64)
                scheduled_path_sec = float(x_vals.sum())
                if scheduled_path_sec <= 0:
                    continue

                wait_sec = estimate_wait_seconds(float(tap_sec[i]), dep_list)
                if np.isnan(wait_sec):
                    continue

                # Observed non-wait time = in-vehicle + station travel time (to be learned).
                observed_non_wait = float(dur[i]) - wait_sec
                if observed_non_wait <= 0:
                    continue

                # Update normal equations using sparse segment participation + intercept.
                for a_local, a_idx in enumerate(seg_idxs):
                    a_val = x_vals[a_local]
                    xtb[group_dir_key][a_idx] += a_val * observed_non_wait
                    for b_local, b_idx in enumerate(seg_idxs):
                        xtx[group_dir_key][a_idx, b_idx] += a_val * x_vals[b_local]
                    xtx[group_dir_key][a_idx, bias_idx] += a_val
                    xtx[group_dir_key][bias_idx, a_idx] += a_val

                xtx[group_dir_key][bias_idx, bias_idx] += 1.0
                xtb[group_dir_key][bias_idx] += observed_non_wait

                used_samples[group_dir_key] += 1

    # Solve ridge-regularized linear systems to get multiplicative factors per group+direction.
    factors: dict[tuple[int, int], np.ndarray] = {}
    station_travel_sec: dict[tuple[int, int], float] = {}
    for group_idx in range(len(SERVICE_GROUPS)):
        group_name = SERVICE_GROUPS[group_idx][0]
        for dir_axis in (0, 1):
            key = (group_idx, dir_axis)
            reg = RIDGE_LAMBDA * np.eye(n_feat, dtype=np.float64)
            reg[bias_idx, bias_idx] = 0.0
            a = xtx[key] + reg
            b = xtb[key]
            try:
                theta = np.linalg.solve(a, b)
            except np.linalg.LinAlgError:
                print("LINEAR ALGEBRA ERROR")
                theta = np.linalg.lstsq(a, b, rcond=None)[0]
            k = theta[:n_seg]
            c = float(theta[bias_idx])
            # Clamp to avoid extreme unrealistic multipliers.
            k = np.clip(k, 0.9, 3.0)
            c = float(np.clip(c, 0.0, 1800.0))
            factors[key] = k
            station_travel_sec[key] = c

            direction_name = "forward" if dir_axis == 0 else "reverse"
            print(f"Samples used ({group_name}, {direction_name}): {used_samples[key]:,}")
            print(f"Estimated station travel ({group_name}, {direction_name}): {c:.2f} sec")
    return factors, station_travel_sec


def build_output_table(
    stop_order: list[str],
    code_to_name: dict[str, str],
    seg_seconds_by_day_var: dict[tuple[int, int], list[int]],
    factors: dict[tuple[int, int], np.ndarray],
    station_travel_sec: dict[tuple[int, int], float],
) -> pd.DataFrame:
    """Build final estimated between-station times table by service-group and direction."""
    n_seg = len(stop_order) - 1
    rows = []
    rev_order = list(reversed(stop_order))

    for group_idx, (group_name, group_days) in enumerate(SERVICE_GROUPS):
        # Use average scheduled segment time inside each service-group.
        fwd_daily = [seg_seconds_by_day_var[(d, 1)] for d in group_days if (d, 1) in seg_seconds_by_day_var]
        rev_daily = [seg_seconds_by_day_var[(d, 2)] for d in group_days if (d, 2) in seg_seconds_by_day_var]
        if not fwd_daily or not rev_daily:
            continue

        fwd_sched = np.mean(np.asarray(fwd_daily, dtype=np.float64), axis=0)
        rev_sched = np.mean(np.asarray(rev_daily, dtype=np.float64), axis=0)
        factors_fwd = factors[(group_idx, 0)]
        factors_rev = factors[(group_idx, 1)]
        station_fwd = station_travel_sec[(group_idx, 0)]
        station_rev = station_travel_sec[(group_idx, 1)]

        # Forward rows for this service-group.
        for i in range(n_seg):
            from_code = stop_order[i]
            to_code = stop_order[i + 1]
            scheduled_sec = float(fwd_sched[i])
            estimated_sec = float(scheduled_sec * factors_fwd[i])
            rows.append(
                {
                    "service_group": group_name,
                    "direction": "forward",
                    "from_code": from_code,
                    "from_name": code_to_name.get(from_code, from_code),
                    "to_code": to_code,
                    "to_name": code_to_name.get(to_code, to_code),
                    "scheduled_sec": round(scheduled_sec, 2),
                    "factor": round(float(factors_fwd[i]), 4),
                    "estimated_sec": round(estimated_sec, 2),
                    "station_travel_sec": round(float(station_fwd), 2),
                }
            )

        # Reverse rows for this service-group.
        for i in range(n_seg):
            from_code = rev_order[i]
            to_code = rev_order[i + 1]
            scheduled_sec = float(rev_sched[i])
            estimated_sec = float(scheduled_sec * factors_rev[i])
            rows.append(
                {
                    "service_group": group_name,
                    "direction": "reverse",
                    "from_code": from_code,
                    "from_name": code_to_name.get(from_code, from_code),
                    "to_code": to_code,
                    "to_name": code_to_name.get(to_code, to_code),
                    "scheduled_sec": round(scheduled_sec, 2),
                    "factor": round(float(factors_rev[i]), 4),
                    "estimated_sec": round(estimated_sec, 2),
                    "station_travel_sec": round(float(station_rev), 2),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    """Run calibration and export final estimated segment times."""
    stop_order, code_to_name = load_terminal_names(TERMINAL_NAME_PATH)
    seg_seconds_by_day_var, departures_by_day_var = load_timetable(TIMETABLE_PATH)

    trip_files = sorted(TRIPS_DIR.glob("*.csv"))
    if not trip_files:
        raise FileNotFoundError(f"No CSV files found in {TRIPS_DIR}")

    print(f"Using {len(trip_files)} trip files for calibration...")
    factors, station_travel_sec = fit_segment_factors(
        stop_order=stop_order,
        seg_seconds_by_day_var=seg_seconds_by_day_var,
        departures_by_day_var=departures_by_day_var,
        trip_files=trip_files,
    )

    out_df = build_output_table(
        stop_order=stop_order,
        code_to_name=code_to_name,
        seg_seconds_by_day_var=seg_seconds_by_day_var,
        factors=factors,
        station_travel_sec=station_travel_sec,
    )

    out_path = Path("estimated_time_between_stations.csv")
    out_df.to_csv(out_path, index=False)

    print("\nFinal estimated time between stations:")
    print(out_df.to_string(index=False))
    print(f"\nSaved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
