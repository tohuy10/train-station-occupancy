"""
Generate occupancy analysis, peak hour statistics, and high-resolution visual charts
for Ho Chi Minh City Metro Line 1 (Ben Thanh - Suoi Tien).
"""

import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for publication-ready figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 1.0

# Directory paths
BASE_DIR = Path(".")
TRIPS_DIR = BASE_DIR / "inputs" / "inputs"
TIMETABLE_PATH = BASE_DIR / "time_table.json"
TERMINAL_NAMES_PATH = BASE_DIR / "terminal_name.txt"
OUTPUT_IMG_DIR = BASE_DIR / "assets" / "images"
OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

tz_name = "Asia/Ho_Chi_Minh"

def load_terminal_names(path: Path):
    stop_order = []
    code_to_name = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        code, name = line.split(":", 1)
        code, name = code.strip(), name.strip()
        stop_order.append(code)
        code_to_name[code] = name
    return stop_order, code_to_name

def parse_hhmmss_to_seconds(hhmmss: str) -> int:
    hh, mm, ss = hhmmss.split(":")
    return int(hh) * 3600 + int(mm) * 60 + int(ss)

def load_timetable(path: Path):
    raw = path.read_text(encoding="utf-8")
    items = json.loads(raw)
    out = {}
    for entry in items:
        var_id = int(entry["varId"])
        seg_seconds = [parse_hhmmss_to_seconds(x) for x in entry["time_to_next_stop"]]
        for dow in entry["days"]:
            out[(int(dow), var_id)] = seg_seconds
    return out

def precompute_paths(stop_order, timetable_seg_seconds):
    n_stops = len(stop_order)
    out = {}
    for dow in range(7):
        for s in range(n_stops):
            for e in range(n_stops):
                if s == e:
                    continue
                var_id = 1 if s < e else 2
                seg_seconds_all = timetable_seg_seconds.get((dow, var_id))
                if seg_seconds_all is None:
                    continue
                if var_id == 1:
                    seg_idxs = list(range(s, e))
                else:
                    s_rev = (n_stops - 1 - s)
                    e_rev = (n_stops - 1 - e)
                    seg_idxs = list(range(s_rev, e_rev))
                path_seconds = np.asarray([seg_seconds_all[i] for i in seg_idxs], dtype=np.float64)
                total = float(path_seconds.sum())
                if total <= 0:
                    continue
                cum = np.cumsum(path_seconds) / total
                start_fracs = np.concatenate(([0.0], cum[:-1]))
                end_fracs = cum
                out[(dow, s, e)] = (seg_idxs, start_fracs, end_fracs)
    return out

def run_analysis():
    print("Loading metadata...")
    stop_order, code_to_name = load_terminal_names(TERMINAL_NAMES_PATH)
    timetable_seg_seconds = load_timetable(TIMETABLE_PATH)
    paths = precompute_paths(stop_order, timetable_seg_seconds)
    
    trip_files = sorted(list(TRIPS_DIR.glob("*.csv")))
    print(f"Found {len(trip_files)} CSV files.")
    
    bin_minutes = 15
    bin_sec = bin_minutes * 60
    bin_starts = np.arange(0, 24 * 3600, bin_sec, dtype=np.int32)
    n_bins = len(bin_starts)
    bin_labels = [f"{b//3600:02d}:{(b%3600)//60:02d}" for b in bin_starts]
    
    segments_fwd = [(stop_order[i], stop_order[i + 1]) for i in range(len(stop_order) - 1)]
    segments_rev = [(stop_order[::-1][i], stop_order[::-1][i + 1]) for i in range(len(stop_order) - 1)]
    n_seg = len(segments_fwd)
    n_sta = len(stop_order)
    
    # Track statistics
    tap_in_by_dow_bin = np.zeros((7, n_bins), dtype=np.int64)
    occ_sum = np.zeros((7, 2, n_seg, n_bins), dtype=np.float64)
    dates_by_dow = [set() for _ in range(7)]
    
    station_tap_in = np.zeros(n_sta, dtype=np.int64)
    station_tap_out = np.zeros(n_sta, dtype=np.int64)
    code_to_idx = {code: i for i, code in enumerate(stop_order)}
    
    total_trips = 0
    valid_trips = 0
    same_station_trips = 0
    
    dur_buckets = ["0-15m", "15-30m", "30-45m", "45-60m", "60-90m", "90m+"]
    dur_counts = np.zeros(len(dur_buckets), dtype=np.int64)

    print("Processing trip data files...")
    start_time_bench = time.time()
    
    for fpath in trip_files:
        print(f"Reading {fpath.name}...")
        for chunk in pd.read_csv(fpath, usecols=["start__station", "end__station", "start__time", "end__time"], chunksize=250_000):
            total_trips += len(chunk)
            
            s_dt = pd.to_datetime(chunk["start__time"], utc=True, errors="coerce").dt.tz_convert(tz_name)
            e_dt = pd.to_datetime(chunk["end__time"], utc=True, errors="coerce").dt.tz_convert(tz_name)
            dur_sec = (e_dt - s_dt).dt.total_seconds()
            
            s_code = chunk["start__station"].astype(str).to_numpy()
            e_code = chunk["end__station"].astype(str).to_numpy()
            
            same_mask = (s_code == e_code)
            same_station_trips += int(same_mask.sum())
            
            dur_min = dur_sec / 60.0
            dur_counts[0] += int(((dur_min >= 0) & (dur_min < 15)).sum())
            dur_counts[1] += int(((dur_min >= 15) & (dur_min < 30)).sum())
            dur_counts[2] += int(((dur_min >= 30) & (dur_min < 45)).sum())
            dur_counts[3] += int(((dur_min >= 45) & (dur_min < 60)).sum())
            dur_counts[4] += int(((dur_min >= 60) & (dur_min < 90)).sum())
            dur_counts[5] += int((dur_min >= 90).sum())
            
            valid_mask = (dur_sec > 0) & (dur_sec <= 90 * 60) & (~same_mask) & s_dt.notna() & e_dt.notna()
            valid_trips += int(valid_mask.sum())
            
            c_valid = chunk.loc[valid_mask]
            s_dt_v = s_dt.loc[valid_mask]
            dur_sec_v = dur_sec.loc[valid_mask].to_numpy()
            
            dows = s_dt_v.dt.dayofweek.to_numpy(dtype=np.int8)
            dates = s_dt_v.dt.date.to_numpy()
            
            s_sec = (s_dt_v.dt.hour * 3600 + s_dt_v.dt.minute * 60 + s_dt_v.dt.second).to_numpy(dtype=np.int32)
            
            s_codes_v = c_valid["start__station"].astype(str).to_numpy()
            e_codes_v = c_valid["end__station"].astype(str).to_numpy()
            
            for dow, date_val, ssec, sc, ec, dur in zip(dows, dates, s_sec, s_codes_v, e_codes_v, dur_sec_v):
                dates_by_dow[dow].add(date_val)
                
                s_idx = code_to_idx.get(sc)
                e_idx = code_to_idx.get(ec)
                if s_idx is None or e_idx is None:
                    continue
                
                station_tap_in[s_idx] += 1
                station_tap_out[e_idx] += 1
                
                bin_idx = int(ssec // bin_sec)
                if 0 <= bin_idx < n_bins:
                    tap_in_by_dow_bin[dow, bin_idx] += 1
                    
                dir_axis = 0 if s_idx < e_idx else 1
                key = (int(dow), int(s_idx), int(e_idx))
                if key not in paths:
                    continue
                
                seg_idxs, s_fracs, e_fracs = paths[key]
                trip_t0 = float(ssec)
                
                for seg_i, sf, ef in zip(seg_idxs, s_fracs, e_fracs):
                    t_entry = trip_t0 + sf * dur
                    t_exit  = trip_t0 + ef * dur
                    
                    if t_exit <= 0.0 or t_entry >= 86400.0:
                        continue
                    t_entry_c = max(0.0, t_entry)
                    t_exit_c  = min(86400.0, t_exit)
                    if t_exit_c <= t_entry_c:
                        continue
                    
                    b0 = int(t_entry_c // bin_sec)
                    b1 = int((t_exit_c - 1e-9) // bin_sec)
                    b0 = max(0, min(b0, n_bins - 1))
                    b1 = max(0, min(b1, n_bins - 1))
                    
                    for bi in range(b0, b1 + 1):
                        win_start = bi * bin_sec
                        win_end   = (bi + 1) * bin_sec
                        overlap = min(t_exit_c, win_end) - max(t_entry_c, win_start)
                        if overlap > 0:
                            occ_sum[dow, dir_axis, seg_i, bi] += overlap

    elapsed_bench = time.time() - start_time_bench
    print(f"Data processing completed in {elapsed_bench:.1f}s.")
    
    dow_counts = [len(dates_by_dow[d]) for d in range(7)]
    print("Days per DOW:", dow_counts)
    
    avg_occ = np.zeros_like(occ_sum)
    for dow in range(7):
        if dow_counts[dow] > 0:
            avg_occ[dow] = occ_sum[dow] / (dow_counts[dow] * bin_sec)
            
    sys_avg_load = avg_occ.sum(axis=(1, 2))  # shape: (7, n_bins)
    
    weekday_load = sys_avg_load[0:5].mean(axis=0)  # Mon-Fri average
    weekend_load = sys_avg_load[5:7].mean(axis=0)  # Sat-Sun average
    
    # 1. System-Wide Hourly Occupancy Line Chart
    plt.figure(figsize=(12, 5), dpi=300)
    hours = np.linspace(0, 24, n_bins, endpoint=False)
    plt.plot(hours, weekday_load, label="Weekday (Mon-Fri Avg)", color="#1f77b4", linewidth=2.5)
    plt.plot(hours, weekend_load, label="Weekend (Sat-Sun Avg)", color="#ff7f0e", linewidth=2.5, linestyle="--")
    
    plt.axvspan(6.5, 8.5, color="#d62728", alpha=0.15, label="Morning Peak (06:30 - 08:30)")
    plt.axvspan(16.5, 18.5, color="#d62728", alpha=0.15, label="Evening Peak (16:30 - 18:30)")
    
    plt.title("HCMC Metro Line 1 - Average System Passenger Load by Time of Day", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Hour of Day (Local Time UTC+7)", fontsize=11, labelpad=10)
    plt.ylabel("Avg Concurrent Passengers on Line", fontsize=11, labelpad=10)
    plt.xticks(np.arange(0, 25, 2), [f"{h:02d}:00" for h in range(0, 25, 2)])
    plt.xlim(5, 23)
    plt.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0")
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_DIR / "hourly_passenger_load.png")
    plt.close()
    
    # 2. Peak Hours Heatmap
    plt.figure(figsize=(13, 6), dpi=300)
    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    sys_hourly_load = sys_avg_load.reshape(7, 24, 4).mean(axis=2) # 7 x 24
    
    sns.heatmap(
        sys_hourly_load,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        yticklabels=dow_names,
        xticklabels=[f"{h:02d}:00" for h in range(24)],
        cbar_kws={'label': 'Avg Passengers in Transit'},
        linewidths=0.5
    )
    plt.title("Passenger Load Heatmap by Day of Week & Hour of Day", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Hour of Day", fontsize=11, labelpad=10)
    plt.ylabel("Day of Week", fontsize=11, labelpad=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_DIR / "occupancy_heatmap.png")
    plt.close()
    
    # 3. Congested Track Segments
    weekday_avg_occ = avg_occ[0:5].mean(axis=0)
    
    m_peak_bins = slice(26, 34) # 06:30 - 08:30
    e_peak_bins = slice(66, 74) # 16:30 - 18:30
    
    fwd_m_peak = weekday_avg_occ[0, :, m_peak_bins].mean(axis=1)
    fwd_e_peak = weekday_avg_occ[0, :, e_peak_bins].mean(axis=1)
    rev_m_peak = weekday_avg_occ[1, :, m_peak_bins].mean(axis=1)
    rev_e_peak = weekday_avg_occ[1, :, e_peak_bins].mean(axis=1)
    
    seg_labels_fwd = [f"{stop_order[i]}→{stop_order[i+1]}" for i in range(n_seg)]
    seg_labels_rev = [f"{stop_order[::-1][i]}→{stop_order[::-1][i+1]}" for i in range(n_seg)]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), dpi=300)
    
    x = np.arange(n_seg)
    width = 0.35
    
    ax1.bar(x - width/2, fwd_m_peak, width, label="Morning Peak (06:30-08:30)", color="#2b5c8f")
    ax1.bar(x + width/2, fwd_e_peak, width, label="Evening Peak (16:30-18:30)", color="#e06d53")
    ax1.set_title("Forward Direction (Ben Thanh → Suoi Tien) Segment Occupancy", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Avg Passengers", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(seg_labels_fwd, rotation=30, ha="right", fontsize=9)
    ax1.legend()
    
    ax2.bar(x - width/2, rev_m_peak, width, label="Morning Peak (06:30-08:30)", color="#2b5c8f")
    ax2.bar(x + width/2, rev_e_peak, width, label="Evening Peak (16:30-18:30)", color="#e06d53")
    ax2.set_title("Reverse Direction (Suoi Tien → Ben Thanh) Segment Occupancy", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Avg Passengers", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(seg_labels_rev, rotation=30, ha="right", fontsize=9)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_DIR / "segment_bottlenecks.png")
    plt.close()

    # 4. Station Boarding and Departure Throughput
    plt.figure(figsize=(12, 5), dpi=300)
    sta_names = [f"{c} ({code_to_name[c]})" for c in stop_order]
    
    x_sta = np.arange(n_sta)
    plt.bar(x_sta - width/2, station_tap_in / 1e3, width, label="Boarding (Tap-In)", color="#1f77b4")
    plt.bar(x_sta + width/2, station_tap_out / 1e3, width, label="Alighting (Tap-Out)", color="#2ca02c")
    
    plt.title("Total Passenger Volume per Station (in Thousands)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Station", fontsize=11, labelpad=10)
    plt.ylabel("Passenger Volume (x1,000)", fontsize=11, labelpad=10)
    plt.xticks(x_sta, sta_names, rotation=40, ha="right", fontsize=9)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_DIR / "station_throughput.png")
    plt.close()
    
    # Save Key Metrics to JSON
    results_summary = {
        "total_rows_scanned": int(total_trips),
        "valid_trips": int(valid_trips),
        "same_station_trips": int(same_station_trips),
        "valid_trip_pct": round(valid_trips / total_trips * 100, 2),
        "same_station_pct": round(same_station_trips / total_trips * 100, 2),
        "total_days_analyzed": sum(dow_counts),
        "dow_counts": dow_counts,
        "duration_distribution": dict(zip(dur_buckets, [int(c) for c in dur_counts])),
        "top_stations_tap_in": sorted(
            [{"station": stop_order[i], "name": code_to_name[stop_order[i]], "tap_in": int(station_tap_in[i])} for i in range(n_sta)],
            key=lambda x: x["tap_in"], reverse=True
        ),
        "top_stations_tap_out": sorted(
            [{"station": stop_order[i], "name": code_to_name[stop_order[i]], "tap_out": int(station_tap_out[i])} for i in range(n_sta)],
            key=lambda x: x["tap_out"], reverse=True
        ),
        "peak_hours_summary": {
            "morning_peak_avg_load": float(weekday_load[26:34].mean()),
            "morning_peak_max_load": float(weekday_load[26:34].max()),
            "evening_peak_avg_load": float(weekday_load[66:74].mean()),
            "evening_peak_max_load": float(weekday_load[66:74].max()),
            "offpeak_avg_load": float(weekday_load[36:60].mean()),
            "night_avg_load": float(weekday_load[84:96].mean()) if n_bins >= 96 else 0.0
        },
        "top_congested_segments_fwd": [
            {"segment": f"{stop_order[i]} → {stop_order[i+1]}", "morning_peak_avg": float(fwd_m_peak[i]), "evening_peak_avg": float(fwd_e_peak[i])}
            for i in np.argsort(-fwd_m_peak)[:5]
        ],
        "top_congested_segments_rev": [
            {"segment": f"{stop_order[::-1][i]} → {stop_order[::-1][i+1]}", "morning_peak_avg": float(rev_m_peak[i]), "evening_peak_avg": float(rev_e_peak[i])}
            for i in np.argsort(-rev_m_peak)[:5]
        ]
    }
    
    with open(BASE_DIR / "analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
        
    print("Analysis complete. Saved outputs to assets/images and analysis_results.json.")

if __name__ == "__main__":
    run_analysis()
