import math
import os
from typing import List

import numpy as np
import pandas as pd

# -------------------
# Config
# -------------------
N_NODES = 3
DATA_FREQ_S = 5  # 60 * 10 minutes
DATA_START_DATE = "2025-08-21T06:00:00Z"
DATA_END_DATE = "2025-08-23T00:00:00Z"
DATA_DIR = "data"

# Base location (Denmark, Sjaelland, Deer Park)
BASE_LAT = 55.785926
BASE_LON = 12.572046

RANDOM_SEED = 42  # Set to None for non-deterministic


# -------------------
# Helpers
# -------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ts_index() -> pd.DatetimeIndex:
    return pd.date_range(
        start=DATA_START_DATE,
        end=DATA_END_DATE,
        freq=f"{DATA_FREQ_S}s",
        tz="UTC",
        inclusive="both",
    )


def diurnal_factor(hours_since_start: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * (hours_since_start - 15.0) / 24.0)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def build_flux_style_header(columns: List[str]) -> str:
    group_map = {
        "_start": "true",
        "_stop": "true",
        "_field": "true",
        "_measurement": "true",
        "host": "true",
    }
    group_line = "#group," + ",".join([group_map.get(c, "false") for c in columns])
    dtype_map = {
        "result": "string",
        "table": "long",
        "_start": "dateTime:RFC3339",
        "_stop": "dateTime:RFC3339",
        "_time": "dateTime:RFC3339",
        "_value": "double",
        "_field": "string",
        "_measurement": "string",
        "host": "string",
    }
    dtype_line = "#datatype," + ",".join([dtype_map.get(c, "string") for c in columns])
    default_map = {"result": "_result"}
    default_line = "#default," + ",".join([default_map.get(c, "") for c in columns])
    return f"{group_line}\n{dtype_line}\n{default_line}\n"


def write_flux_csv(df: pd.DataFrame, path: str, header: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(header)
        f.write("," + ",".join(df.columns) + "\n")
        df_out = df.copy()
        df_out.insert(0, "", "")
        df_out.to_csv(f, index=False, header=False, float_format="%.6f")


# -------------------
# Main
# -------------------
def main():
    rng = np.random.default_rng(RANDOM_SEED)
    ensure_dir(DATA_DIR)

    idx = ts_index()
    n = len(idx)
    hours = (idx - idx[0]).total_seconds().to_numpy() / 3600.0
    dcycle = diurnal_factor(hours)

    # 1. Generate WIDE sensor data
    all_nodes_df_list = []
    for i in range(N_NODES):
        node_id = f"node-{i:03d}"
        lat = BASE_LAT + rng.uniform(-0.0004, 0.0004)
        lon = BASE_LON + rng.uniform(-0.0005, 0.0005)
        df = pd.DataFrame(
            {
                "_time": idx,
                "host": node_id,
                "latitude": lat + rng.normal(0.0, 0.00003, size=n),
                "longitude": lon + rng.normal(0.0, 0.00003, size=n),
                "temperature_c": (
                    rng.uniform(22.0, 30.0)
                    + 4.0 * dcycle
                    + rng.normal(0.0, 0.4, size=n)
                ),
                "humidity_pct": np.clip(
                    (
                        rng.uniform(35.0, 65.0)
                        - 5.0 * dcycle
                        + rng.normal(0.0, 1.5, size=n)
                    ),
                    15.0,
                    95.0,
                ),
                "wind_speed_ms": np.clip(
                    np.abs(rng.normal(3.0, 2.0, size=n) + 1.5 * (dcycle + 1.0)),
                    0.0,
                    12.0,
                ),
            }
        )
        all_nodes_df_list.append(df)
    sensor_df_wide = pd.concat(all_nodes_df_list, ignore_index=True)

    # 2. Transform sensor data to NARROW and write
    sensor_df_narrow = sensor_df_wide.melt(
        id_vars=["_time", "host"],
        value_vars=[
            "latitude",
            "longitude",
            "temperature_c",
            "humidity_pct",
            "wind_speed_ms",
        ],
        var_name="_field",
        value_name="_value",
    )
    sensor_df_narrow["result"] = ""
    sensor_df_narrow["table"] = 0
    sensor_df_narrow["_start"] = DATA_START_DATE
    sensor_df_narrow["_stop"] = DATA_END_DATE
    sensor_df_narrow["_measurement"] = "sensor_readings"
    sensor_df_narrow["_time"] = sensor_df_narrow["_time"].dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    sensor_columns = [
        "result",
        "table",
        "_start",
        "_stop",
        "_time",
        "_value",
        "_field",
        "_measurement",
        "host",
    ]
    sensor_header = build_flux_style_header(sensor_columns)
    write_flux_csv(
        sensor_df_narrow.sort_values(by=["_time", "host", "_field"])[sensor_columns],
        os.path.join(DATA_DIR, "sensor_readings_flux.csv"),
        sensor_header,
    )

    # 3. Generate and transform server metrics
    server_rows = []
    for t, g in sensor_df_wide.groupby("_time", sort=True):
        clat, clon = g["latitude"].mean(), g["longitude"].mean()
        dists = [
            haversine_m(row.latitude, row.longitude, clat, clon)
            for row in g.itertuples(index=False)
        ]
        radius_m = max(25.0, max(dists) if dists else 25.0)
        temp_mean, hum_mean, wind_mean = (
            g["temperature_c"].mean(),
            g["humidity_pct"].mean(),
            g["wind_speed_ms"].mean(),
        )

        def scale(v, vmin, vmax):
            return float(np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0))

        tsc, hsc, wsc = (
            scale(temp_mean, 20, 45),
            1.0 - scale(hum_mean, 10, 90),
            scale(wind_mean, 0, 12),
        )
        risk = float(np.clip((0.5 * tsc + 0.3 * hsc + 0.2 * wsc) * 100.0, 0, 100))
        server_rows.append(
            {
                "_time": t,
                "latitude": clat,
                "longitude": clon,
                "radius_m": radius_m,
                "fire_likelihood_pct": risk,
            }
        )

    server_df_narrow = pd.DataFrame(server_rows).melt(
        id_vars=["_time"],
        value_vars=[
            "latitude",
            "longitude",
            "radius_m",
            "fire_likelihood_pct",
        ],
        var_name="_field",
        value_name="_value",
    )
    server_df_narrow["result"] = ""
    server_df_narrow["table"] = 0
    server_df_narrow["_start"] = DATA_START_DATE
    server_df_narrow["_stop"] = DATA_END_DATE
    server_df_narrow["_measurement"] = "server_metrics"
    server_df_narrow["_time"] = server_df_narrow["_time"].dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    server_columns = [
        "result",
        "table",
        "_start",
        "_stop",
        "_time",
        "_value",
        "_field",
        "_measurement",
    ]
    server_header = build_flux_style_header(server_columns)
    write_flux_csv(
        server_df_narrow[server_columns],
        os.path.join(DATA_DIR, "server_metrics_flux.csv"),
        server_header,
    )

    print(f"Wrote Flux-style CSVs to: {os.path.abspath(DATA_DIR)}")
    print(" - sensor_readings_flux.csv")
    print(" - server_metrics_flux.csv")


if __name__ == "__main__":
    main()
