import os
import json
import math
import numpy as np
import pandas as pd
import torch
import re
import urllib.request

# ============================================================
# Flask-safe Matplotlib
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
import random

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    "OUT": "processed",
    "STATIC": "static",
    "SEQ_LEN": 24,
    "SUBSET_ROWS": 2000,
    "SUBSET_NODES": 30,
    "TEST_POINTS": 50,
    "EPOCHS": 5,
    "BATCH_SIZE": 64,
    "LR": 0.0005
}

OUT = CONFIG["OUT"]
STATIC = CONFIG["STATIC"]
SEQ_LEN = CONFIG["SEQ_LEN"]
SUBSET_ROWS = CONFIG["SUBSET_ROWS"]
SUBSET_NODES = CONFIG["SUBSET_NODES"]
TEST_POINTS = CONFIG["TEST_POINTS"]
EPOCHS = CONFIG["EPOCHS"]
BATCH_SIZE = CONFIG["BATCH_SIZE"]
LEARNING_RATE = CONFIG["LR"]

# ============================================================
# LOGGING CONFIG
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

from utils import create_seq, normalize, denormalize
from model import GAT_LSTM

# ============================================================
# D7 CONFIGURATION
# ============================================================
D7_DIR = os.path.join("data", "d7")

# This public 228-station file must correspond to the 228 columns
# in traffic_d7.csv.  Its row order is used to map:
#
# traffic_d7.csv column 0 -> metadata row 0
# traffic_d7.csv column 1 -> metadata row 1
# ...
#
# The file is expected to contain:
# ID, Fwy, Dir, District, Latitude, Longitude
D7_228_METADATA = os.path.join(
    D7_DIR,
    "PeMSD7_M_Station_Info_228.csv"
)

# User's additional D7 metadata file.  It is used only as an
# optional enrichment source; it is NOT used to establish the
# 228-column ordering.
D7_TEXT_METADATA = os.path.join(
    D7_DIR,
    "d07_text_meta_2023_12_22.txt"
)

# Keep I-405 from dominating the 30-node subset.
#
# The selection is NOT "3 sensors per freeway".  Instead:
#   1. every freeway represented in the 228 stations gets coverage
#   2. the remaining slots are allocated from freeway population
#      using sqrt(count), which reduces domination by large groups
#   3. no freeway can exceed this cap
#   4. sensors within each freeway are spatially spread
D7_MAX_SENSORS_PER_FREEWAY = 6


# ============================================================
# SEATTLE CONFIGURATION
# ============================================================
SEATTLE_DIR = os.path.join("data", "seattle")

# The Seattle speed matrix is a pandas pickle whose column names
# identify the loop detectors, for example d005es15036.
SEATTLE_SENSOR_NAMES = os.path.join(
    SEATTLE_DIR,
    "sensor_names.json"
)

# Local coordinate lookup used by the heatmap.
SEATTLE_COORD_LOOKUP = os.path.join(
    SEATTLE_DIR,
    "seattle_sensor_locations.csv"
)

# WSDOT's Northwest WebFlow database contains Name + Latitude +
# Longitude for roadway sensor locations.  The pipeline downloads
# it once, keeps the copy under data/seattle, and then builds the
# small project-specific lookup above.
SEATTLE_WSDOT_LOOKUP = os.path.join(
    SEATTLE_DIR,
    "NG_WebFlowNW.csv"
)

SEATTLE_WSDOT_URL = (
    "https://data.wsdot.wa.gov/traffic/WebFlow/"
    "NG_WebFlowNW.csv"
)

# Prevent a single Seattle roadway/corridor from dominating the
# 30-node model subset, while still allowing major corridors more
# representation than tiny groups.
SEATTLE_MAX_SENSORS_PER_ROADWAY = 8


# ============================================================
# LOGGING
# ============================================================
def emit(log_fn, message):
    """
    Unified logging:
    - Uses logging module
    - Still supports Flask streaming
    """

    logger.info(message)

    if log_fn is not None:
        try:
            log_fn(message)
        except Exception:
            pass

# ============================================================
# PATHS
# ============================================================
def get_paths(dataset):
    dataset = dataset.lower()

    paths = {
        "bay": os.path.join(
            "data", "bay", "pems-bay.h5"
        ),

        "d7": os.path.join(
            "data", "d7", "traffic_d7.csv"
        ),

        "seattle": os.path.join(
            "data", "seattle", "speed_matrix_2015"
        )
    }

    return paths.get(dataset)


# ============================================================
# DIRECTORY
# ============================================================
def ensure_dir(dataset):
    os.makedirs(
        os.path.join(OUT, dataset),
        exist_ok=True
    )

    os.makedirs(
        STATIC,
        exist_ok=True
    )


# ============================================================
# LOAD BAY DATA
# ============================================================
def load_bay_data(log_fn=print):
    path = get_paths("bay")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"BAY dataset not found: {path}"
        )

    emit(
        log_fn,
        "[INFO] Loading BAY from " + path
    )

    try:
        with pd.HDFStore(path, mode="r") as store:
            keys = store.keys()

            emit(
                log_fn,
                f"[INFO] HDF5 keys: {keys}"
            )

            if "/speed" not in keys:
                raise ValueError(
                    "BAY HDF5 does not contain /speed"
                )

            speed_obj = store.get("/speed")

        if isinstance(speed_obj, pd.DataFrame):
            emit(
                log_fn,
                f"[INFO] BAY HDF5 shape: {speed_obj.shape}"
            )

            numeric_df = speed_obj.select_dtypes(
                include=[np.number]
            )

            emit(
                log_fn,
                f"[INFO] Numeric BAY sensor columns: "
                f"{numeric_df.shape[1]}"
            )

            data = numeric_df.to_numpy(
                dtype=np.float32
            )
        else:
            data = np.asarray(
                speed_obj,
                dtype=np.float32
            )

        if data.ndim != 2:
            raise ValueError(
                f"Unexpected BAY data dimensions: {data.shape}"
            )

        data = np.nan_to_num(
            data,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        emit(
            log_fn,
            f"[INFO] BAY data loaded successfully: {data.shape}"
        )

        return data

    except Exception as e:
        emit(
            log_fn,
            "[WARN] BAY direct HDF5 read failed."
        )
        emit(
            log_fn,
            f"[WARN] {e}"
        )

        try:
            with pd.HDFStore(path, mode="r") as store:
                raw = store.get("/speed")

            if isinstance(raw, pd.DataFrame):
                numeric_df = raw.apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                data = numeric_df.to_numpy(
                    dtype=np.float32
                )
            else:
                data = np.asarray(
                    raw,
                    dtype=np.float32
                )

            data = np.nan_to_num(
                data,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            emit(
                log_fn,
                f"[INFO] BAY fallback successful: {data.shape}"
            )

            return data

        except Exception as fallback_error:
            raise RuntimeError(
                "Unable to read PEMS-BAY HDF5. "
                f"Original error: {e}; "
                f"Fallback error: {fallback_error}"
            )


# ============================================================
# ROBUST D7 CSV LOADER
# ============================================================
def load_d7_data(path, log_fn=print):
    """
    Load the user's 12672 x 228 D7 speed matrix.

    The supplied traffic_d7.csv is a headerless numeric matrix.
    Using numpy.loadtxt first prevents pandas from accidentally
    treating the first traffic row as a header.
    """

    emit(
        log_fn,
        f"[INFO] Loading D7 from {path}"
    )

    # Preferred path for the known headerless PeMS-style matrix.
    try:
        data = np.loadtxt(
            path,
            delimiter=",",
            dtype=np.float32
        )

        if data.ndim == 2 and data.shape[1] == 228:
            emit(
                log_fn,
                f"[INFO] D7 shape verified: {data.shape}"
            )

            data = np.nan_to_num(
                data,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            return data

    except Exception as e:
        emit(
            log_fn,
            f"[WARN] Direct numeric D7 read failed: {e}"
        )

    # Fallback for a CSV that contains a header.
    df = pd.read_csv(path)

    numeric_df = df.select_dtypes(
        include=[np.number]
    )

    if numeric_df.shape[1] == 0:
        numeric_df = df.apply(
            pd.to_numeric,
            errors="coerce"
        )

    data = numeric_df.to_numpy(
        dtype=np.float32
    )

    data = np.nan_to_num(
        data,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    if data.ndim != 2 or data.shape[1] != 228:
        raise ValueError(
            "D7 traffic matrix must contain exactly "
            f"228 sensor columns. Got {data.shape}"
        )

    emit(
        log_fn,
        f"[INFO] D7 shape verified: {data.shape}"
    )

    return data


# ============================================================
# LOAD D7 METADATA
# ============================================================
def load_d7_metadata(log_fn=print):
    """
    Load the 228-station lookup.

    The 228-station CSV is the authoritative ordering source for
    traffic_d7.csv because both contain 228 stations.

    Optional text metadata is used to enrich station information,
    but it is NOT used to reorder the traffic matrix.
    """

    if not os.path.exists(D7_228_METADATA):
        raise FileNotFoundError(
            "D7 228-station lookup not found:\n"
            f"{D7_228_METADATA}\n\n"
            "Place PeMSD7_M_Station_Info_228.csv in data\\d7."
        )

    emit(
        log_fn,
        "[INFO] D7 228-station lookup found:"
    )
    emit(
        log_fn,
        D7_228_METADATA
    )

    meta = pd.read_csv(
        D7_228_METADATA
    )

    # Normalize column names.
    meta.columns = [
        str(c).strip()
        for c in meta.columns
    ]

    required = {
        "ID",
        "Fwy",
        "Dir",
        "Latitude",
        "Longitude"
    }

    missing = required.difference(
        set(meta.columns)
    )

    if missing:
        raise ValueError(
            "D7 228-station metadata is missing columns: "
            + ", ".join(sorted(missing))
        )

    # Numeric conversion.
    for col in [
        "ID",
        "Fwy",
        "Latitude",
        "Longitude"
    ]:
        meta[col] = pd.to_numeric(
            meta[col],
            errors="coerce"
        )

    meta["Dir"] = (
        meta["Dir"]
        .astype(str)
        .str.strip()
    )

    meta = meta.dropna(
        subset=[
            "ID",
            "Fwy",
            "Latitude",
            "Longitude"
        ]
    ).reset_index(drop=True)

    if len(meta) != 228:
        raise ValueError(
            "The D7 228-station lookup must contain exactly "
            f"228 valid rows. Found {len(meta)}."
        )

    # --------------------------------------------------------
    # Optional text metadata enrichment.
    #
    # The user's text file has at least one malformed row.
    # Read with a regex delimiter and skip malformed rows so
    # that this auxiliary file cannot break the pipeline.
    # --------------------------------------------------------
    if os.path.exists(D7_TEXT_METADATA):
        try:
            text_meta = pd.read_csv(
                D7_TEXT_METADATA,
                sep=r"\s+",
                engine="python",
                on_bad_lines="skip"
            )

            if "ID" in text_meta.columns:
                text_meta["ID"] = pd.to_numeric(
                    text_meta["ID"],
                    errors="coerce"
                )

                text_meta = text_meta.dropna(
                    subset=["ID"]
                )

                # Preserve useful fields where available.
                enrich_cols = [
                    "County",
                    "City",
                    "State_PM",
                    "Abs_PM",
                    "Length",
                    "Type",
                    "Lanes",
                    "Name"
                ]

                available = [
                    c for c in enrich_cols
                    if c in text_meta.columns
                ]

                if available:
                    enrich = text_meta[
                        ["ID"] + available
                    ].drop_duplicates(
                        subset=["ID"]
                    )

                    meta = meta.merge(
                        enrich,
                        on="ID",
                        how="left"
                    )

                    emit(
                        log_fn,
                        "[INFO] Optional D7 text metadata "
                        "enrichment loaded."
                    )

        except Exception as e:
            emit(
                log_fn,
                "[WARN] Optional D7 text metadata "
                f"could not be loaded: {e}"
            )
    else:
        emit(
            log_fn,
            "[INFO] Optional D7 text metadata not found; "
            "continuing with 228-station lookup."
        )

    emit(
        log_fn,
        "[INFO] D7 228-station lookup loaded successfully."
    )

    return meta



# ============================================================
# SEATTLE SENSOR NAME / COORDINATE HELPERS
# ============================================================
def _normalise_seattle_sensor_name(value):
    """
    Return the geographic station key used to match Seattle loop
    detectors with WSDOT WebFlow locations.

    IMPORTANT: the Seattle Loop matrix and WSDOT do not use exactly
    the same station-name representation.  The Seattle matrix contains
    names such as::

        d005es15036
        i005es15125

    while WSDOT WebFlow commonly contains device-qualified names such as::

        005es15036:_MN_Stn
        005es15036:_MSH__5

    The part before the colon identifies the physical route/milepost
    location.  The suffix after the colon identifies the WSDOT device
    or lane configuration and must NOT participate in the geographic
    lookup.

    We therefore:
      1. remove any WSDOT device suffix after ':'
      2. remove the leading Seattle d/i direction marker
      3. remove remaining punctuation/whitespace

    Thus both ``d005es15036`` and ``005es15036:_MN_Stn`` resolve to
    the same geographic key ``005es15036``.
    """
    text = str(value).strip().lower()

    # WSDOT WebFlow station names may contain a device/lane suffix
    # after a colon.  It is not part of the physical station key.
    if ":" in text:
        text = text.split(":", 1)[0]

    # Remove the leading direction marker used by the Seattle Loop
    # dataset: d = decreasing direction, i = increasing direction.
    if len(text) > 1 and text[0] in {"d", "i"}:
        text = text[1:]

    # Remove whitespace and punctuation.
    text = re.sub(r"[^a-z0-9]", "", text)

    return text


def _save_seattle_sensor_names(sensor_names, log_fn=print):
    os.makedirs(SEATTLE_DIR, exist_ok=True)

    payload = {
        "dataset": "Seattle Loop",
        "count": int(len(sensor_names)),
        "sensor_names": [str(x) for x in sensor_names]
    }

    with open(
        SEATTLE_SENSOR_NAMES,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            payload,
            f,
            indent=4
        )

    emit(
        log_fn,
        f"[INFO] Seattle sensor names saved: "
        f"{SEATTLE_SENSOR_NAMES}"
    )


def _load_seattle_sensor_names(log_fn=print):
    if not os.path.exists(SEATTLE_SENSOR_NAMES):
        raise FileNotFoundError(
            "Seattle sensor-name mapping not found: "
            f"{SEATTLE_SENSOR_NAMES}"
        )

    with open(
        SEATTLE_SENSOR_NAMES,
        "r",
        encoding="utf-8"
    ) as f:
        payload = json.load(f)

    names = payload.get(
        "sensor_names",
        []
    )

    if not names:
        raise ValueError(
            "Seattle sensor-name mapping is empty."
        )

    return [str(x) for x in names]


def _download_seattle_wsdot_lookup(log_fn=print):
    """
    Download the WSDOT WebFlow location database once.

    The file is deliberately kept in data/seattle so subsequent
    runs use the local copy rather than repeatedly downloading it.
    """
    os.makedirs(
        SEATTLE_DIR,
        exist_ok=True
    )

    if os.path.exists(SEATTLE_WSDOT_LOOKUP):
        emit(
            log_fn,
            "[INFO] Seattle WSDOT lookup already available:"
        )
        emit(
            log_fn,
            SEATTLE_WSDOT_LOOKUP
        )
        return SEATTLE_WSDOT_LOOKUP

    emit(
        log_fn,
        "[INFO] Seattle coordinate lookup not found."
    )
    emit(
        log_fn,
        "[INFO] Downloading WSDOT WebFlow location lookup..."
    )

    try:
        urllib.request.urlretrieve(
            SEATTLE_WSDOT_URL,
            SEATTLE_WSDOT_LOOKUP
        )
    except Exception as e:
        raise RuntimeError(
            "Unable to download the Seattle WSDOT location "
            "lookup automatically. Download "
            f"{SEATTLE_WSDOT_URL} manually and save it as "
            f"{SEATTLE_WSDOT_LOOKUP}. Original error: {e}"
        )

    emit(
        log_fn,
        f"[INFO] Seattle WSDOT lookup saved: "
        f"{SEATTLE_WSDOT_LOOKUP}"
    )

    return SEATTLE_WSDOT_LOOKUP


def _read_seattle_wsdot_lookup(path, log_fn=print):
    """
    Read WSDOT WebFlow CSV while explicitly locating its real header.

    The WSDOT file can begin with Version/DateTime/Count metadata.
    This reader prints the discovered header and sample rows before
    any coordinate filtering so that matching problems are observable.
    """

    emit(log_fn, f"[DIAGNOSTIC] Reading Seattle WSDOT file: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Seattle WSDOT lookup not found: {path}"
        )

    raw_lines = []

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        errors="replace"
    ) as f:
        for index, line in enumerate(f):
            if index >= 25:
                break
            raw_lines.append(line.rstrip("\r\n"))

    emit(log_fn, "[DIAGNOSTIC] First WSDOT file lines:")
    for index, line in enumerate(raw_lines):
        emit(log_fn, f"    [{index:02d}] {line[:300]!r}")

    header_index = None

    for index, line in enumerate(raw_lines):
        cells = [
            cell.strip().strip('"').lstrip("\ufeff")
            for cell in line.split(",")
        ]
        lowered = {cell.lower() for cell in cells}

        if (
            "name" in lowered
            and "roadway" in lowered
            and "location" in lowered
            and "milepost" in lowered
        ):
            header_index = index
            break

    if header_index is None:
        with open(
            path,
            "r",
            encoding="utf-8-sig",
            errors="replace"
        ) as f:
            for index, line in enumerate(f):
                cells = [
                    cell.strip().strip('"').lstrip("\ufeff")
                    for cell in line.rstrip("\r\n").split(",")
                ]
                lowered = {cell.lower() for cell in cells}

                if (
                    "name" in lowered
                    and "roadway" in lowered
                    and "location" in lowered
                    and "milepost" in lowered
                ):
                    header_index = index
                    break

    if header_index is None:
        raise ValueError(
            "Could not locate the actual WSDOT WebFlow CSV header. "
            "Expected Name, Roadway, Location and Milepost columns. "
            f"File: {path}"
        )

    emit(
        log_fn,
        f"[DIAGNOSTIC] WSDOT CSV header located at physical "
        f"line {header_index + 1}."
    )

    df = pd.read_csv(
        path,
        skiprows=header_index,
        skipinitialspace=True,
        engine="python"
    )

    df.columns = [
        str(c).strip().strip('"').lstrip("\ufeff")
        for c in df.columns
    ]

    emit(log_fn, "[DIAGNOSTIC] WSDOT columns discovered:")
    emit(log_fn, "    " + repr(df.columns.tolist()))
    emit(log_fn, f"[DIAGNOSTIC] WSDOT raw data rows: {len(df)}")

    if "Name" not in df.columns:
        # Case-insensitive recovery.
        name_candidates = [
            c for c in df.columns
            if str(c).strip().lower() == "name"
        ]

        if name_candidates:
            df = df.rename(
                columns={name_candidates[0]: "Name"}
            )
        else:
            raise ValueError(
                "WSDOT WebFlow file does not contain a Name column. "
                "Found columns: "
                + ", ".join(str(c) for c in df.columns)
            )

    emit(
        log_fn,
        "[DIAGNOSTIC] First 10 parsed WSDOT records BEFORE "
        "coordinate filtering:"
    )

    for index, (_, row) in enumerate(df.head(10).iterrows()):
        values = [
            f"{column}={row[column]!r}"
            for column in df.columns
        ]

        emit(
            log_fn,
            f"    [{index:02d}] " + " | ".join(values)
        )

    return df


def _seattle_route_milepost(sensor_name):
    """
    Extract the stable route + milepost identity from a Seattle
    Loop detector name.

    Examples
    --------
    d005es15036 -> ('005', 150.36)
    i405es01234 -> ('405', 12.34)
    """
    text = str(sensor_name).strip().lower()

    if ":" in text:
        text = text.split(":", 1)[0]

    if len(text) > 1 and text[0] in {"d", "i"}:
        text = text[1:]

    text = re.sub(r"[^a-z0-9]", "", text)

    match = re.match(
        r"^(\d{3})es(\d{5})$",
        text
    )

    if not match:
        return None, None

    route = match.group(1)
    milepost = int(match.group(2)) / 100.0

    return route, milepost


def _seattle_roadway_route(value):
    """Extract the three-digit Washington route number from Roadway."""
    text = str(value).strip().upper()
    match = re.search(r"\b(\d{3})\b", text)
    if match:
        return match.group(1)
    return None


def _seattle_numeric_milepost(value):
    try:
        number = float(value)
        if np.isfinite(number):
            return number
    except Exception:
        pass
    return None



def diagnose_seattle_matching(
    sensor_names,
    wsdot,
    name_col,
    roadway_col=None,
    milepost_col=None,
    lat_col=None,
    lon_col=None,
    log_fn=print
):
    """
    Diagnostic-only inspection of the actual Seattle detector names
    and the actual WSDOT rows loaded by this pipeline.

    This function intentionally does NOT modify matching.  Its purpose
    is to show exactly what the two sides look like before any lookup is
    attempted.  This prevents us from guessing about the WSDOT Name
    format.
    """

    emit(log_fn, "")
    emit(log_fn, "=" * 72)
    emit(log_fn, "[DIAGNOSTIC] SEATTLE SENSOR / WSDOT MATCHING")
    emit(log_fn, "=" * 72)

    emit(
        log_fn,
        f"[DIAGNOSTIC] Seattle detector count: {len(sensor_names)}"
    )

    emit(log_fn, "[DIAGNOSTIC] First 20 Seattle detector names:")

    for i, sensor_name in enumerate(sensor_names[:20]):
        normalized = _normalise_seattle_sensor_name(sensor_name)
        route, milepost = _seattle_route_milepost(sensor_name)

        emit(
            log_fn,
            f"    [{i:02d}] original={sensor_name!r} | "
            f"normalized={normalized!r} | "
            f"route={route!r} | milepost={milepost!r}"
        )

    emit(log_fn, "")
    emit(
        log_fn,
        "[DIAGNOSTIC] WSDOT columns: "
        + str([str(c) for c in wsdot.columns.tolist()])
    )
    emit(
        log_fn,
        f"[DIAGNOSTIC] WSDOT rows after coordinate filtering: {len(wsdot)}"
    )

    emit(log_fn, "[DIAGNOSTIC] First 20 WSDOT rows used for matching:")

    for i, (_, row) in enumerate(wsdot.head(20).iterrows()):
        original_name = row.get(name_col, "")
        normalized = _normalise_seattle_sensor_name(original_name)
        route, milepost = _seattle_route_milepost(original_name)

        roadway = row.get(roadway_col, "") if roadway_col else ""
        explicit_mp = row.get(milepost_col, "") if milepost_col else ""
        latitude = row.get(lat_col, "") if lat_col else ""
        longitude = row.get(lon_col, "") if lon_col else ""

        emit(
            log_fn,
            f"    [{i:02d}] name={str(original_name)!r} | "
            f"normalized={normalized!r} | "
            f"route_from_name={route!r} | "
            f"mp_from_name={milepost!r} | "
            f"roadway={str(roadway)!r} | "
            f"column_milepost={str(explicit_mp)!r} | "
            f"lat={str(latitude)!r} | lon={str(longitude)!r}"
        )

    # ------------------------------------------------------------
    # Compare normalized Name keys.
    # ------------------------------------------------------------
    seattle_keys = {
        _normalise_seattle_sensor_name(x)
        for x in sensor_names
        if _normalise_seattle_sensor_name(x)
    }

    wsdot_keys = {
        _normalise_seattle_sensor_name(x)
        for x in wsdot[name_col].dropna().tolist()
        if _normalise_seattle_sensor_name(x)
    }

    exact_matches = sorted(
        seattle_keys.intersection(wsdot_keys)
    )

    emit(log_fn, "")
    emit(log_fn, "[DIAGNOSTIC] NORMALIZED NAME COMPARISON")
    emit(
        log_fn,
        f"    Seattle unique normalized keys : {len(seattle_keys)}"
    )
    emit(
        log_fn,
        f"    WSDOT unique normalized keys   : {len(wsdot_keys)}"
    )
    emit(
        log_fn,
        f"    Exact normalized matches       : {len(exact_matches)}"
    )

    if exact_matches:
        emit(log_fn, "    First normalized matches:")
        for key in exact_matches[:20]:
            emit(log_fn, f"        {key}")
    else:
        emit(log_fn, "    *** NO NORMALIZED NAME MATCHES ***")

    # ------------------------------------------------------------
    # Compare route + milepost keys.
    # ------------------------------------------------------------
    seattle_route_mp = set()
    for sensor_name in sensor_names:
        route, milepost = _seattle_route_milepost(sensor_name)
        if route is not None and milepost is not None:
            seattle_route_mp.add((str(route).zfill(3), round(float(milepost), 2)))

    wsdot_route_mp = set()
    for _, row in wsdot.iterrows():
        route, milepost = _seattle_route_milepost(row.get(name_col, ""))

        if route is None and roadway_col:
            route = _seattle_roadway_route(row.get(roadway_col, ""))

        if milepost is None and milepost_col:
            milepost = _seattle_numeric_milepost(row.get(milepost_col, ""))

        if route is not None and milepost is not None:
            wsdot_route_mp.add((str(route).zfill(3), round(float(milepost), 2)))

    route_mp_matches = sorted(
        seattle_route_mp.intersection(wsdot_route_mp)
    )

    emit(log_fn, "")
    emit(log_fn, "[DIAGNOSTIC] ROUTE + MILEPOST COMPARISON")
    emit(
        log_fn,
        f"    Seattle route/MP keys : {len(seattle_route_mp)}"
    )
    emit(
        log_fn,
        f"    WSDOT route/MP keys   : {len(wsdot_route_mp)}"
    )
    emit(
        log_fn,
        f"    Route/MP matches      : {len(route_mp_matches)}"
    )

    if route_mp_matches:
        emit(log_fn, "    First route/MP matches:")
        for route, milepost in route_mp_matches[:20]:
            emit(
                log_fn,
                f"        route={route}, milepost={milepost:.2f}"
            )
    else:
        emit(log_fn, "    *** NO ROUTE/MILEPOST MATCHES ***")

    # ------------------------------------------------------------
    # Explicit test for the first failing detector shown by the
    # user's error message.
    # ------------------------------------------------------------
    example = "d005es15036"
    example_normalized = _normalise_seattle_sensor_name(example)
    example_route, example_mp = _seattle_route_milepost(example)

    exact_rows = wsdot[
        wsdot[name_col].astype(str).map(
            _normalise_seattle_sensor_name
        ) == example_normalized
    ]

    route_rows = wsdot[
        wsdot["route_key"].astype(str).str.zfill(3)
        == str(example_route).zfill(3)
    ] if "route_key" in wsdot.columns and example_route is not None else pd.DataFrame()

    if not route_rows.empty and example_mp is not None:
        route_rows = route_rows[
            pd.to_numeric(
                route_rows["milepost_key"],
                errors="coerce"
            ).sub(float(example_mp)).abs() <= 0.01
        ]

    emit(log_fn, "")
    emit(log_fn, "[DIAGNOSTIC] EXPLICIT TEST: d005es15036")
    emit(log_fn, f"    normalized = {example_normalized!r}")
    emit(log_fn, f"    route      = {example_route!r}")
    emit(log_fn, f"    milepost   = {example_mp!r}")
    emit(log_fn, f"    exact Name matches = {len(exact_rows)}")
    emit(log_fn, f"    route/MP candidates = {len(route_rows)}")

    if not exact_rows.empty:
        emit(log_fn, "    Exact candidate rows:")
        for _, row in exact_rows.head(10).iterrows():
            emit(
                log_fn,
                f"        name={row.get(name_col, '')!r} | "
                f"lat={row.get(lat_col, '')!r} | "
                f"lon={row.get(lon_col, '')!r}"
            )

    if not route_rows.empty:
        emit(log_fn, "    Route/MP candidate rows:")
        for _, row in route_rows.head(10).iterrows():
            emit(
                log_fn,
                f"        name={row.get(name_col, '')!r} | "
                f"roadway={row.get(roadway_col, '')!r} | "
                f"milepost={row.get(milepost_col, '')!r} | "
                f"lat={row.get(lat_col, '')!r} | "
                f"lon={row.get(lon_col, '')!r}"
            )

    emit(log_fn, "")
    emit(log_fn, "=" * 72)
    emit(log_fn, "[DIAGNOSTIC] END SEATTLE MATCHING DIAGNOSTICS")
    emit(log_fn, "=" * 72)
    emit(log_fn, "")

def build_seattle_coordinate_lookup(
    sensor_names,
    log_fn=print
):
    """
    Build a real geographic lookup for the 323 Seattle Loop
    detectors.

    Matching is deliberately multi-stage because the Seattle Loop
    matrix and WSDOT WebFlow do not use the same identifier format.

    Primary match:
        normalized physical station key

    Fallback match:
        route number + milepost

    For example, the Seattle matrix contains ``d005es15036`` while
    WSDOT may publish ``005es15036:_MN___1``.  Both represent the
    physical location at route 005, milepost 150.36.  WSDOT documents
    that its Name field is of the form RRResMMMMM and that Latitude /
    Longitude identify the cabinet location.  We use those real
    coordinates; no synthetic grid is created.
    """
    os.makedirs(SEATTLE_DIR, exist_ok=True)

    required_sensor_keys = {
        _normalise_seattle_sensor_name(x)
        for x in sensor_names
    }

    # ------------------------------------------------------------
    # 1. Always rebuild the compact project lookup from the local
    #    WSDOT source.  This is intentional: an earlier failed
    #    Seattle lookup could otherwise remain cached and continue
    #    producing the same zero-match result.
    # ------------------------------------------------------------
    if os.path.exists(SEATTLE_COORD_LOOKUP):
        emit(
            log_fn,
            "[INFO] Existing Seattle coordinate lookup will be "
            "rebuilt from the local WSDOT source."
        )

    # ------------------------------------------------------------
    # 2. Get the WSDOT WebFlow source.
    # ------------------------------------------------------------
    raw_lookup = _download_seattle_wsdot_lookup(
        log_fn
    )

    wsdot = _read_seattle_wsdot_lookup(
        raw_lookup,
        log_fn=log_fn
    )

    # Make WSDOT column names case-insensitive and whitespace-safe.
    column_map = {
        str(c).strip().lower(): c
        for c in wsdot.columns
    }

    def wcol(name):
        return column_map.get(name.lower())

    name_col = wcol("name")
    lat_col = wcol("latitude")
    lon_col = wcol("longitude")
    roadway_col = wcol("roadway")
    milepost_col = wcol("milepost")
    location_col = wcol("location")

    emit(log_fn, "[DIAGNOSTIC] Resolved WSDOT columns:")
    emit(log_fn, f"    Name      = {name_col!r}")
    emit(log_fn, f"    Roadway   = {roadway_col!r}")
    emit(log_fn, f"    Location  = {location_col!r}")
    emit(log_fn, f"    Milepost  = {milepost_col!r}")
    emit(log_fn, f"    Latitude  = {lat_col!r}")
    emit(log_fn, f"    Longitude = {lon_col!r}")

    if not name_col:
        raise ValueError(
            "Seattle WSDOT lookup does not contain a Name column. "
            "Found columns: "
            + ", ".join(str(c) for c in wsdot.columns)
        )

    # ------------------------------------------------------------
    # DO NOT filter on coordinates yet.
    #
    # We first inspect identifiers. This is essential because a
    # WSDOT WebFlow file can contain Name/Roadway/Milepost records
    # whose Latitude/Longitude values are zero.
    # ------------------------------------------------------------

    if lat_col is not None and lon_col is not None:

        wsdot[lat_col] = pd.to_numeric(
            wsdot[lat_col],
            errors="coerce"
        )

        wsdot[lon_col] = pd.to_numeric(
            wsdot[lon_col],
            errors="coerce"
        )

        numeric_coordinate_mask = (
            wsdot[lat_col].notna()
            & wsdot[lon_col].notna()
        )

        nonzero_coordinate_mask = (
            numeric_coordinate_mask
            & (wsdot[lat_col] != 0)
            & (wsdot[lon_col] != 0)
        )

        plausible_coordinate_mask = (
            nonzero_coordinate_mask
            & wsdot[lat_col].between(45.0, 50.0)
            & wsdot[lon_col].between(-125.0, -120.0)
        )

        zero_coordinate_count = int(
            (
                wsdot[lat_col].fillna(0).eq(0)
                & wsdot[lon_col].fillna(0).eq(0)
            ).sum()
        )

        emit(log_fn, "[DIAGNOSTIC] WSDOT coordinate availability:")
        emit(log_fn, f"    Total rows                     : {len(wsdot)}")
        emit(
            log_fn,
            f"    Numeric Latitude/Longitude rows: "
            f"{int(numeric_coordinate_mask.sum())}"
        )
        emit(
            log_fn,
            f"    Non-zero coordinate rows       : "
            f"{int(nonzero_coordinate_mask.sum())}"
        )
        emit(
            log_fn,
            f"    Plausible Washington rows      : "
            f"{int(plausible_coordinate_mask.sum())}"
        )
        emit(
            log_fn,
            f"    Latitude=0 AND Longitude=0     : "
            f"{zero_coordinate_count}"
        )

    else:

        plausible_coordinate_mask = pd.Series(
            False,
            index=wsdot.index
        )

        emit(
            log_fn,
            "[DIAGNOSTIC] Latitude/Longitude columns are absent "
            "from this WSDOT file."
        )

    # ------------------------------------------------------------
    # Identifier keys are created BEFORE coordinate filtering.
    # ------------------------------------------------------------

    wsdot["lookup_key"] = (
        wsdot[name_col]
        .astype(str)
        .map(_normalise_seattle_sensor_name)
    )

    wsdot["route_key"] = (
        wsdot[name_col]
        .astype(str)
        .map(
            lambda x:
                _seattle_route_milepost(x)[0]
        )
    )

    wsdot["milepost_key"] = (
        wsdot[name_col]
        .astype(str)
        .map(
            lambda x:
                _seattle_route_milepost(x)[1]
        )
    )

    if roadway_col:

        missing_route = wsdot["route_key"].isna()

        wsdot.loc[
            missing_route,
            "route_key"
        ] = (
            wsdot.loc[
                missing_route,
                roadway_col
            ].map(
                _seattle_roadway_route
            )
        )

    if milepost_col:

        missing_mp = wsdot["milepost_key"].isna()

        wsdot.loc[
            missing_mp,
            "milepost_key"
        ] = (
            wsdot.loc[
                missing_mp,
                milepost_col
            ].map(
                _seattle_numeric_milepost
            )
        )

    # ------------------------------------------------------------
    # Run diagnostics BEFORE dropping unusable coordinates.
    # ------------------------------------------------------------

    diagnose_seattle_matching(
        sensor_names=sensor_names,
        wsdot=wsdot,
        name_col=name_col,
        roadway_col=roadway_col,
        milepost_col=milepost_col,
        lat_col=lat_col,
        lon_col=lon_col,
        log_fn=log_fn
    )

    # ------------------------------------------------------------
    # Now require real coordinates for the final geographic lookup.
    # ------------------------------------------------------------

    if lat_col is None or lon_col is None:
        raise ValueError(
            "Seattle WSDOT file has no Latitude/Longitude columns. "
            "The actual WSDOT columns were printed above."
        )

    wsdot = wsdot[
        wsdot["lookup_key"].astype(str).str.len() > 0
    ].copy()

    wsdot = wsdot[
        plausible_coordinate_mask.loc[
            wsdot.index
        ]
    ].copy()

    emit(
        log_fn,
        "[INFO] Seattle WSDOT rows with valid geographic "
        f"coordinates after filtering: {len(wsdot)}"
    )

    if len(wsdot) == 0:
        raise ValueError(
            "Seattle WSDOT file contains no usable non-zero "
            "Washington Latitude/Longitude values. "
            "The identifier diagnostics above show whether the "
            "Seattle detector names match WSDOT."
        )

    # ------------------------------------------------------------
    # 4. Build lookup dictionaries.  Do NOT collapse by route+MP
    #    before matching, because the same milepost can occur in both
    #    directions.  We only collapse exact physical Name keys.
    # ------------------------------------------------------------
    exact_lookup = {}

    for _, row in wsdot.iterrows():
        key = str(row["lookup_key"])
        if key and key not in exact_lookup:
            exact_lookup[key] = row

    route_mp_lookup = {}

    for _, row in wsdot.iterrows():
        route = row["route_key"]
        mp = row["milepost_key"]

        if pd.isna(route) or pd.isna(mp):
            continue

        try:
            route = str(route).zfill(3)
            mp = float(mp)
        except Exception:
            continue

        key = (route, round(mp, 2))
        route_mp_lookup.setdefault(key, []).append(row)

    emit(
        log_fn,
        "[INFO] Seattle WSDOT geographic keys available: "
        f"{len(exact_lookup)} exact, "
        f"{len(route_mp_lookup)} route/milepost"
    )

    # ------------------------------------------------------------
    # 5. Match every detector.
    # ------------------------------------------------------------
    rows = []
    unmatched = []
    exact_count = 0
    route_mp_count = 0

    for matrix_index, sensor_name in enumerate(sensor_names):
        sensor_name = str(sensor_name)
        normalized = _normalise_seattle_sensor_name(
            sensor_name
        )

        row = exact_lookup.get(normalized)
        match_method = "exact"

        if row is None:
            route, milepost = _seattle_route_milepost(
                sensor_name
            )

            candidates = route_mp_lookup.get(
                (route, round(milepost, 2)),
                []
            ) if route is not None and milepost is not None else []

            if candidates:
                # If WSDOT has multiple rows for the same physical
                # route/milepost, choose the first valid cabinet record.
                row = candidates[0]
                match_method = "route_milepost"

        if row is None:
            unmatched.append(sensor_name)
            continue

        if match_method == "exact":
            exact_count += 1
        else:
            route_mp_count += 1

        record = {
            "matrix_column": int(matrix_index),
            "sensor_name": sensor_name,
            "latitude": float(row[lat_col]),
            "longitude": float(row[lon_col]),
            "match_method": match_method
        }

        if name_col in row.index and pd.notna(row[name_col]):
            record["wsdot_name"] = str(row[name_col])

        if roadway_col and pd.notna(row[roadway_col]):
            roadway_value = str(row[roadway_col]).strip()
            if roadway_value:
                record["roadway"] = roadway_value

        if "roadway" not in record:
            route, _ = _seattle_route_milepost(sensor_name)
            record["roadway"] = (
                route if route is not None else "Unknown"
            )

        if location_col and pd.notna(row[location_col]):
            record["location"] = str(row[location_col]).strip()

        if milepost_col and pd.notna(row[milepost_col]):
            mp = _seattle_numeric_milepost(row[milepost_col])
            if mp is not None:
                record["milepost"] = float(mp)
        else:
            _, mp = _seattle_route_milepost(sensor_name)
            if mp is not None:
                record["milepost"] = float(mp)

        rows.append(record)

    lookup = pd.DataFrame(rows)

    emit(
        log_fn,
        "[INFO] Seattle coordinate matching: "
        f"{len(lookup)}/{len(sensor_names)} matched "
        f"({exact_count} exact + {route_mp_count} route/milepost)."
    )

    if len(lookup) < SUBSET_NODES:
        sample = unmatched[:15]
        raise ValueError(
            "Only "
            f"{len(lookup)} of {len(sensor_names)} Seattle "
            "detectors could be matched to real coordinates. "
            f"At least {SUBSET_NODES} are required. "
            f"First unmatched sensors: {sample}"
        )

    lookup = lookup.sort_values(
        "matrix_column"
    ).reset_index(drop=True)

    lookup.to_csv(
        SEATTLE_COORD_LOOKUP,
        index=False
    )

    emit(
        log_fn,
        "[INFO] Seattle real coordinate lookup built: "
        f"{len(lookup)} matched detectors."
    )

    emit(
        log_fn,
        f"[INFO] Seattle coordinate lookup saved: "
        f"{SEATTLE_COORD_LOOKUP}"
    )

    if unmatched:
        emit(
            log_fn,
            f"[WARN] {len(unmatched)} Seattle detectors "
            "were not matched and will not be selected."
        )

    return lookup

def load_seattle_metadata(
    sensor_names=None,
    log_fn=print
):
    if sensor_names is None:
        sensor_names = _load_seattle_sensor_names(
            log_fn
        )

    lookup = build_seattle_coordinate_lookup(
        sensor_names,
        log_fn
    )

    # Ensure matrix order can be reconstructed.
    lookup["matrix_column"] = pd.to_numeric(
        lookup["matrix_column"],
        errors="coerce"
    )

    lookup = lookup.dropna(
        subset=["matrix_column"]
    ).copy()

    lookup["matrix_column"] = lookup[
        "matrix_column"
    ].astype(int)

    lookup = lookup.sort_values(
        "matrix_column"
    ).reset_index(
        drop=True
    )

    return lookup


def select_seattle_sensors(
    data,
    sensor_names,
    log_fn=print
):
    """
    Select 30 Seattle detectors that are geographically and
    network representative.

    The old implementation simply took columns 0..29.  Because
    the Seattle matrix is ordered by detector identity rather than
    map geography, that can collapse the visualization into a
    narrow corridor.  This selection uses roadway-aware quotas
    followed by geographic farthest-point sampling.
    """
    if data.shape[1] != len(sensor_names):
        raise ValueError(
            "Seattle sensor-name count does not match "
            f"data columns: {len(sensor_names)} vs "
            f"{data.shape[1]}."
        )

    meta = load_seattle_metadata(
        sensor_names,
        log_fn
    ).copy()

    # build_seattle_coordinate_lookup intentionally stores its
    # geographic fields as lowercase ``latitude``/``longitude``.
    # The shared spatial sampler uses the D7 metadata convention
    # ``Latitude``/``Longitude``.  Normalize the Seattle metadata
    # here before any spatial sampling is attempted.
    if "latitude" in meta.columns and "Latitude" not in meta.columns:
        meta["Latitude"] = pd.to_numeric(
            meta["latitude"],
            errors="coerce"
        )

    if "longitude" in meta.columns and "Longitude" not in meta.columns:
        meta["Longitude"] = pd.to_numeric(
            meta["longitude"],
            errors="coerce"
        )

    emit(
        log_fn,
        "[DIAGNOSTIC] Seattle metadata columns before spatial selection:"
    )
    emit(
        log_fn,
        "    " + repr(meta.columns.tolist())
    )

    if "Latitude" not in meta.columns or "Longitude" not in meta.columns:
        raise ValueError(
            "Seattle metadata does not contain usable Latitude/Longitude "
            "columns after coordinate lookup. Available columns: "
            + ", ".join(str(c) for c in meta.columns)
        )

    coordinate_valid = (
        np.isfinite(meta["Latitude"].to_numpy(dtype=np.float64))
        & np.isfinite(meta["Longitude"].to_numpy(dtype=np.float64))
    )

    if int(coordinate_valid.sum()) < SUBSET_NODES:
        raise ValueError(
            "Seattle metadata contains only "
            f"{int(coordinate_valid.sum())} valid coordinates; "
            f"{SUBSET_NODES} are required for spatial selection."
        )

    if len(meta) < SUBSET_NODES:
        raise ValueError(
            f"Only {len(meta)} Seattle sensors have real "
            f"coordinates; {SUBSET_NODES} are required."
        )

    if "roadway" not in meta.columns:
        meta["roadway"] = "Unknown"

    meta["RoadwayNorm"] = (
        meta["roadway"]
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )

    groups = {
        roadway: list(group.index)
        for roadway, group in meta.groupby(
            "RoadwayNorm",
            sort=True
        )
    }

    groups = {
        roadway: indices
        for roadway, indices in groups.items()
        if indices
    }

    target = min(
        SUBSET_NODES,
        len(meta)
    )

    roadway_names = sorted(
        groups.keys()
    )

    quotas = {
        roadway: 0
        for roadway in roadway_names
    }

    # Guarantee broad roadway coverage where possible.
    if len(roadway_names) <= target:
        for roadway in roadway_names:
            quotas[roadway] = 1

    remaining = target - sum(
        quotas.values()
    )

    while remaining > 0:
        candidates = [
            roadway
            for roadway in roadway_names
            if quotas[roadway] < min(
                len(groups[roadway]),
                SEATTLE_MAX_SENSORS_PER_ROADWAY
            )
        ]

        if not candidates:
            break

        best = max(
            candidates,
            key=lambda roadway: (
                math.sqrt(
                    len(groups[roadway])
                ) /
                max(
                    quotas[roadway],
                    1
                ),
                len(groups[roadway])
            )
        )

        quotas[best] += 1
        remaining -= 1

    # Fill unused capacity if the roadway cap prevented 30.
    if remaining > 0:
        candidates = [
            roadway
            for roadway in roadway_names
            if quotas[roadway] < len(
                groups[roadway]
            )
        ]

        while remaining > 0 and candidates:
            best = max(
                candidates,
                key=lambda roadway: (
                    len(groups[roadway]) -
                    quotas[roadway]
                )
            )

            quotas[best] += 1
            remaining -= 1

            candidates = [
                roadway
                for roadway in roadway_names
                if quotas[roadway] < len(
                    groups[roadway]
                )
            ]

    selected_indices = []

    for roadway in roadway_names:
        quota = quotas[roadway]

        if quota <= 0:
            continue

        group_df = meta.loc[
            groups[roadway]
        ]

        selected = _spatially_spread_indices(
            group_df,
            quota
        )

        selected_indices.extend(
            selected
        )

    selected_indices = sorted(
        selected_indices
    )

    if len(selected_indices) != target:
        raise RuntimeError(
            "Seattle sensor selection produced "
            f"{len(selected_indices)} sensors; "
            f"{target} required."
        )

    selected_meta = meta.loc[
        selected_indices
    ].copy()

    selected_meta["traffic_column"] = (
        selected_meta["matrix_column"]
        .astype(int)
    )

    # Save reproducible selection.
    selection_payload = {
        "dataset": "Seattle",
        "total_source_sensors": int(
            data.shape[1]
        ),
        "selected_sensors": int(
            len(selected_indices)
        ),
        "max_sensors_per_roadway": int(
            SEATTLE_MAX_SENSORS_PER_ROADWAY
        ),
        "selected_indices": [
            int(x)
            for x in selected_meta[
                "matrix_column"
            ].tolist()
        ],
        "roadway_distribution": {
            roadway: int(
                quotas[roadway]
            )
            for roadway in roadway_names
            if quotas[roadway] > 0
        },
        "stations": []
    }

    for _, row in selected_meta.iterrows():
        station = {
            "traffic_column": int(
                row["matrix_column"]
            ),
            "sensor_name": str(
                row["sensor_name"]
            ),
            "latitude": float(
                row["latitude"]
            ),
            "longitude": float(
                row["longitude"]
            )
        }

        for field in [
            "wsdot_name",
            "roadway",
            "location",
            "milepost"
        ]:
            if field in row and pd.notna(row[field]):
                station[field] = (
                    float(row[field])
                    if field == "milepost"
                    else str(row[field])
                )

        selection_payload[
            "stations"
        ].append(station)

    selection_path = os.path.join(
        OUT,
        "seattle",
        "selected_sensors.json"
    )

    with open(
        selection_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            selection_payload,
            f,
            indent=4
        )

    mapping = []

    for model_index, (_, row) in enumerate(
        selected_meta.iterrows()
    ):
        item = {
            "model_sensor": int(model_index),
            "traffic_column": int(
                row["matrix_column"]
            ),
            "sensor_name": str(
                row["sensor_name"]
            ),
            "latitude": float(
                row["latitude"]
            ),
            "longitude": float(
                row["longitude"]
            )
        }

        for field in [
            "wsdot_name",
            "roadway",
            "location",
            "milepost"
        ]:
            if field in row and pd.notna(row[field]):
                item[field] = (
                    float(row[field])
                    if field == "milepost"
                    else str(row[field])
                )

        mapping.append(item)

    mapping_path = os.path.join(
        OUT,
        "seattle",
        "sensor_mapping.json"
    )

    with open(
        mapping_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            mapping,
            f,
            indent=4
        )

    selected_meta.to_csv(
        os.path.join(
            OUT,
            "seattle",
            "selected_sensor_metadata.csv"
        ),
        index=False
    )

    emit(
        log_fn,
        f"[INFO] Seattle selected {len(mapping)} "
        "real sensors across the network."
    )

    emit(
        log_fn,
        "[INFO] Selected Seattle roadway distribution: "
        + str(
            selection_payload[
                "roadway_distribution"
            ]
        )
    )

    emit(
        log_fn,
        f"[INFO] Seattle sensor selection saved: "
        f"{selection_path}"
    )

    emit(
        log_fn,
        f"[INFO] Seattle model sensor mapping saved: "
        f"{mapping_path}"
    )

    return (
        [
            int(x)
            for x in selected_meta[
                "matrix_column"
            ].tolist()
        ],
        selected_meta
    )


# ============================================================
# LOAD RAW DATA
# ============================================================
def load_raw_data(dataset, log_fn=print):
    dataset = dataset.lower()

    path = get_paths(dataset)

    if not path:
        raise ValueError(
            f"Unsupported dataset: {dataset}"
        )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    # --------------------------------------------------------
    # BAY
    # --------------------------------------------------------
    if dataset == "bay":
        return load_bay_data(log_fn)

    # --------------------------------------------------------
    # D7
    # --------------------------------------------------------
    if dataset == "d7":
        return load_d7_data(
            path,
            log_fn
        )

    # --------------------------------------------------------
    # SEATTLE
    # --------------------------------------------------------
    if dataset == "seattle":
        emit(
            log_fn,
            f"[INFO] Loading Seattle from {path}"
        )

        try:
            data_obj = pd.read_pickle(path)
        except Exception:
            data_obj = np.load(
                path,
                allow_pickle=True
            )

        if isinstance(data_obj, pd.DataFrame):
            numeric_df = data_obj.select_dtypes(
                include=[np.number]
            )

            sensor_names = [
                str(column)
                for column in numeric_df.columns
            ]

            if numeric_df.shape[1] == 0:
                raise ValueError(
                    "Seattle pickle contains no numeric "
                    "sensor columns."
                )

            data = numeric_df.to_numpy(
                dtype=np.float32
            )

        else:
            data = np.asarray(
                data_obj,
                dtype=np.float32
            )

            if data.ndim != 2:
                raise ValueError(
                    f"Seattle data must be 2-D. Got {data.shape}"
                )

            # If a raw ndarray is supplied, use an existing local
            # sensor-name file.  The normal project path is a
            # pandas DataFrame and therefore preserves the names.
            if os.path.exists(SEATTLE_SENSOR_NAMES):
                sensor_names = _load_seattle_sensor_names(
                    log_fn
                )
            else:
                raise ValueError(
                    "Seattle data is an ndarray and no sensor-name "
                    "mapping exists. The project needs the original "
                    "DataFrame column names to map real coordinates."
                )

        data = np.nan_to_num(
            data,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        if len(sensor_names) != data.shape[1]:
            raise ValueError(
                "Seattle sensor-name count does not match "
                f"data columns: {len(sensor_names)} vs "
                f"{data.shape[1]}."
            )

        _save_seattle_sensor_names(
            sensor_names,
            log_fn
        )

        emit(
            log_fn,
            f"[INFO] Seattle numerical shape: {data.shape}"
        )

        emit(
            log_fn,
            f"[INFO] Seattle detector names preserved: "
            f"{len(sensor_names)}"
        )

        return data

    raise ValueError(
        f"Unsupported dataset: {dataset}"
    )


# ============================================================
# DATA CONVERSION
# ============================================================
def convert_data(dataset, log_fn=print):
    dataset = dataset.lower()

    ensure_dir(dataset)

    emit(
        log_fn,
        f"[INFO] Reading raw {dataset.upper()} data..."
    )

    data = load_raw_data(
        dataset,
        log_fn
    )

    if data.ndim != 2:
        raise ValueError(
            f"Converted data must be 2-D. Got {data.shape}"
        )

    np.save(
        os.path.join(
            OUT,
            dataset,
            "data.npy"
        ),
        data
    )

    emit(
        log_fn,
        f"[INFO] Raw data saved: "
        f"{OUT}/{dataset}/data.npy"
    )

    emit(
        log_fn,
        f"[INFO] Data shape: {data.shape}"
    )

    emit(
        log_fn,
        f"[INFO] Speed range: "
        f"{np.min(data):.2f} - {np.max(data):.2f} mph"
    )

    return data


# ============================================================
# D7 SPATIAL SELECTION HELPERS
# ============================================================
def _normalise_freeway(value):
    """
    Convert freeway identifiers to stable strings.

    Examples:
        405.0 -> "405"
        "I-405" -> "405"
        "405" -> "405"
    """
    text = str(value).strip().upper()

    text = text.replace("I-", "")
    text = text.replace("I ", "")
    text = text.replace("SR-", "")
    text = text.replace("SR ", "")
    text = text.replace("CA-", "")
    text = text.replace("CA ", "")

    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except Exception:
        pass

    return text


def _distance_matrix(points):
    """
    Approximate geographic distance in kilometres.

    At Los Angeles latitudes, longitude degrees are scaled by
    cos(latitude), which is sufficient for deterministic station
    spacing without requiring another GIS dependency.
    """
    points = np.asarray(
        points,
        dtype=np.float64
    )

    if len(points) == 0:
        return np.empty((0, 0))

    lat = np.radians(
        points[:, 0]
    )

    scale = np.cos(
        np.mean(lat)
    )

    xy = np.column_stack(
        [
            points[:, 0] * 111.0,
            points[:, 1] * 111.0 * scale
        ]
    )

    diff = (
        xy[:, None, :] -
        xy[None, :, :]
    )

    return np.sqrt(
        np.sum(
            diff ** 2,
            axis=2
        )
    )


def _spatially_spread_indices(
    group_df,
    quota
):
    """
    Deterministically choose 'quota' stations from one freeway
    while spreading them geographically.

    Farthest-point sampling is used instead of taking the first
    rows, which prevents a corridor from being represented by a
    tight cluster of adjacent stations.
    """

    indices = list(
        group_df.index
    )

    if quota >= len(indices):
        return indices

    if quota <= 0:
        return []

    points = group_df[
        ["Latitude", "Longitude"]
    ].to_numpy(
        dtype=np.float64
    )

    # Work in local group positions.
    distances = _distance_matrix(points)

    # Start near the geographic centre.
    centre = np.mean(
        points,
        axis=0
    )

    first = int(
        np.argmin(
            np.sum(
                (points - centre) ** 2,
                axis=1
            )
        )
    )

    selected = [first]

    while len(selected) < quota:
        min_dist = np.min(
            distances[
                selected,
                :
            ],
            axis=0
        )

        # Avoid selecting an already-selected point.
        min_dist[selected] = -1.0

        next_pos = int(
            np.argmax(min_dist)
        )

        selected.append(
            next_pos
        )

    return [
        indices[pos]
        for pos in selected
    ]


def select_d7_sensors(
    data,
    log_fn=print
):
    """
    Select a geographically/network representative 30-node
    subset from the 228 D7 sensors.

    The previous implementation selected 22 of 30 sensors on
    I-405.  This implementation explicitly prevents a single
    freeway from dominating the GAT/LSTM input.

    The original 228-column order is preserved in the returned
    selected_indices list.
    """

    if data.shape[1] != 228:
        raise ValueError(
            "D7 sensor selection expects exactly 228 columns. "
            f"Got {data.shape[1]}."
        )

    meta = load_d7_metadata(
        log_fn
    )

    if len(meta) != data.shape[1]:
        raise ValueError(
            "D7 metadata/data mismatch: "
            f"{len(meta)} metadata rows vs "
            f"{data.shape[1]} traffic columns."
        )

    # --------------------------------------------------------
    # Stable freeway labels.
    # --------------------------------------------------------
    meta = meta.copy()

    meta["FwyNorm"] = meta["Fwy"].apply(
        _normalise_freeway
    )

    # Remove rows with unusable coordinates.
    valid_coord = (
        np.isfinite(
            meta["Latitude"]
        ) &
        np.isfinite(
            meta["Longitude"]
        )
    )

    if not valid_coord.all():
        bad = int(
            (~valid_coord).sum()
        )

        emit(
            log_fn,
            f"[WARN] {bad} D7 metadata rows have invalid "
            "coordinates."
        )

    meta = meta[
        valid_coord
    ].copy()

    if len(meta) < SUBSET_NODES:
        raise ValueError(
            f"Only {len(meta)} D7 stations have valid "
            f"coordinates; {SUBSET_NODES} are required."
        )

    groups = {
        freeway: list(group.index)
        for freeway, group in meta.groupby(
            "FwyNorm",
            sort=True
        )
    }

    groups = {
        freeway: idxs
        for freeway, idxs in groups.items()
        if len(idxs) > 0
    }

    if not groups:
        raise ValueError(
            "No D7 freeway groups could be formed."
        )

    target = min(
        SUBSET_NODES,
        len(meta)
    )

    freeway_names = sorted(
        groups.keys()
    )

    emit(
        log_fn,
        "[INFO] D7 freeways represented in "
        f"228-station lookup: {freeway_names}"
    )

    # --------------------------------------------------------
    # Step 1: guarantee at least one station per freeway,
    # whenever the number of freeways does not exceed target.
    # --------------------------------------------------------
    quotas = {
        freeway: 0
        for freeway in freeway_names
    }

    if len(freeway_names) <= target:
        for freeway in freeway_names:
            quotas[freeway] = 1

    remaining = target - sum(
        quotas.values()
    )

    # --------------------------------------------------------
    # Step 2: distribute remaining slots using sqrt(count).
    #
    # This intentionally reduces the dominance of large
    # freeways such as I-405 while still giving larger networks
    # more representation than tiny groups.
    # --------------------------------------------------------
    while remaining > 0:
        candidates = [
            freeway
            for freeway in freeway_names
            if (
                quotas[freeway] <
                min(
                    len(groups[freeway]),
                    D7_MAX_SENSORS_PER_FREEWAY
                )
            )
        ]

        if not candidates:
            break

        best = max(
            candidates,
            key=lambda freeway: (
                math.sqrt(
                    len(groups[freeway])
                ) /
                max(
                    quotas[freeway],
                    1
                ),
                len(groups[freeway])
            )
        )

        quotas[best] += 1
        remaining -= 1

    # If the cap made it impossible to reach target, fill any
    # remaining capacity from groups with unused stations.
    if remaining > 0:
        candidates = [
            freeway
            for freeway in freeway_names
            if quotas[freeway] < len(groups[freeway])
        ]

        while remaining > 0 and candidates:
            best = max(
                candidates,
                key=lambda freeway: (
                    len(groups[freeway]) -
                    quotas[freeway]
                )
            )

            quotas[best] += 1
            remaining -= 1

            candidates = [
                freeway
                for freeway in freeway_names
                if quotas[freeway] < len(groups[freeway])
            ]

    # --------------------------------------------------------
    # Step 3: spatially spread stations inside each freeway.
    # --------------------------------------------------------
    selected_indices = []

    for freeway in freeway_names:
        quota = quotas[freeway]

        if quota <= 0:
            continue

        group_df = meta.loc[
            groups[freeway]
        ]

        selected = _spatially_spread_indices(
            group_df,
            quota
        )

        selected_indices.extend(
            selected
        )

    # Preserve original traffic-matrix ordering.
    selected_indices = sorted(
        selected_indices
    )

    # Safety check.
    if len(selected_indices) != target:
        raise RuntimeError(
            "D7 sensor selection produced "
            f"{len(selected_indices)} sensors; "
            f"{target} were required."
        )

    selected_meta = meta.loc[
        selected_indices
    ].copy()

    # Add original traffic column number.
    selected_meta["traffic_column"] = [
        int(i)
        for i in selected_indices
    ]

    # --------------------------------------------------------
    # Save selection for reproducibility.
    # --------------------------------------------------------
    selection_payload = {
        "dataset": "D7",
        "total_source_sensors": int(data.shape[1]),
        "selected_sensors": int(len(selected_indices)),
        "max_sensors_per_freeway": int(
            D7_MAX_SENSORS_PER_FREEWAY
        ),
        "selected_indices": [
            int(i)
            for i in selected_indices
        ],
        "freeway_distribution": {
            freeway: int(
                quotas.get(freeway, 0)
            )
            for freeway in freeway_names
            if quotas.get(freeway, 0) > 0
        },
        "stations": []
    }

    for _, row in selected_meta.iterrows():
        station = {
            "traffic_column": int(
                row["traffic_column"]
            ),
            "station_id": int(
                row["ID"]
            ),
            "freeway": str(
                row["FwyNorm"]
            ),
            "direction": str(
                row["Dir"]
            ),
            "latitude": float(
                row["Latitude"]
            ),
            "longitude": float(
                row["Longitude"]
            )
        }

        if "Name" in row and pd.notna(row["Name"]):
            station["name"] = str(
                row["Name"]
            )

        selection_payload[
            "stations"
        ].append(station)

    selection_path = os.path.join(
        OUT,
        "d7",
        "selected_sensors.json"
    )

    with open(
        selection_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            selection_payload,
            f,
            indent=4
        )

    # Also save the selected metadata as CSV.
    selected_meta_path = os.path.join(
        OUT,
        "d7",
        "selected_station_metadata.csv"
    )

    selected_meta.to_csv(
        selected_meta_path,
        index=False
    )

    emit(
        log_fn,
        f"[INFO] D7 selected {len(selected_indices)} "
        "sensors across the freeway network."
    )

    emit(
        log_fn,
        "[INFO] Selected freeway distribution: "
        + str(
            selection_payload[
                "freeway_distribution"
            ]
        )
    )

    emit(
        log_fn,
        f"[INFO] D7 sensor selection saved: "
        f"{selection_path}"
    )

    return (
        selected_indices,
        selected_meta
    )


# ============================================================
# SUBSET
# ============================================================
def subset_data(dataset, log_fn=print):
    dataset = dataset.lower()

    ensure_dir(dataset)

    data_path = os.path.join(
        OUT,
        dataset,
        "data.npy"
    )

    if not os.path.exists(data_path):
        emit(
            log_fn,
            "[WARN] data.npy missing."
        )

        convert_data(
            dataset,
            log_fn=log_fn
        )

    data = np.load(
        data_path
    ).astype(
        np.float32
    )

    emit(
        log_fn,
        f"[INFO] Before subset: {data.shape}"
    )

    rows = min(
        SUBSET_ROWS,
        data.shape[0]
    )

    # --------------------------------------------------------
    # D7: use metadata-driven spatial/network selection.
    # --------------------------------------------------------
    if dataset == "d7":
        selected_indices, selected_meta = (
            select_d7_sensors(
                data,
                log_fn=log_fn
            )
        )

        subset = data[
            :rows,
            selected_indices
        ]

        # Save the mapping in the exact order used by the model.
        model_mapping = []

        for model_index, source_index in enumerate(
            selected_indices
        ):
            row = selected_meta[
                selected_meta["traffic_column"] ==
                source_index
            ].iloc[0]

            model_mapping.append(
                {
                    "model_sensor": int(
                        model_index
                    ),
                    "traffic_column": int(
                        source_index
                    ),
                    "station_id": int(
                        row["ID"]
                    ),
                    "freeway": str(
                        row["FwyNorm"]
                    ),
                    "direction": str(
                        row["Dir"]
                    ),
                    "latitude": float(
                        row["Latitude"]
                    ),
                    "longitude": float(
                        row["Longitude"]
                    )
                }
            )

        mapping_path = os.path.join(
            OUT,
            dataset,
            "sensor_mapping.json"
        )

        with open(
            mapping_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                model_mapping,
                f,
                indent=4
            )

        emit(
            log_fn,
            f"[INFO] D7 model sensor mapping saved: "
            f"{mapping_path}"
        )

    elif dataset == "seattle":
        sensor_names = _load_seattle_sensor_names(
            log_fn
        )

        selected_indices, selected_meta = (
            select_seattle_sensors(
                data,
                sensor_names,
                log_fn=log_fn
            )
        )

        subset = data[
            :rows,
            selected_indices
        ]

    else:
        nodes = min(
            SUBSET_NODES,
            data.shape[1]
        )

        selected_indices = list(
            range(nodes)
        )

        subset = data[
            :rows,
            :nodes
        ]

    subset = np.nan_to_num(
        subset,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(
        np.float32
    )

    emit(
        log_fn,
        f"[INFO] Subset shape: {subset.shape}"
    )

    subset_path = os.path.join(
        OUT,
        dataset,
        "subset.npy"
    )

    np.save(
        subset_path,
        subset
    )

    emit(
        log_fn,
        f"[INFO] Subset saved: {subset_path}"
    )

    return subset


# ============================================================
# TRAIN MODEL
# ============================================================
def train_model(dataset, log_fn=print):
    dataset = dataset.lower()
    # ============================================================
    # REPRODUCIBILITY
    # ============================================================

    SEED = 42

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    emit(log_fn, f"[INFO] Seed set to {SEED}")

    ensure_dir(dataset)

    subset_path = os.path.join(
        OUT,
        dataset,
        "subset.npy"
    )

    if not os.path.exists(subset_path):
        raise FileNotFoundError(
            f"Subset not found: {subset_path}. "
            "Run subset_data() first."
        )

    data = np.load(
        subset_path
    ).astype(
        np.float32
    )

    emit(
        log_fn,
        f"[INFO] Training data shape: {data.shape}"
    )

    if data.shape[1] != SUBSET_NODES:
        raise ValueError(
            "Current GAT_LSTM configuration expects "
            f"{SUBSET_NODES} nodes, but subset contains "
            f"{data.shape[1]}."
        )

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------
    emit(
        log_fn,
        "[STEP] Normalizing training data..."
    )

    data_norm, mean, std = normalize(
        data
    )

    mean = np.asarray(
        mean,
        dtype=np.float32
    )

    std = np.asarray(
        std,
        dtype=np.float32
    )

    std = np.where(
        std < 1e-6,
        1.0,
        std
    )

    np.save(
        os.path.join(
            OUT,
            dataset,
            "mean.npy"
        ),
        mean
    )

    np.save(
        os.path.join(
            OUT,
            dataset,
            "std.npy"
        ),
        std
    )

    emit(
        log_fn,
        f"[INFO] Saved mean: {OUT}/{dataset}/mean.npy"
    )

    emit(
        log_fn,
        f"[INFO] Saved std: {OUT}/{dataset}/std.npy"
    )

    # --------------------------------------------------------
    # SEQUENCES
    # --------------------------------------------------------
    X, Y = create_seq(
        data_norm,
        seq_len=SEQ_LEN
    )

    emit(
        log_fn,
        f"[INFO] X shape: {X.shape}"
    )

    emit(
        log_fn,
        f"[INFO] Y shape: {Y.shape}"
    )

    if len(X) == 0:
        raise ValueError(
            "No training sequences were generated."
        )

    X = torch.tensor(
        X,
        dtype=torch.float32
    )

    Y = torch.tensor(
        Y,
        dtype=torch.float32
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------
    emit(
        log_fn,
        f"[INFO] Creating GAT + LSTM for "
        f"{data.shape[1]} nodes."
    )

    model = GAT_LSTM()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    loss_fn = torch.nn.MSELoss()

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------
    model.train()

    for epoch in range(EPOCHS):
        permutation = torch.randperm(
            X.size(0)
        )

        total_loss = 0.0
        batch_count = 0

        for start in range(
            0,
            X.size(0),
            BATCH_SIZE
        ):
            indices = permutation[
                start:start + BATCH_SIZE
            ]

            xb = X[indices]
            yb = Y[indices]

            optimizer.zero_grad()

            output = model(
                xb
            )

            loss = loss_fn(
                output,
                yb
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        average_loss = (
            total_loss /
            max(batch_count, 1)
        )

        emit(
            log_fn,
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- Loss: {average_loss:.6f}"
        )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------
    model_path = os.path.join(
        OUT,
        dataset,
        "model.pth"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    metadata = {
        "dataset": dataset,
        "nodes": int(data.shape[1]),
        "seq_len": SEQ_LEN,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE
    }

    if dataset == "d7":
        metadata[
            "source_sensors"
        ] = 228

        metadata[
            "selection_file"
        ] = os.path.join(
            OUT,
            "d7",
            "selected_sensors.json"
        )

    if dataset == "seattle":
        metadata[
            "selection_file"
        ] = os.path.join(
            OUT,
            "seattle",
            "selected_sensors.json"
        )

        metadata[
            "coordinate_lookup"
        ] = SEATTLE_COORD_LOOKUP

    with open(
        os.path.join(
            OUT,
            dataset,
            "model_metadata.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=4
        )

    emit(
        log_fn,
        f"[INFO] Model saved: {model_path}"
    )

    return model_path


# ============================================================
# LOAD MODEL SENSOR LOCATIONS
# ============================================================
def load_sensor_locations(
    dataset,
    N,
    log_fn=print
):
    dataset = dataset.lower()

    # --------------------------------------------------------
    # D7 / SEATTLE real coordinates
    # --------------------------------------------------------
    if dataset in {"d7", "seattle"}:
        mapping_path = os.path.join(
            OUT,
            dataset,
            "sensor_mapping.json"
        )

        if not os.path.exists(mapping_path):
            raise FileNotFoundError(
                f"{dataset.upper()} sensor mapping not found: "
                f"{mapping_path}. "
                f"Run subset_data('{dataset}') first."
            )

        with open(
            mapping_path,
            "r",
            encoding="utf-8"
        ) as f:
            mapping = json.load(f)

        if len(mapping) != N:
            raise ValueError(
                f"{dataset.upper()} mapping contains "
                f"{len(mapping)} sensors but prediction "
                f"contains {N}."
            )

        coords = np.asarray(
            [
                [
                    float(item["latitude"]),
                    float(item["longitude"])
                ]
                for item in mapping
            ],
            dtype=np.float64
        )

        emit(
            log_fn,
            f"[INFO] {dataset.upper()} real station "
            f"coordinates used: {len(coords)}"
        )

        return coords

    # --------------------------------------------------------
    # BAY REAL COORDINATES
    # --------------------------------------------------------
    if dataset == "bay":
        coord_path = os.path.join(
            "data",
            "bay",
            "graph_sensor_locations_bay.csv"
        )

        if os.path.exists(coord_path):
            try:
                coords_df = pd.read_csv(
                    coord_path,
                    header=None
                )

                if coords_df.shape[1] < 3:
                    raise ValueError(
                        "BAY coordinate file must contain "
                        "at least 3 columns."
                    )

                coords = coords_df.iloc[
                    :N,
                    1:3
                ].apply(
                    pd.to_numeric,
                    errors="coerce"
                ).to_numpy(
                    dtype=np.float64
                )

                valid = np.isfinite(
                    coords
                ).all(axis=1)

                coords = coords[
                    valid
                ]

                if len(coords) < N:
                    raise ValueError(
                        f"Only {len(coords)} valid coordinates "
                        f"found; {N} required."
                    )

                emit(
                    log_fn,
                    f"[INFO] Real BAY coordinates used: "
                    f"{N}"
                )

                return coords[:N]

            except Exception as e:
                emit(
                    log_fn,
                    f"[WARN] BAY coordinate read failed: {e}"
                )

    # --------------------------------------------------------
    # No synthetic coordinates.
    # --------------------------------------------------------
    raise ValueError(
        f"No real sensor-coordinate mapping is available "
        f"for {dataset.upper()}."
    )


# ============================================================
# LOAD SENSOR METADATA FOR HEATMAP
# ============================================================
def load_heatmap_sensor_metadata(
    dataset,
    N,
    log_fn=print
):
    dataset = dataset.lower()

    if dataset == "d7":
        mapping_path = os.path.join(
            OUT,
            "d7",
            "sensor_mapping.json"
        )

        if not os.path.exists(mapping_path):
            raise FileNotFoundError(
                f"D7 sensor mapping not found: {mapping_path}"
            )

        with open(
            mapping_path,
            "r",
            encoding="utf-8"
        ) as f:
            mapping = json.load(f)

        if len(mapping) != N:
            raise ValueError(
                f"D7 sensor mapping has {len(mapping)} "
                f"entries; expected {N}."
            )

        return mapping

    if dataset == "seattle":
        mapping_path = os.path.join(
            OUT,
            "seattle",
            "sensor_mapping.json"
        )

        if not os.path.exists(mapping_path):
            raise FileNotFoundError(
                f"Seattle sensor mapping not found: "
                f"{mapping_path}"
            )

        with open(
            mapping_path,
            "r",
            encoding="utf-8"
        ) as f:
            mapping = json.load(f)

        if len(mapping) != N:
            raise ValueError(
                f"Seattle sensor mapping has {len(mapping)} "
                f"entries; expected {N}."
            )

        return mapping

    # Generic metadata for BAY.
    return [
        {
            "model_sensor": int(i)
        }
        for i in range(N)
    ]


# ============================================================
# CONGESTION SCORE
# ============================================================
def compute_congestion(
    speed,
    free_flow
):
    """
    Convert speed into a transparent relative congestion score.

    ratio = current speed / free-flow speed

    Relative mapping:
        >= 90% -> 0.00
        75%    -> 0.50
        60%    -> 0.67
        50%    -> 1.00

    This score represents congestion severity relative to the
    sensor's own free-flow speed. It is NOT prediction error.
    """

    speed = np.asarray(
        speed,
        dtype=np.float32
    )

    free_flow = np.asarray(
        free_flow,
        dtype=np.float32
    )

    free_flow = np.maximum(
        free_flow,
        1.0
    )

    ratio = (
        speed /
        free_flow
    )

    congestion = np.interp(
        ratio,
        [
            0.50,
            0.60,
            0.75,
            0.90,
            1.00
        ],
        [
            1.00,
            0.67,
            0.50,
            0.15,
            0.00
        ]
    )

    congestion = np.clip(
        congestion,
        0.0,
        1.0
    )

    return congestion.astype(
        np.float32
    )


# ============================================================
# CONGESTION LEVEL
# ============================================================
def congestion_level(value):
    if value >= 0.75:
        return "Severe"

    if value >= 0.50:
        return "High"

    if value >= 0.25:
        return "Moderate"

    return "Low"


# ============================================================
# HEATMAP FRAME GENERATION
# ============================================================
def create_heatmap_frames(
    dataset,
    pred,
    actual,
    log_fn=print
):
    dataset = dataset.lower()

    ensure_dir(dataset)

    N = pred.shape[1]

    coords = load_sensor_locations(
        dataset,
        N,
        log_fn=log_fn
    )

    if len(coords) != N:
        raise ValueError(
            f"Coordinate count {len(coords)} "
            f"does not match sensor count {N}"
        )

    sensor_metadata = load_heatmap_sensor_metadata(
        dataset,
        N,
        log_fn=log_fn
    )

    # --------------------------------------------------------
    # Free-flow speed
    # --------------------------------------------------------
    subset_path = os.path.join(
        OUT,
        dataset,
        "subset.npy"
    )

    if os.path.exists(subset_path):
        subset = np.load(
            subset_path
        ).astype(
            np.float32
        )

        free_flow = np.percentile(
            subset,
            85,
            axis=0
        )
    else:
        free_flow = np.percentile(
            actual,
            85,
            axis=0
        )

    free_flow = np.maximum(
        free_flow,
        1.0
    )

    center_lat = float(
        np.mean(coords[:, 0])
    )

    center_lon = float(
        np.mean(coords[:, 1])
    )

    # --------------------------------------------------------
    # Frames
    # --------------------------------------------------------
    frames = []

    num_frames = min(
        TEST_POINTS,
        pred.shape[0],
        actual.shape[0]
    )

    for t in range(num_frames):
        predicted_speed = np.asarray(
            pred[t],
            dtype=np.float32
        )

        actual_speed = np.asarray(
            actual[t],
            dtype=np.float32
        )

        predicted_congestion = compute_congestion(
            predicted_speed,
            free_flow
        )

        actual_congestion = compute_congestion(
            actual_speed,
            free_flow
        )

        predicted_points = []
        actual_points = []

        predicted_sensors = []
        actual_sensors = []

        for i in range(N):
            lat = float(
                coords[i][0]
            )

            lon = float(
                coords[i][1]
            )

            pred_speed_i = float(
                predicted_speed[i]
            )

            actual_speed_i = float(
                actual_speed[i]
            )

            pred_cong_i = float(
                predicted_congestion[i]
            )

            actual_cong_i = float(
                actual_congestion[i]
            )

            meta = sensor_metadata[i]

            # ------------------------------------------------
            # Leaflet heat layer
            # ------------------------------------------------
            predicted_points.append(
                [
                    lat,
                    lon,
                    pred_cong_i
                ]
            )

            actual_points.append(
                [
                    lat,
                    lon,
                    actual_cong_i
                ]
            )

            # ------------------------------------------------
            # Common station metadata
            # ------------------------------------------------
            pred_sensor = {
                "sensor": int(i),
                "latitude": lat,
                "longitude": lon,
                "speed": round(
                    pred_speed_i,
                    2
                ),
                "free_flow": round(
                    float(free_flow[i]),
                    2
                ),
                "speed_reduction_percent": round(
                    max(
                        0.0,
                        (
                            1.0 -
                            pred_speed_i /
                            max(
                                float(free_flow[i]),
                                1.0
                            )
                        )
                    ) * 100.0,
                    1
                ),
                "congestion": round(
                    pred_cong_i,
                    4
                ),
                "level": congestion_level(
                    pred_cong_i
                )
            }

            actual_sensor = {
                "sensor": int(i),
                "latitude": lat,
                "longitude": lon,
                "speed": round(
                    actual_speed_i,
                    2
                ),
                "free_flow": round(
                    float(free_flow[i]),
                    2
                ),
                "speed_reduction_percent": round(
                    max(
                        0.0,
                        (
                            1.0 -
                            actual_speed_i /
                            max(
                                float(free_flow[i]),
                                1.0
                            )
                        )
                    ) * 100.0,
                    1
                ),
                "congestion": round(
                    actual_cong_i,
                    4
                ),
                "level": congestion_level(
                    actual_cong_i
                )
            }

            # D7 / Seattle station details.
            if dataset == "d7":
                for item in (
                    pred_sensor,
                    actual_sensor
                ):
                    item["station_id"] = int(
                        meta["station_id"]
                    )

                    item["freeway"] = str(
                        meta["freeway"]
                    )

                    item["direction"] = str(
                        meta["direction"]
                    )

                    if "name" in meta:
                        item["name"] = str(
                            meta["name"]
                        )

            elif dataset == "seattle":
                for item in (
                    pred_sensor,
                    actual_sensor
                ):
                    item["sensor_name"] = str(
                        meta["sensor_name"]
                    )

                    if "roadway" in meta:
                        item["roadway"] = str(
                            meta["roadway"]
                        )

                    if "location" in meta:
                        item["location"] = str(
                            meta["location"]
                        )

                    if "milepost" in meta:
                        item["milepost"] = float(
                            meta["milepost"]
                        )

                    if "wsdot_name" in meta:
                        item["wsdot_name"] = str(
                            meta["wsdot_name"]
                        )

            predicted_sensors.append(
                pred_sensor
            )

            actual_sensors.append(
                actual_sensor
            )

        # Frame statistics.
        predicted_stats = {
            "avg_speed": round(
                float(np.mean(predicted_speed)),
                2
            ),
            "min_speed": round(
                float(np.min(predicted_speed)),
                2
            ),
            "max_speed": round(
                float(np.max(predicted_speed)),
                2
            ),
            "avg_congestion": round(
                float(np.mean(predicted_congestion)),
                4
            ),
            "max_congestion": round(
                float(np.max(predicted_congestion)),
                4
            )
        }

        actual_stats = {
            "avg_speed": round(
                float(np.mean(actual_speed)),
                2
            ),
            "min_speed": round(
                float(np.min(actual_speed)),
                2
            ),
            "max_speed": round(
                float(np.max(actual_speed)),
                2
            ),
            "avg_congestion": round(
                float(np.mean(actual_congestion)),
                4
            ),
            "max_congestion": round(
                float(np.max(actual_congestion)),
                4
            )
        }

        frames.append(
            {
                "timestep": int(t),
                "predicted": predicted_points,
                "actual": actual_points,
                "predicted_sensors": predicted_sensors,
                "actual_sensors": actual_sensors,
                "predicted_stats": predicted_stats,
                "actual_stats": actual_stats
            }
        )

    # --------------------------------------------------------
    # Representative frame
    #
    # The viewer no longer exposes a slider.  Select the frame
    # with the highest actual average congestion so that the
    # static map shows a meaningful traffic state.
    # --------------------------------------------------------
    if frames:
        representative_frame = max(
            frames,
            key=lambda frame:
                frame["actual_stats"]["avg_congestion"]
        )

        representative_timestep = int(
            representative_frame["timestep"]
        )
    else:
        representative_timestep = 0

    heatmap = {
        "dataset": dataset.upper(),
        "num_sensors": int(N),
        "num_frames": int(len(frames)),
        "representative_timestep": representative_timestep,
        "center": {
            "lat": center_lat,
            "lon": center_lon
        },
        "free_flow_speed": [
            float(x)
            for x in free_flow
        ],
        "sensor_location_definition":
            (
                "Seattle uses real detector coordinates matched "
                "from the local WSDOT WebFlow lookup; D7 uses the "
                "228-station geographic lookup. No synthetic "
                "sensor grid is used."
            ),
        "congestion_definition": {
            "description":
                "Congestion is based on reduction from "
                "each sensor's free-flow speed.",
            "formula":
                "speed / free_flow followed by the "
                "relative-speed interpolation. The same "
                "per-sensor calculation is applied to both "
                "predicted and actual speeds.",
            "levels": {
                "Low": "0.00 - <0.25",
                "Moderate": "0.25 - <0.50",
                "High": "0.50 - <0.75",
                "Severe": "0.75 - 1.00"
            }
        },
        "frames": frames
    }

    heatmap_path = os.path.join(
        STATIC,
        f"heatmap_{dataset}.json"
    )

    with open(
        heatmap_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            heatmap,
            f
        )

    emit(
        log_fn,
        f"[INFO] Heatmap generated: {heatmap_path}"
    )

    emit(
        log_fn,
        f"[INFO] Sensors per frame: {N}"
    )

    emit(
        log_fn,
        f"[INFO] Representative heatmap timestep: "
        f"{representative_timestep}"
    )

    return heatmap_path


# ============================================================
# TEST MODEL
# ============================================================
def test_model(dataset, log_fn=print):
    dataset = dataset.lower()

    ensure_dir(dataset)

    subset_path = os.path.join(
        OUT,
        dataset,
        "subset.npy"
    )

    if not os.path.exists(subset_path):
        raise FileNotFoundError(
            f"Subset not found: {subset_path}"
        )

    emit(
        log_fn,
        "[INFO] Loading test data..."
    )

    data = np.load(
        subset_path
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # TRAINING NORMALIZATION PARAMETERS
    # --------------------------------------------------------
    mean_path = os.path.join(
        OUT,
        dataset,
        "mean.npy"
    )

    std_path = os.path.join(
        OUT,
        dataset,
        "std.npy"
    )

    if not os.path.exists(mean_path):
        raise FileNotFoundError(
            f"Missing normalization file: {mean_path}"
        )

    if not os.path.exists(std_path):
        raise FileNotFoundError(
            f"Missing normalization file: {std_path}"
        )

    mean = np.load(
        mean_path
    ).astype(
        np.float32
    )

    std = np.load(
        std_path
    ).astype(
        np.float32
    )

    std = np.where(
        std < 1e-6,
        1.0,
        std
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Use the SAME mean/std learned during training.
    #
    # Do NOT call normalize(data) here because that would
    # recompute normalization from the test data.
    # --------------------------------------------------------
    data_norm = (
        data - mean
    ) / std

    data_norm = np.nan_to_num(
        data_norm,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # SEQUENCES
    # --------------------------------------------------------
    X, Y = create_seq(
        data_norm,
        seq_len=SEQ_LEN
    )

    if len(X) < TEST_POINTS:
        raise ValueError(
            f"Only {len(X)} test sequences available; "
            f"{TEST_POINTS} required."
        )

    X_test = torch.tensor(
        X[-TEST_POINTS:],
        dtype=torch.float32
    )

    Y_test_norm = Y[
        -TEST_POINTS:
    ]

    emit(
        log_fn,
        f"[INFO] Test X shape: {X_test.shape}"
    )

    emit(
        log_fn,
        f"[INFO] Test Y shape: {Y_test_norm.shape}"
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------
    model_path = os.path.join(
        OUT,
        dataset,
        "model.pth"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    emit(
        log_fn,
        f"[INFO] Loading model configured for "
        f"{data.shape[1]} nodes..."
    )

    model = GAT_LSTM()

    state = torch.load(
        model_path,
        map_location="cpu"
    )

    model.load_state_dict(
        state
    )

    model.eval()

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------
    emit(
        log_fn,
        "[INFO] Generating predictions..."
    )

    with torch.no_grad():
        pred_norm = model(
            X_test
        ).cpu().numpy()

    # --------------------------------------------------------
    # DE-NORMALIZE
    # --------------------------------------------------------
    pred = denormalize(
        pred_norm,
        mean,
        std
    )

    actual = denormalize(
        Y_test_norm,
        mean,
        std
    )

    pred = np.asarray(
        pred,
        dtype=np.float32
    )

    actual = np.asarray(
        actual,
        dtype=np.float32
    )

    emit(
        log_fn,
        f"[INFO] Prediction shape: {pred.shape}"
    )

    emit(
        log_fn,
        f"[INFO] Actual shape: {actual.shape}"
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------
    mae = float(
        np.mean(
            np.abs(
                pred - actual
            )
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (pred - actual) ** 2
            )
        )
    )

    emit(
        log_fn,
        f"[RESULT] MAE: {mae:.4f}"
    )

    emit(
        log_fn,
        f"[RESULT] RMSE: {rmse:.4f}"
    )

    # --------------------------------------------------------
    # PREDICTION PLOT
    # --------------------------------------------------------
    plot_path = os.path.join(
        STATIC,
        f"{dataset}_plot.png"
    )

    plt.figure(
        figsize=(12, 7)
    )

    num_plot_sensors = min(
        3,
        pred.shape[1]
    )

    for i in range(num_plot_sensors):
        plt.plot(
            actual[:, i],
            label=f"Actual Sensor {i}"
        )

        plt.plot(
            pred[:, i],
            "--",
            label=f"Predicted Sensor {i}"
        )

    plt.title(
        f"{dataset.upper()} - Actual vs Predicted"
    )

    plt.xlabel(
        "Forecast Time Step"
    )

    plt.ylabel(
        "Traffic Speed (mph)"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        plot_path,
        dpi=120
    )

    plt.close()

    emit(
        log_fn,
        f"[INFO] Prediction plot saved: {plot_path}"
    )

    # --------------------------------------------------------
    # HEATMAP
    # --------------------------------------------------------
    emit(
        log_fn,
        "[STEP] Computing congestion hotspots..."
    )

    heatmap_path = create_heatmap_frames(
        dataset,
        pred,
        actual,
        log_fn=log_fn
    )

    emit(
        log_fn,
        "[DONE] Prediction plot and heatmap generated."
    )

    return (
        pred,
        actual,
        mae,
        rmse,
        plot_path,
        heatmap_path
    )


# ============================================================
# DIRECT TEST
# ============================================================
if __name__ == "__main__":
    print(
        "pipeline.py loaded successfully."
    )
    print(
        "Use app.py to run the complete Flask pipeline."
    )
