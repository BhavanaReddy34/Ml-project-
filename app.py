from flask import (
    Flask,
    render_template,
    Response,
    jsonify,
    stream_with_context
)

import os
import time
import traceback
import json

from pipeline import (
    convert_data,
    subset_data,
    train_model,
    test_model
)


app = Flask(__name__)


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():
    return render_template(
        "index.html"
    )


# ============================================================
# PIPELINE STREAM
# ============================================================

@app.route("/run_stream/<dataset>")
def run_stream(dataset):

    dataset = dataset.lower()

    valid_datasets = {
        "bay",
        "d7",
        "seattle"
    }

    if dataset not in valid_datasets:
        return Response(
            "[ERROR] Unsupported dataset\n",
            mimetype="text/plain"
        )

    @stream_with_context
    def generate():

        try:

            yield (
                f"[START] Pipeline requested for "
                f"{dataset.upper()}\n"
            )

            yield (
                f"[START] Traffic prediction pipeline "
                f"started for {dataset.upper()}\n"
            )

            def log_fn(message):
                print(
                    message,
                    flush=True
                )

            # ------------------------------------------------
            # STEP 1
            # ------------------------------------------------

            yield (
                "[STEP 1/4] Data Conversion "
                "in progress...\n"
            )

            convert_data(
                dataset,
                log_fn=log_fn
            )

            yield (
                "[DONE] Data Conversion completed.\n"
            )

            # ------------------------------------------------
            # STEP 2
            # ------------------------------------------------

            yield (
                "[STEP 2/4] Subset of data "
                "being extracted...\n"
            )

            subset_data(
                dataset,
                log_fn=log_fn
            )

            yield (
                "[DONE] Data subset extraction "
                "completed.\n"
            )

            # ------------------------------------------------
            # STEP 3
            # ------------------------------------------------

            yield (
                f"[STEP 3/4] Model training on "
                f"{dataset.upper()}...\n"
            )

            train_model(
                dataset,
                log_fn=log_fn
            )

            yield (
                "[DONE] Model training completed.\n"
            )

            # ------------------------------------------------
            # STEP 4
            # ------------------------------------------------

            yield (
                f"[STEP 4/4] Testing on "
                f"{dataset.upper()}...\n"
            )

            (
                pred,
                actual,
                mae,
                rmse,
                plot_path,
                heatmap_path
            ) = test_model(
                dataset,
                log_fn=log_fn
            )

            yield (
                f"[RESULT] MAE = {mae:.4f}\n"
            )

            yield (
                f"[RESULT] RMSE = {rmse:.4f}\n"
            )

            yield (
                f"[RESULT] Prediction plot: "
                f"{plot_path}\n"
            )

            yield (
                f"[RESULT] Heatmap data: "
                f"{heatmap_path}\n"
            )

            yield (
                "[SUCCESS] Complete GAT + LSTM "
                "pipeline finished successfully.\n"
            )

        except Exception as e:

            traceback.print_exc()

            print(
                "[ERROR]",
                repr(e),
                flush=True
            )

            yield (
                f"[ERROR] {str(e)}\n"
            )

    response = Response(
        generate(),
        mimetype="text/plain"
    )

    response.headers[
        "Cache-Control"
    ] = "no-cache"

    response.headers[
        "X-Accel-Buffering"
    ] = "no"

    response.headers[
        "Connection"
    ] = "keep-alive"

    return response


# ============================================================
# RESULTS
# ============================================================

@app.route("/results/<dataset>")
def results(dataset):

    dataset = dataset.lower()

    plot_file = os.path.join(
        "static",
        f"{dataset}_plot.png"
    )

    heatmap_file = os.path.join(
        "static",
        f"heatmap_{dataset}.json"
    )

    if not os.path.exists(plot_file):
        return jsonify({
            "error": "Prediction plot not found"
        }), 404

    if not os.path.exists(heatmap_file):
        return jsonify({
            "error": "Heatmap data not found"
        }), 404

    timestamp = int(
        time.time() * 1000
    )

    return jsonify({

        "plot":
            f"/static/{dataset}_plot.png"
            f"?t={timestamp}",

        "heatmap":
            f"/build_heatmap/{dataset}"
            f"?t={timestamp}"
    })


# ============================================================
# HEATMAP VIEWER
# ============================================================

@app.route("/build_heatmap/<dataset>")
def build_heatmap(dataset):
    """
    Render one representative congestion heatmap.

    The time-step slider has intentionally been removed because
    the 50 test points are closely spaced and did not provide a
    useful visual time series.

    The viewer automatically uses the representative timestep
    selected by pipeline.py: the frame with the highest actual
    average congestion.

    Predicted / Actual buttons remain so the user can compare
    the model output against the observed traffic state at the
    same representative timestep.
    """

    dataset = dataset.lower()

    valid_datasets = {
        "bay",
        "d7",
        "seattle"
    }

    if dataset not in valid_datasets:
        return (
            "<h3>Unsupported dataset.</h3>"
        ), 400

    heatmap_path = os.path.join(
        "static",
        f"heatmap_{dataset}.json"
    )

    if not os.path.exists(heatmap_path):
        return (
            "<h3>No heatmap data available.</h3>"
        ), 404

    with open(
        heatmap_path,
        "r",
        encoding="utf-8"
    ) as f:
        heatmap_data = f.read()

    # ------------------------------------------------------------
    # Use a plain template + replacement instead of a Python
    # f-string.  This prevents JavaScript { } blocks from causing
    # Python f-string syntax errors.
    # ------------------------------------------------------------
    html = r"""
<!DOCTYPE html>
<html>

<head>

<meta charset="UTF-8">

<title>
    __DATASET__ Traffic Congestion
</title>

<link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
/>

<script
    src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js">
</script>

<script
    src="https://unpkg.com/leaflet.heat/dist/leaflet-heat.js">
</script>

<style>

html,
body {
    margin: 0;
    padding: 0;
    height: 100%;
    font-family: Arial, sans-serif;
}

#controls {

    min-height: 142px;

    background: white;

    padding: 10px 14px;

    box-sizing: border-box;

    border-bottom: 1px solid #bbb;

}

#title {

    font-weight: bold;

    font-size: 18px;

    margin-right: 14px;

}

#summary {

    margin-top: 8px;

    font-size: 13px;

    line-height: 19px;

}

#stats {

    margin-top: 5px;

    display: flex;

    flex-wrap: wrap;

    gap: 7px 18px;

    font-size: 13px;

}

.stat {

    white-space: nowrap;

}

.stat b {

    color: #222;

}

#buttons {

    margin-top: 8px;

}

.modeButton {

    padding: 5px 12px;

    margin-right: 5px;

    cursor: pointer;

    border: 1px solid #777;

    background: #eeeeee;

    border-radius: 3px;

}

.modeButton.active {

    background: #17365d;

    color: white;

}

#map {

    width: 100%;

    height: calc(100% - 142px);

}

.legend {

    background: white;

    padding: 9px;

    line-height: 20px;

    color: #333;

    box-shadow: 0 1px 4px rgba(0,0,0,0.3);

}

.legend i {

    width: 18px;

    height: 18px;

    float: left;

    margin-right: 6px;

    opacity: 0.9;

}

.sensor-popup {

    font-size: 13px;

    line-height: 18px;

}

</style>

</head>

<body>

<div id="controls">

    <div>

        <span id="title">
            __DATASET__ Traffic Congestion
        </span>

        <span id="sensorCount"></span>

        <span id="frameCount"></span>

    </div>

    <div id="summary">

        Representative congestion state.

        <span id="frameLabel"></span>

    </div>

    <div id="stats"></div>

    <div id="buttons">

        <button
            id="predictedButton"
            class="modeButton active"
            onclick="setMode('predicted')"
        >
            Predicted
        </button>

        <button
            id="actualButton"
            class="modeButton"
            onclick="setMode('actual')"
        >
            Actual
        </button>

    </div>

</div>

<div id="map"></div>

<script>

const heatmapData =
    __DATA__;

const datasetName =
    "__DATASET_LOWER__";

let currentMode =
    "predicted";

let heatLayer =
    null;

let sensorLayer =
    null;

const map =
    L.map("map");


/* ============================================================
   BASE MAP
   ============================================================ */

L.tileLayer(
    "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    {
        maxZoom: 19,
        attribution:
            "&copy; OpenStreetMap contributors"
    }
).addTo(map);


/* ============================================================
   COLOUR / LABEL
   ============================================================ */

function getColor(value) {

    if (value >= 0.75) {
        return "#ff0000";
    }

    if (value >= 0.50) {
        return "#ff9900";
    }

    if (value >= 0.25) {
        return "#ffff00";
    }

    return "#00cc00";
}


function getLabel(value) {

    if (value >= 0.75) {
        return "Severe";
    }

    if (value >= 0.50) {
        return "High";
    }

    if (value >= 0.25) {
        return "Moderate";
    }

    return "Low";
}


function fmt(value, digits) {

    const n =
        Number(value);

    return Number.isFinite(n)
        ? n.toFixed(digits)
        : "-";
}


/* ============================================================
   DATA ACCESS
   ============================================================ */

function getSensors(frame) {

    if (currentMode === "predicted") {
        return frame.predicted_sensors || [];
    }

    return frame.actual_sensors || [];
}


function getHeatPoints(frame) {

    if (currentMode === "predicted") {
        return frame.predicted || [];
    }

    return frame.actual || [];
}


/* ============================================================
   REPRESENTATIVE FRAME
   ============================================================ */

function chooseRepresentativeFrame() {

    const frames =
        heatmapData.frames || [];

    if (frames.length === 0) {
        return 0;
    }

    /*
       pipeline.py already stores the selected representative
       timestep.  Prefer that value so the browser and pipeline
       cannot disagree.
    */

    if (
        heatmapData.representative_timestep !== undefined
    ) {

        const stored =
            Number(
                heatmapData.representative_timestep
            );

        if (
            Number.isInteger(stored) &&
            stored >= 0 &&
            stored < frames.length
        ) {
            return stored;
        }
    }

    /*
       Safe fallback if an older heatmap JSON is opened.
    */

    let bestIndex = 0;

    let bestScore = -Infinity;

    frames.forEach(
        function(frame, index) {

            const sensors =
                frame.actual_sensors || [];

            let total = 0;

            let count = 0;

            sensors.forEach(
                function(sensor) {

                    const value =
                        Number(
                            sensor.congestion
                        );

                    if (
                        Number.isFinite(value)
                    ) {

                        total += value;

                        count += 1;
                    }
                }
            );

            if (count > 0) {

                const score =
                    total / count;

                if (
                    score > bestScore
                ) {

                    bestScore =
                        score;

                    bestIndex =
                        index;
                }
            }
        }
    );

    return bestIndex;
}


const representativeFrameIndex =
    chooseRepresentativeFrame();


/* ============================================================
   HEADER / STATS
   ============================================================ */

function updateHeader(frame) {

    const sensors =
        getSensors(frame);

    const frameCount =
        heatmapData.frames
            ? heatmapData.frames.length
            : 0;

    document.getElementById(
        "sensorCount"
    ).innerText =
        "Sensors: " +
        sensors.length;

    document.getElementById(
        "frameCount"
    ).innerText =
        " | Available test frames: " +
        frameCount;

    document.getElementById(
        "frameLabel"
    ).innerText =
        " | Representative timestep: " +
        frame.timestep +
        " | " +
        (
            currentMode === "predicted"
                ? "Predicted"
                : "Actual"
        );
}


function updateStats(sensors) {

    if (
        !sensors ||
        sensors.length === 0
    ) {

        document.getElementById(
            "stats"
        ).innerHTML =
            "<span class='stat'>" +
            "<b>No sensor data available</b>" +
            "</span>";

        return;
    }

    let speedTotal = 0;

    let freeFlowTotal = 0;

    let congestionTotal = 0;

    let minSpeed = Infinity;

    let maxCongestion = -Infinity;

    let low = 0;

    let moderate = 0;

    let high = 0;

    let severe = 0;


    sensors.forEach(
        function(sensor) {

            const speed =
                Number(sensor.speed);

            const freeFlow =
                Number(sensor.free_flow);

            const congestion =
                Number(sensor.congestion);


            if (
                Number.isFinite(speed)
            ) {

                speedTotal += speed;

                minSpeed =
                    Math.min(
                        minSpeed,
                        speed
                    );
            }


            if (
                Number.isFinite(freeFlow)
            ) {

                freeFlowTotal +=
                    freeFlow;
            }


            if (
                Number.isFinite(congestion)
            ) {

                congestionTotal +=
                    congestion;

                maxCongestion =
                    Math.max(
                        maxCongestion,
                        congestion
                    );


                if (
                    congestion >= 0.75
                ) {

                    severe += 1;

                } else if (
                    congestion >= 0.50
                ) {

                    high += 1;

                } else if (
                    congestion >= 0.25
                ) {

                    moderate += 1;

                } else {

                    low += 1;
                }
            }
        }
    );


    const count =
        sensors.length;

    const avgSpeed =
        speedTotal / count;

    const avgFreeFlow =
        freeFlowTotal / count;

    const avgCongestion =
        congestionTotal / count;


    document.getElementById(
        "stats"
    ).innerHTML =

        "<span class='stat'>" +
        "<b>Avg speed:</b> " +
        fmt(avgSpeed, 1) +
        " mph</span>" +

        "<span class='stat'>" +
        "<b>Avg free-flow:</b> " +
        fmt(avgFreeFlow, 1) +
        " mph</span>" +

        "<span class='stat'>" +
        "<b>Avg congestion:</b> " +
        fmt(
            avgCongestion * 100,
            1
        ) +
        "%</span>" +

        "<span class='stat'>" +
        "<b>Minimum speed:</b> " +
        fmt(minSpeed, 1) +
        " mph</span>" +

        "<span class='stat'>" +
        "<b>Maximum congestion:</b> " +
        fmt(
            maxCongestion * 100,
            1
        ) +
        "%</span>" +

        "<span class='stat'>" +
        "<b>Low:</b> " +
        low +
        "</span>" +

        "<span class='stat'>" +
        "<b>Moderate:</b> " +
        moderate +
        "</span>" +

        "<span class='stat'>" +
        "<b>High:</b> " +
        high +
        "</span>" +

        "<span class='stat'>" +
        "<b>Severe:</b> " +
        severe +
        "</span>";
}


/* ============================================================
   LEGEND
   ============================================================ */

const legend =
    L.control({
        position: "bottomright"
    });


legend.onAdd =
    function() {

        const div =
            L.DomUtil.create(
                "div",
                "legend"
            );

        div.innerHTML =
            "<b>Congestion</b><br>" +

            '<i style="background:#00cc00"></i>' +
            "Low<br>" +

            '<i style="background:#ffff00"></i>' +
            "Moderate<br>" +

            '<i style="background:#ff9900"></i>' +
            "High<br>" +

            '<i style="background:#ff0000"></i>' +
            "Severe";

        return div;
    };


legend.addTo(map);


/* ============================================================
   MAP
   ============================================================ */

function updateMap() {

    const frames =
        heatmapData.frames || [];

    if (
        frames.length === 0
    ) {

        document.getElementById(
            "frameLabel"
        ).innerText =
            " | No heatmap frames available";

        return;
    }


    const frame =
        frames[
            representativeFrameIndex
        ];


    if (!frame) {
        return;
    }


    if (
        heatLayer !== null
    ) {

        map.removeLayer(
            heatLayer
        );

        heatLayer = null;
    }


    if (
        sensorLayer !== null
    ) {

        map.removeLayer(
            sensorLayer
        );

        sensorLayer = null;
    }


    const heatPoints =
        getHeatPoints(frame);

    const sensors =
        getSensors(frame);


    /*
       The heat radius is deliberately smaller than the earlier
       version.  The heat field should show local influence around
       measured detectors rather than painting a whole corridor.
    */

    let heatRadius = 14;

    let heatBlur = 9;

    let minimumOpacity = 0.25;


    if (
        datasetName === "seattle"
    ) {

        heatRadius = 11;

        heatBlur = 7;

        minimumOpacity = 0.22;

    } else if (
        datasetName === "d7"
    ) {

        heatRadius = 14;

        heatBlur = 9;

        minimumOpacity = 0.25;
    }


    heatLayer =
        L.heatLayer(
            heatPoints,
            {
                radius:
                    heatRadius,

                blur:
                    heatBlur,

                maxZoom:
                    17,

                max:
                    1.0,

                minOpacity:
                    minimumOpacity,

                gradient: {
                    0.00: "#00cc00",
                    0.25: "#ffff00",
                    0.50: "#ff9900",
                    0.75: "#ff0000",
                    1.00: "#990000"
                }
            }
        ).addTo(map);


    sensorLayer =
        L.layerGroup();


    sensors.forEach(
        function(sensor) {

            const congestion =
                Number(
                    sensor.congestion
                ) || 0;

            const colour =
                getColor(
                    congestion
                );


            const marker =
                L.circleMarker(
                    [
                        Number(
                            sensor.latitude
                        ),
                        Number(
                            sensor.longitude
                        )
                    ],
                    {
                        radius: 7,

                        fillColor:
                            colour,

                        color:
                            "#222",

                        weight: 1,

                        opacity: 1,

                        fillOpacity: 0.90
                    }
                );


            const speed =
                Number(
                    sensor.speed
                );

            const freeFlow =
                Number(
                    sensor.free_flow
                );

            const speedReduction =
                Number.isFinite(speed) &&
                Number.isFinite(freeFlow) &&
                freeFlow > 0

                    ? (
                        1 -
                        speed /
                        freeFlow
                    ) * 100

                    : congestion * 100;


            let popup =
                "<div class='sensor-popup'>";


            if (
                datasetName === "d7"
            ) {

                popup +=
                    "<b>Sensor:</b> " +
                    (
                        sensor.sensor ??
                        "-"
                    ) +
                    "<br>" +

                    "<b>Station ID:</b> " +
                    (
                        sensor.station_id ??
                        "-"
                    ) +
                    "<br>" +

                    "<b>Freeway:</b> " +
                    (
                        sensor.freeway ??
                        "-"
                    ) +
                    "<br>" +

                    "<b>Direction:</b> " +
                    (
                        sensor.direction ??
                        "-"
                    ) +
                    "<br><br>";

            } else if (
                datasetName === "seattle"
            ) {

                popup +=
                    "<b>Sensor:</b> " +
                    (
                        sensor.sensor_name ??
                        "-"
                    ) +
                    "<br>" +

                    "<b>Roadway:</b> " +
                    (
                        sensor.roadway ??
                        "-"
                    ) +
                    "<br>" +

                    "<b>Location:</b> " +
                    (
                        sensor.location ??
                        "-"
                    ) +
                    "<br>" +

                    "<b>Milepost:</b> " +
                    (
                        sensor.milepost !== undefined
                            ? fmt(
                                sensor.milepost,
                                2
                            )
                            : "-"
                    ) +
                    "<br><br>";

            } else {

                popup +=
                    "<b>Sensor:</b> " +
                    (
                        sensor.sensor ??
                        "-"
                    ) +
                    "<br><br>";
            }


            popup +=
                "<b>Speed:</b> " +
                fmt(speed, 2) +
                " mph<br>" +

                "<b>Free-flow:</b> " +
                fmt(freeFlow, 2) +
                " mph<br>" +

                "<b>Speed reduction:</b> " +
                fmt(
                    speedReduction,
                    1
                ) +
                "%<br>" +

                "<b>Congestion:</b> " +
                fmt(
                    congestion * 100,
                    1
                ) +
                "%<br>" +

                "<b>Level:</b> " +
                getLabel(
                    congestion
                ) +
                "<br><br>" +

                "<b>Location:</b><br>" +

                fmt(
                    Number(
                        sensor.latitude
                    ),
                    5
                ) +
                ", " +

                fmt(
                    Number(
                        sensor.longitude
                    ),
                    5
                ) +

                "</div>";


            marker.bindPopup(
                popup
            );


            sensorLayer.addLayer(
                marker
            );
        }
    );


    sensorLayer.addTo(
        map
    );


    /*
       Fit the map to the real detector positions.  This is
       especially important for Seattle because the selected
       detectors now span the real network instead of a synthetic
       grid.
    */

    const bounds =
        sensors
            .map(
                function(sensor) {
                    return [
                        Number(
                            sensor.latitude
                        ),
                        Number(
                            sensor.longitude
                        )
                    ];
                }
            )
            .filter(
                function(point) {
                    return (
                        Number.isFinite(
                            point[0]
                        ) &&
                        Number.isFinite(
                            point[1]
                        )
                    );
                }
            );


    if (
        bounds.length > 0
    ) {

        map.fitBounds(
            bounds,
            {
                padding: [
                    30,
                    30
                ],

                maxZoom:
                    datasetName === "seattle"
                        ? 11
                        : 10
            }
        );

    } else if (
        heatmapData.center
    ) {

        map.setView(
            [
                heatmapData.center.lat,
                heatmapData.center.lon
            ],
            10
        );
    }


    updateHeader(
        frame
    );

    updateStats(
        sensors
    );
}


/* ============================================================
   MODE
   ============================================================ */

function setMode(mode) {

    currentMode =
        mode;


    document.getElementById(
        "predictedButton"
    ).classList.toggle(
        "active",
        mode === "predicted"
    );


    document.getElementById(
        "actualButton"
    ).classList.toggle(
        "active",
        mode === "actual"
    );


    updateMap();
}


/* ============================================================
   INITIAL DISPLAY
   ============================================================ */

updateMap();

</script>

</body>

</html>
"""

    html = html.replace(
        "__DATASET__",
        dataset.upper()
    )

    html = html.replace(
        "__DATASET_LOWER__",
        dataset
    )

    html = html.replace(
        "__DATA__",
        heatmap_data
    )

    return html


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        threaded=True,
        use_reloader=False
    )
