#!/usr/bin/env python3
"""JSONEachRow executable UDF client; one input row is one complete series."""

import json
import os
import sys
import urllib.request


def evaluate(row, operation, endpoint, key):
    if operation == "forecast":
        targets = [row["values"]]
        item = {"targets": targets}
        options = {"horizon": row["horizon"]}
    elif operation == "multivariate":
        targets = row["targets"]
        item = {
            "targets": targets,
            "past_covariates": row["past_covariates"],
            "future_covariates": row["future_covariates"],
        }
        options = {"horizon": row["horizon"]}
    else:
        targets = [row["values"]]
        item = {"targets": targets}
        options = json.loads(row["options"])
        if not isinstance(options, dict) or set(options) - {"horizon", "warmup", "only_last_n", "threshold"}:
            raise ValueError("unsupported anomaly options")
    route = "anomalies" if operation == "anomalies" else "forecast"
    body = {"model": os.environ.get("TIMESERIES_MODEL", "timesfm-3.0"), "inputs": [item], **options}
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/timeseries/" + route,
        data=json.dumps(body, allow_nan=False).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    # Never follow a redirect with a model service credential.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    with urllib.request.build_opener(NoRedirect).open(request, timeout=190) as response:
        raw = response.read((4 << 20) + 1)
        if len(raw) > 4 << 20:
            raise ValueError("model service response exceeds size limit")
        result = json.loads(raw)["predictions"][0]
    fields = ["expected", "lower_bound", "upper_bound"]
    if route == "anomalies":
        fields += ["anomaly_score", "is_anomaly"]
    rows = [
        [{name: result[name][channel][t] for name in fields}
         for t in range(len(result["expected"][channel]))]
        for channel in range(len(targets))
    ]
    if route == "forecast":
        levels = result["quantile_levels"]
        for channel, points in enumerate(rows):
            for t, point in enumerate(points):
                values = result["quantiles"][channel][t]
                if len(values) != len(levels):
                    raise ValueError("quantile levels and values do not match")
                point["quantiles"] = [{"level": level, "value": value} for level, value in zip(levels, values)]
    return rows if operation == "multivariate" else rows[0]


def main():
    endpoint = os.environ.get("TIMESERIES_URL", "http://127.0.0.1:8091")
    key = os.environ.get("TIMESERIES_API_KEY", "")
    if not key:
        raise ValueError("TIMESERIES_API_KEY is required")
    for line in sys.stdin:
        if line.strip():
            result = evaluate(json.loads(line), sys.argv[1], endpoint, key)
            print(json.dumps({"result": result}, allow_nan=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # A failed inference is a query error, never a row of normal scores.
        print("time-series UDF failed; check the model service request logs", file=sys.stderr)
        raise SystemExit(1) from None
