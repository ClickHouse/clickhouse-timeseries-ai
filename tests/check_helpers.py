"""Check SQL quantile selection and rejection without making model requests."""

import json
import os
import urllib.error
import urllib.request

endpoint = os.environ.get("TIMESERIES_CLICKHOUSE_URL", "http://127.0.0.1:8125")
point = """CAST(tuple(5., 1., 9.,
    arrayMap(i -> tuple(i / 10., toFloat64(i)), range(1, 10))),
    'Tuple(expected Float64, lower_bound Float64, upper_bound Float64,
        quantiles Array(Tuple(level Float64, value Float64)))')"""


def query(expression):
    sql = f"WITH {point} AS point SELECT {expression} AS result FORMAT JSONEachRow"
    with urllib.request.urlopen(urllib.request.Request(endpoint, data=sql.encode()), timeout=10) as response:
        return json.load(response)["result"]


assert query("aiForecastQuantile(point, 0.8)") == 8
assert query("aiForecastQuantile(point, 0.5)") == 5
for coverage, lower, upper in ((0.8, 1, 9), (0.6, 2, 8), (0.4, 3, 7), (0.2, 4, 6)):
    assert query(f"aiForecastInterval(point, {coverage})") == {"lower_bound": lower, "upper_bound": upper}

unsupported = [f"aiForecastQuantile(point, {level})" for level in (0.025, 0.95, -0.1, 1)]
unsupported += [f"aiForecastInterval(point, {coverage})" for coverage in (0.95, 0.9, 0, 1, -0.5)]
unsupported += ["aiForecastQuantile(point, toFloat64('nan'))", "aiForecastInterval(point, toFloat64('inf'))"]
for expression in unsupported:
    try:
        query(expression)
    except urllib.error.HTTPError as exc:
        error = exc.read().decode()
        assert "Requested quantile is not available" in error or "Coverage must be finite" in error, error
    else:
        raise AssertionError("SQL helper accepted an unsupported request: " + expression)
print("SQL quantile helpers: native selection and unsupported-request rejection OK")
