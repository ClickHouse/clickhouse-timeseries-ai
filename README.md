# Time-series foundation models in ClickHouse

Run TimesFM 3.0 locally and call it from ClickHouse SQL for zero-shot forecasting,
multivariate forecasting, and historical anomaly detection. No per-series training,
cloud account, internal gateway, or hosted inference service is required.

This is a local evaluation demo, not a production service. Google's downloaded
TimesFM 3.0 weights have **non-commercial, non-production restrictions**. Read the
[checkpoint license](https://huggingface.co/google/timesfm-3.0-pytorch/blob/43046b85ec22d584a13f8098c2ed39c889e129c2/LICENSE)
before downloading or using them. The code license in this repository does not
change the model-weight license.

## Quick start

Requirements: Docker with Compose v2, internet access for the first build and
checkpoint download, and enough memory for PyTorch, the model, and ClickHouse.
Allow roughly 8 GB of available memory and several GB of disk space as a starting
point; this is not a measured minimum. The default is CPU-only on Linux containers,
including Docker Desktop and OrbStack on macOS. No GPU is required.

After cloning this repository, run from its root:

```sh
docker compose up --build -d --wait --wait-timeout 1200
```

The first start downloads the pinned checkpoint. Later starts reuse the model-cache
volume. ClickHouse starts after the model is ready. If startup takes longer than
20 minutes, inspect `docker compose logs model`; the wait command timing out does
not remove the containers or cache.

Run the blog-style forecast and anomaly examples:

```sh
docker compose exec -T clickhouse clickhouse-client --multiquery < examples/forecast.sql
docker compose exec -T clickhouse clickhouse-client --multiquery < examples/anomalies.sql
docker compose exec -T clickhouse clickhouse-client --multiquery < examples/multivariate.sql
```

Or send SQL over HTTP:

```sh
curl --fail-with-body -sS http://127.0.0.1:8125/ --data-binary @examples/forecast.sql
```

All three executable UDFs and both SQL selection helpers are registered on first
startup. You do not need to create them manually. Inputs in these examples are synthetic, and inference
results come from the real pretrained model, not stored outputs.

## How it works

```text
ClickHouse SQL → Python executable UDF → local HTTP service → TimesFM 3.0
              ← typed arrays/tuples   ← JSON predictions   ← quantiles
```

SQL constructs ordered numerical arrays. The UDF sends one HTTP request per row.
The HTTP service validates inputs, runs the model, and returns all nine native
quantiles, along with median and q10/q90 convenience fields. Historical scoring
forecasts held-out observations from earlier
history, then compares the forecasts with actual values. It does not train a model.

| File | Purpose |
|---|---|
| `compose.yaml` | CPU model service, ClickHouse, health checks, and persistent volumes |
| `server/model.py` | Pinned TimesFM adapter and quantile mapping |
| `server/engine.py` | Input bounds, forecasts, causal scoring, and usage counts |
| `server/serve.py` | Bearer authentication, body limit, and one active request |
| `clickhouse/udf.py` | JSONEachRow HTTP client and named tuple output |
| `clickhouse/functions.xml` | Three executable UDF definitions |
| `clickhouse/helpers.sql` | SQL-only native quantile and interval selection |
| `examples/` | SQL and HTTP requests using synthetic data |
| `tests/` | Unit tests, HTTP smoke, and SQL assertions |

## SQL interface

```sql
aiForecast(values, horizon)
aiForecastMultivariate(targets, horizon, past_covariates, future_covariates)
aiDetectAnomalies(values, options_json)
```

`values` is `Array(Float64)`; `horizon` is `UInt32`. Forecast output is an array
of named tuples `(expected, lower_bound, upper_bound, quantiles)`, one tuple per
future step. `quantiles` is `Array(Tuple(level Float64, value Float64))` and contains
all nine native levels, q10 through q90, returned by the same model call.

Given one forecast step `point`, `aiForecastQuantile(point, 0.8)` selects q80.
`aiForecastInterval(point, 0.6)` returns named q20/q80 bounds for a nominal 60%
interval. Use `arrayMap(p -> aiForecastQuantile(p, 0.8), forecast)` to select a
trajectory from a stored or returned forecast array. Neither helper makes a model
request. They require native levels and reject unsupported requests, including a
95% interval; there is no interpolation, extrapolation, or implicit calibration.
The helpers use a small numerical tolerance when matching levels so arithmetic
such as `(1 - 0.8) / 2` can select q10.

Multivariate inputs are `Array(Array(Float64))`: one inner array per target or
covariate channel. All history channels have the same length. Known-future
covariates must cover history **plus** the forecast horizon. The result contains
one forecast array per target. For no covariates, use
`CAST([], 'Array(Array(Float64))')` for each absent covariate argument.

Anomaly output is aligned to the input history and contains
`(expected, lower_bound, upper_bound, anomaly_score, is_anomaly)`. Options are a JSON
string; use `'{}'` for defaults:

| Option | Default | Meaning |
|---|---|---|
| `warmup` | 32 | Initial context-only observations, minimum 8 |
| `horizon` | 16 | Number of observations forecast in each held-out window |
| `only_last_n` | 0 | Score only the last N observations; 0 scores all after warmup |
| `threshold` | 1.0 | Flag scores strictly greater than this positive value |

The bounds are q10 and q90: a **nominal 80% prediction interval**, not a calibrated
confidence guarantee. The anomaly score divides the absolute forecast error by
the distance from the median to the corresponding interval bound, with a numerical
floor. It is **not a probability**. Threshold 1 tests the nominal interval and is
not a calibrated alert threshold. Unscored positions have null predictions and
scores, with false flags; they were not evaluated as normal.
Anomaly output does not include the forecast quantile array. Its q10/q90 reference
and score threshold remain fixed independently of forecast interval selection.

SQL owns ordering, aggregation, timezones, regular sampling, and gap filling. The
model receives numbers, not timestamps. Independent series belong in separate UDF
rows; related targets can share a multivariate input. A failed inference fails the
query instead of returning apparently normal scores.

## HTTP API

The model service listens on `http://127.0.0.1:8091`. Both routes require a bearer key:

```text
POST /v1/timeseries/forecast
POST /v1/timeseries/anomalies
```

```sh
curl --fail-with-body -sS http://127.0.0.1:8091/v1/timeseries/forecast \
  -H 'Authorization: Bearer local-evaluation-only' \
  -H 'Content-Type: application/json' \
  --data-binary @examples/forecast.json
```

Each `inputs` entry has `targets` shaped `[target][history_step]`, plus optional
`past_covariates` and `future_covariates`. Use `"model": "timesfm-3.0"`.
For anomaly requests, known-future covariates cover the submitted history only.
Anomaly options use the same names as the SQL JSON options and go at the request
top level. The median and convenience bounds are `[target][step]` matrices.
Forecast predictions also include `quantile_levels` and a numeric `quantiles`
array shaped `[target][step][quantile]`. Anomaly predictions retain their original
shape and do not expose the quantile cube. Responses identify the exact
checkpoint revision and count model calls and numerical input/output values.

`GET /healthz` requires no key and becomes available after model loading. Invalid
requests return 400, bad credentials 401, oversized bodies 413, and concurrent
model calls 429. Responses exceeding 4 MiB also return 413; reduce the number of
inputs, targets, or horizon if this occurs. There are no automatic retries. The UDF waits up to 190 seconds
per HTTP request. Large historical-scoring jobs can exceed that timeout on CPU.

## Configuration and security

Defaults are enough for the quick start. To change keys or host ports:

```sh
cp .env.example .env
chmod 600 .env
# Edit .env, then recreate the services:
docker compose up -d --wait --wait-timeout 1200
```

`.env` is ignored by Git and excluded from Docker builds. Compose gives the same
service key to the UDF and model server. Credentials do not go in SQL. If you change
the key, update the HTTP examples accordingly. `MODEL_PORT` and `CLICKHOUSE_PORT`
change host ports without changing communication inside the Compose network.

Only loopback ports are published. ClickHouse uses its default user without a
password for this isolated local demo; the model uses a known development key.
**Do not expose these ports to a network, use real sensitive data, or treat this
as a production security boundary.** Other local users/processes can reach them.
There is no TLS, rate limiter, monetary budget enforcement, or production auth.

The service limits requests to 16 inputs, 8 total channels per input, 8–8,192
historical steps, a horizon of 1–256, 131,072 submitted values, 128 historical
windows, and 2,097,152 context-value work units. Request bodies are bounded to
4 MiB; the UDF also bounds response reads to 4 MiB. Only one inference request
runs at a time. These are demo limits, not the model's advertised capacity.

The checkpoint pin defaults to `43046b85ec22d584a13f8098c2ed39c889e129c2`.
Changing `TIMESFM_REVISION` requires another full 40-character commit SHA, a
compatible checkpoint, and a review of its license. The checkpoint is downloaded
from Hugging Face; weights are not included in this repository or its images.

## Run the model on the host instead

For development, install [uv](https://docs.astral.sh/uv/) and Python 3.11–3.13:

```sh
uv sync --frozen
export TIMESFM_REVISION=43046b85ec22d584a13f8098c2ed39c889e129c2
export TIMESFM_DEVICE=cpu
export TIMESERIES_API_KEY=local-evaluation-only
uv run --frozen python server/serve.py
```

This starts the model server only, on loopback port 8091. It does not start or
reconfigure ClickHouse. The default Compose path runs both components together.
Linux installs use CPU-only PyTorch wheels; this lockfile is for CPU evaluation,
not a CUDA deployment. On macOS, the native PyTorch package also runs on CPU here.

## Tests

Unit tests do not download weights:

```sh
uv sync --frozen
uv run --frozen python tests/run.py
```

Against the running Compose stack, verify the actual model and UDF HTTP client:

```sh
docker compose exec -T model python smoke.py
```

Or run the same checks from the host:

```sh
PYTHONPATH=clickhouse uv run --frozen python tests/smoke.py
```

The host command uses port 8091 and key `local-evaluation-only` by default. Override
`TIMESERIES_URL` and `TIMESERIES_API_KEY` for custom settings; it does not load `.env`.
Then run the end-to-end SQL assertions:

```sh
docker compose exec -T clickhouse clickhouse-client --multiquery < tests/smoke.sql
docker compose exec -T -e TIMESERIES_CLICKHOUSE_URL=http://127.0.0.1:8123 clickhouse python3 - < tests/check_helpers.py
```

The SQL smoke prints five zeroes on success. It checks forecasts, native quantile
and interval selection, multivariate
forecasts with covariates, trailing anomaly scoring, and historical scoring.
These are functional checks, not accuracy or performance benchmarks. GitHub
Actions runs unit tests and validates Compose configuration without downloading
model weights.

If you update an existing demo, rebuild the images and register the SQL helpers
in its existing database volume:

```sh
docker compose up --build -d --wait --wait-timeout 1200
docker compose exec -T clickhouse clickhouse-client --multiquery < clickhouse/helpers.sql
```

Do not delete the volume to update the executable UDF: its script is installed
outside the data directory, so rebuilding the image updates the code. Helper SQL
runs automatically only on first database initialization. Forecast tuples have
four fields; named median and q10/q90 fields retain their meanings, but positional
consumers must account for the quantile array.

## Stop and troubleshoot

```sh
docker compose logs -f model
docker compose ps
docker compose down
```

`down` keeps ClickHouse data and cached weights. `docker compose down --volumes`
deletes both volumes permanently; use it only when you want to discard that data
and download the checkpoint again next time.

If a port is in use, choose other host ports in `.env`. For out-of-memory errors,
increase Docker's memory allocation and reduce history length or target count.
For slow first starts, check Docker's access to PyPI, the PyTorch CPU wheel index,
and Hugging Face. For UDF failures, inspect `docker compose logs model clickhouse`.

## License

The demo code is MIT-licensed; see [LICENSE](LICENSE). TimesFM, PyTorch, ClickHouse,
and other dependencies retain their own licenses. In particular, the TimesFM 3.0
checkpoint's non-commercial, non-production terms apply independently. This demo
does not include Timestral or register functions in ClickHouse Cloud.
