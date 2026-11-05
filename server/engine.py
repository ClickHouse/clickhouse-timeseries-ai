"""Bounded forecasting and causal historical scoring for the evaluation API."""

import math

MODEL = "timesfm-3.0"
MAX_POINTS = 8192
MAX_HORIZON = 256
MAX_INPUTS = 16
MAX_CHANNELS = 8
MAX_VALUES = 131072
MAX_WINDOWS = 128
MAX_WORK = 2097152


def integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer between {low} and {high}")
    return value


def matrix(value, name, length=None, required=False):
    if not isinstance(value, list) or len(value) > MAX_CHANNELS:
        raise ValueError(f"{name} must be an array of at most {MAX_CHANNELS} channels")
    if required and not value:
        raise ValueError(f"{name} must contain at least one channel")
    for channel in value:
        if not isinstance(channel, list) or not channel or len(channel) > MAX_POINTS + MAX_HORIZON:
            raise ValueError(f"{name} channels must be nonempty bounded arrays")
        if length is None:
            length = len(channel)
        if len(channel) != length:
            raise ValueError(f"{name} channels have an incorrect length")
        if any(type(x) not in (int, float) or abs(x) > 1e20 or not math.isfinite(x) for x in channel):
            raise ValueError(f"{name} values must be finite numbers with magnitude at most 1e20")
    return value


def validate(body, operation, model_name=MODEL):
    if not isinstance(body, dict) or body.get("model") != model_name:
        raise ValueError(f"model must be {model_name}")
    allowed = {"model", "inputs", "horizon"}
    if operation == "anomalies":
        allowed |= {"only_last_n", "warmup", "threshold"}
    if set(body) - allowed:
        raise ValueError("unknown request field")
    horizon = integer(body.get("horizon", 16), "horizon", 1, MAX_HORIZON)
    inputs = body.get("inputs")
    if not isinstance(inputs, list) or not 1 <= len(inputs) <= MAX_INPUTS:
        raise ValueError(f"inputs must contain 1 to {MAX_INPUTS} series")
    count = work = windows = 0
    warmup = integer(body.get("warmup", 32), "warmup", 8, MAX_POINTS - 1)
    last = integer(body.get("only_last_n", 0), "only_last_n", 0, MAX_POINTS)
    threshold = body.get("threshold", 1.0)
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be a finite positive number")
    for item in inputs:
        if not isinstance(item, dict) or set(item) - {"targets", "past_covariates", "future_covariates"}:
            raise ValueError("each input must contain targets and optional covariates")
        targets = matrix(item.get("targets"), "targets", required=True)
        size = len(targets[0])
        integer(size, "history length", 8, MAX_POINTS)
        past = matrix(item.get("past_covariates", []), "past_covariates", size)
        future_size = size + horizon if operation == "forecast" else size
        future = matrix(item.get("future_covariates", []), "future_covariates", future_size)
        channels = len(targets) + len(past) + len(future)
        if channels > MAX_CHANNELS:
            raise ValueError(f"at most {MAX_CHANNELS} total channels per input")
        item_values = sum(len(c) for c in targets + past + future)
        count += item_values
        if operation == "anomalies":
            if size <= warmup:
                raise ValueError("history must be longer than warmup")
            if last and last > size - warmup:
                raise ValueError("only_last_n must leave at least warmup observations")
            start = size - last if last else warmup
            for offset in range(start, size, horizon):
                work += offset * channels
                windows += 1
        else:
            work += item_values
    if count > MAX_VALUES or work > MAX_WORK or windows > MAX_WINDOWS:
        raise ValueError("request exceeds the total values or forecast-window work limit")
    return inputs, horizon, warmup, last, float(threshold)


def check_prediction(prediction, channels, horizon, require_quantiles=False):
    for name in ("expected", "lower_bound", "upper_bound"):
        values = prediction[name]
        if len(values) != channels or any(len(c) != horizon for c in values):
            raise RuntimeError("model returned an incorrect prediction shape")
        if any(not math.isfinite(x) for c in values for x in c):
            raise RuntimeError("model returned a nonfinite prediction")
    for c in range(channels):
        for t in range(horizon):
            if not prediction["lower_bound"][c][t] <= prediction["expected"][c][t] <= prediction["upper_bound"][c][t]:
                raise RuntimeError("model returned unordered quantiles")
    if require_quantiles:
        levels = prediction.get("quantile_levels")
        if (not isinstance(levels, list) or not levels
                or any(type(q) not in (int, float) or not math.isfinite(q) or not 0 < q < 1 for q in levels)
                or any(a >= b for a, b in zip(levels, levels[1:]))
                or any(q not in levels for q in (0.1, 0.5, 0.9))):
            raise RuntimeError("model returned invalid quantile levels")
        quantiles = prediction.get("quantiles")
        if (not isinstance(quantiles, list) or len(quantiles) != channels
                or any(not isinstance(channel, list) or len(channel) != horizon for channel in quantiles)):
            raise RuntimeError("model returned an incorrect quantile shape")
        for c, channel in enumerate(quantiles):
            for t, values in enumerate(channel):
                if (not isinstance(values, list) or len(values) != len(levels)
                        or any(type(x) not in (int, float) or not math.isfinite(x) for x in values)):
                    raise RuntimeError("model returned invalid quantile values")
                if any(a > b for a, b in zip(values, values[1:])):
                    raise RuntimeError("model returned unordered quantiles")
                for name, level in (("expected", 0.5), ("lower_bound", 0.1), ("upper_bound", 0.9)):
                    if prediction[name][c][t] != values[levels.index(level)]:
                        raise RuntimeError("model returned inconsistent quantile summaries")


class Engine:
    def __init__(self, model, revision, model_name=MODEL):
        self.model = model
        self.revision = revision
        self.model_name = model_name

    def run(self, body, operation):
        if operation not in ("forecast", "anomalies"):
            raise ValueError("unknown operation")
        inputs, horizon, warmup, last, threshold = validate(body, operation, self.model_name)
        predictions = []
        calls = input_values = output_values = 0
        for item in inputs:
            targets = item["targets"]
            channels, size = len(targets), len(targets[0])
            if operation == "forecast":
                result = self.model.predict(item, horizon)
                check_prediction(result, channels, horizon, require_quantiles=True)
                calls += 1
                input_values += sum(len(c) for c in targets + item.get("past_covariates", []) + item.get("future_covariates", []))
                output_values += channels * horizon
            else:
                start = size - last if last else warmup
                result = {name: [[None] * size for _ in targets] for name in ("expected", "lower_bound", "upper_bound", "anomaly_score")}
                result["is_anomaly"] = [[False] * size for _ in targets]
                for offset in range(start, size, horizon):
                    steps = min(horizon, size - offset)
                    window = {"targets": [c[:offset] for c in targets]}
                    for name in ("past_covariates", "future_covariates"):
                        end = offset if name == "past_covariates" else offset + steps
                        window[name] = [c[:end] for c in item.get(name, [])]
                    pred = self.model.predict(window, steps)
                    check_prediction(pred, channels, steps)
                    calls += 1
                    input_values += sum(len(c) for rows in window.values() for c in rows)
                    output_values += channels * steps
                    for c in range(channels):
                        for t in range(steps):
                            index = offset + t
                            for name in ("expected", "lower_bound", "upper_bound"):
                                result[name][c][index] = pred[name][c][t]
                            expected = pred["expected"][c][t]
                            observed = targets[c][index]
                            side = pred["upper_bound"][c][t] - expected if observed >= expected else expected - pred["lower_bound"][c][t]
                            scale = max(side, max(abs(expected), 1.0) * 1e-6)
                            score = abs(observed - expected) / scale
                            result["anomaly_score"][c][index] = score
                            result["is_anomaly"][c][index] = score > threshold
            predictions.append(result)
        return {
            "model": self.model_name,
            "model_revision": self.revision,
            "operation": operation,
            "interval_quantiles": [0.1, 0.9],
            "predictions": predictions,
            "usage": {"model_calls": calls, "input_values": input_values, "output_values": output_values},
        }
