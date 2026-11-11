"""Load only the selected, pinned checkpoint when the server starts."""

import os
import re

from engine import Engine

MODEL_NAMES = ("timesfm-3.0", "timestral-209m")


def load_engine(model_name="timesfm-3.0"):
    if model_name not in MODEL_NAMES:
        raise ValueError("unsupported model")
    prefix = "TIMESFM" if model_name == "timesfm-3.0" else "TIMESTRAL"
    revision = os.environ.get(prefix + "_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"{prefix}_REVISION must pin a 40-character Hugging Face commit SHA")

    device = os.environ.get(prefix + "_DEVICE", "cpu")
    if model_name == "timestral-209m":
        from timestral import TimestralForecaster

        forecaster = TimestralForecaster.from_pretrained(
            "mistralai/Timestral-209M", revision=revision, device=device,
        )
        return Engine(TimestralModel(forecaster), revision, model_name)

    from timesfm3 import ModelConfig, TimesFM3Evaluator

    config = ModelConfig(
        checkpoint_path="google/timesfm-3.0-pytorch",
        revision=revision,
        per_core_batch_size=4,
        device=device,
    )
    return Engine(TimesFMModel(TimesFM3Evaluator(config)), revision)


class TimesFMModel:
    def __init__(self, forecaster):
        self.forecaster = forecaster

    def predict(self, item, horizon):
        import numpy as np

        def optional(name):
            rows = item.get(name)
            return [np.asarray(rows, dtype=np.float32)] if rows else None

        outputs = list(self.forecaster.predict_batch(
            contexts=[np.asarray(item["targets"], dtype=np.float32)],
            horizon=horizon,
            past_only_covariates=optional("past_covariates"),
            past_future_covariates=optional("future_covariates"),
            return_quantiles=True,
            make_positive=False,
            sort_quantiles=True,
            use_symmetric_averaging=False,
        ))
        quantiles = np.asarray(outputs[0].quantiles)
        if quantiles.ndim == 2:
            quantiles = quantiles[None, :, :]
        return {
            "expected": quantiles[:, :, 4].tolist(),
            "lower_bound": quantiles[:, :, 0].tolist(),
            "upper_bound": quantiles[:, :, 8].tolist(),
            "quantile_levels": [i / 10 for i in range(1, 10)],
            "quantiles": quantiles.tolist(),
        }


class TimestralModel:
    def __init__(self, forecaster):
        self.forecaster = forecaster

    def predict(self, item, horizon):
        import numpy as np

        def optional(name):
            rows = item.get(name)
            return np.asarray([rows], dtype=np.float32) if rows else None

        forecast = self.forecaster.forecast(
            context=np.asarray([item["targets"]], dtype=np.float32),
            prediction_length=horizon,
            past_covariates=optional("past_covariates"),
            known_covariates=optional("future_covariates"),
            symmetric_averaging=False,
            clip_nonnegative=False,
        )
        # Timestral returns a torch tensor, possibly on a GPU, with a batch axis.
        quantiles = forecast.quantiles.detach().cpu().numpy()[0]
        levels = list(forecast.quantile_levels)
        return {
            "expected": quantiles[:, :, levels.index(0.5)].tolist(),
            "lower_bound": quantiles[:, :, levels.index(0.1)].tolist(),
            "upper_bound": quantiles[:, :, levels.index(0.9)].tolist(),
            "quantile_levels": levels,
            "quantiles": quantiles.tolist(),
        }
