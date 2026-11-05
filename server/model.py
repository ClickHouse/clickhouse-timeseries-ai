"""Load the pinned TimesFM checkpoint only when the server starts."""

import os
import re

from engine import Engine


def load_engine():
    revision = os.environ.get("TIMESFM_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("TIMESFM_REVISION must pin a 40-character Hugging Face commit SHA")

    from timesfm3 import ModelConfig, TimesFM3Evaluator

    config = ModelConfig(
        checkpoint_path="google/timesfm-3.0-pytorch",
        revision=revision,
        per_core_batch_size=4,
        device=os.environ.get("TIMESFM_DEVICE", "cpu"),
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
