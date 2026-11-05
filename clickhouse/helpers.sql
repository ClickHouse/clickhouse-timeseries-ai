-- Select native quantiles from a forecast step; no model request is made.
CREATE OR REPLACE FUNCTION aiForecastQuantile AS (prediction, level) ->
    arrayFirst(q -> abs(q.level - level) < 1e-12, prediction.quantiles).value
    + throwIf(NOT arrayExists(q -> abs(q.level - level) < 1e-12, prediction.quantiles),
        'Requested quantile is not available; inspect prediction.quantiles');

-- Coverage is nominal. Both equal-tail bounds must be present in the output.
CREATE OR REPLACE FUNCTION aiForecastInterval AS (prediction, coverage) ->
    CAST(tuple(
        aiForecastQuantile(prediction, (1 - coverage) / 2)
            + throwIf(NOT isFinite(coverage) OR coverage <= 0 OR coverage >= 1,
                'Coverage must be finite and between zero and one'),
        aiForecastQuantile(prediction, (1 + coverage) / 2)
    ), 'Tuple(lower_bound Float64, upper_bound Float64)');
