-- Run with clickhouse-client --multiquery. A failed assertion fails the query.
WITH aiForecast(arrayMap(x -> 10.0 + sin(x / 5.0), range(64)), toUInt32(4)) AS prediction
SELECT throwIf(length(prediction) != 4 OR arrayExists(p -> NOT isFinite(p.expected) OR p.lower_bound > p.expected OR p.expected > p.upper_bound, prediction), 'invalid forecast');

WITH aiForecast(arrayMap(x -> 10.0 + sin(x / 5.0), range(64)), toUInt32(4)) AS prediction
SELECT throwIf(arrayExists(p -> length(p.quantiles) != 9
    OR aiForecastQuantile(p, 0.5) != p.expected
    OR aiForecastQuantile(p, 0.8) < p.expected
    OR aiForecastInterval(p, 0.8).lower_bound != p.lower_bound
    OR aiForecastInterval(p, 0.8).upper_bound != p.upper_bound
    OR aiForecastInterval(p, 0.6).lower_bound != aiForecastQuantile(p, 0.2), prediction), 'invalid quantile selection');

WITH aiForecastMultivariate(
    [arrayMap(x -> 10.0 + sin(x / 5.0), range(64)), arrayMap(x -> 20.0 + 2 * sin(x / 5.0), range(64))],
    toUInt32(4),
    [arrayMap(x -> toFloat64(x % 7), range(64))],
    [arrayMap(x -> toFloat64(x % 7 < 2), range(68))]
) AS prediction
SELECT throwIf(length(prediction) != 2 OR arrayExists(c -> length(c) != 4, prediction), 'invalid multivariate forecast');

WITH aiDetectAnomalies(
    arrayMap(x -> if(x = 60, 100.0, 10.0 + sin(x / 5.0)), range(64)),
    '{"only_last_n":4,"horizon":4}'
) AS prediction
SELECT throwIf(length(prediction) != 64 OR arrayCount(p -> isNotNull(p.expected), prediction) != 4 OR prediction[61].is_anomaly != 1, 'invalid trailing anomaly result');

WITH aiDetectAnomalies(arrayMap(x -> 10.0 + sin(x / 5.0), range(64)), '{"horizon":16}') AS prediction
SELECT throwIf(length(prediction) != 64 OR arrayCount(p -> isNotNull(p.expected), prediction) != 32, 'invalid historical anomaly result');
