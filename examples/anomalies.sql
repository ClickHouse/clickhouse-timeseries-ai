-- Score only the last eight observations, including an injected spike.
WITH aiDetectAnomalies(
    arrayMap(x -> if(x = 60, 100.0, 10.0 + sin(x / 5.0)), range(64)),
    '{"only_last_n":8,"horizon":8,"threshold":3}'
) AS results
SELECT
    step - 1 AS observation,
    prediction.expected,
    prediction.lower_bound,
    prediction.upper_bound,
    prediction.anomaly_score,
    prediction.is_anomaly
FROM (SELECT results)
ARRAY JOIN results AS prediction, arrayEnumerate(results) AS step
WHERE isNotNull(prediction.expected)
FORMAT PrettyCompact;
