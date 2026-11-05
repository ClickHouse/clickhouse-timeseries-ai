-- Two related targets, one historical covariate and one known-future covariate.
SELECT aiForecastMultivariate(
    [arrayMap(x -> 10.0 + sin(x / 5.0), range(64)),
     arrayMap(x -> 20.0 + 2 * sin(x / 5.0), range(64))],
    toUInt32(4),
    [arrayMap(x -> toFloat64(x % 7), range(64))],
    [arrayMap(x -> toFloat64(x % 7 < 2), range(68))]
) AS forecast
FORMAT JSONEachRow;
