-- Forecast four future values from 64 regularly spaced observations.
SELECT aiForecast(
    arrayMap(x -> 10.0 + sin(x / 5.0), range(64)),
    toUInt32(4)
) AS forecast
FORMAT JSONEachRow;
