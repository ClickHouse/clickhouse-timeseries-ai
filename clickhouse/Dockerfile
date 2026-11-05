FROM clickhouse/clickhouse-server:26.8
USER root
RUN apt-get update && apt-get install -y --no-install-recommends python3 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY clickhouse/udf.py /opt/timeseries/timeseries_udf
RUN chmod 755 /opt/timeseries/timeseries_udf
COPY clickhouse/functions.xml /etc/clickhouse-server/timeseries-functions.xml
COPY clickhouse/config.xml /etc/clickhouse-server/config.d/timeseries.xml
COPY clickhouse/helpers.sql /docker-entrypoint-initdb.d/01-timeseries-helpers.sql
