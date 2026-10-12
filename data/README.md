# Data

Everything here is downloaded by `python -m src.data` and cached in `data/raw/` (git-ignored).
The joined hourly table is written to `data/processed.parquet` (also git-ignored).

| Source | What | Licence / terms |
|---|---|---|
| IESO public reports, `https://reports-public.ieso.ca/public/Demand/PUB_Demand_{year}.csv` | Hourly `Ontario Demand` and `Market Demand` (MW), 2018-2025 | IESO public reports, used under the IESO website terms of use with attribution to the IESO |
| Open-Meteo Historical Weather API (`archive-api.open-meteo.com`) | Hourly `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m` for Toronto, Ottawa and Windsor | CC BY 4.0, attribution "Weather data by Open-Meteo.com" |
| `holidays` Python package | Ontario statutory holidays (`holidays.Canada(prov="ON")`) | MIT |

## Time handling

- IESO `Hour` is Hour Ending 1-24 in Eastern Standard Time all year (no daylight saving),
  so every day has exactly 24 rows, including daylight-saving change days.
- Each row is labelled by the start of its hour in EST: `timestamp = Date + (Hour - 1) hours`.
  Hour Ending 1 on 2024-01-01 becomes `2024-01-01 00:00`.
- Weather is requested in UTC and shifted by -5 hours to EST, so there is no daylight-saving
  ambiguity on either side of the join. Weather values are Open-Meteo's instantaneous value
  at the start of the hour.
- Weather is the unweighted mean of the three cities.

## Known data issues

- `PUB_Demand_2025.csv` has no row for 2025-05-01 Hour Ending 1, so `2025-05-01 00:00` is missing.
  It is the only gap in 2018-2025. There are no duplicates, zeros or negative values.
- The IESO server drops connections from the default `python-requests` User-Agent and rate-limits
  bursts of requests, so `src/data.py` sends a descriptive User-Agent and retries with backoff.
