"""Download, cache and load IESO demand, Open-Meteo weather and Ontario holidays.

All timestamps in this project are naive datetimes in Eastern Standard Time
(UTC-5, no daylight saving), labelled by the *start* of the hour. IESO
"Hour Ending 1" on 2024-01-01 covers 00:00-01:00 EST and becomes 2024-01-01 00:00.
"""

from pathlib import Path

import holidays
import time

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_PATH = RAW_DIR.parent / "processed.parquet"

IESO_URL = "https://reports-public.ieso.ca/public/Demand/PUB_Demand_{year}.csv"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
CITIES = {
    "toronto": (43.65, -79.38),
    "ottawa": (45.42, -75.70),
    "windsor": (42.32, -83.04),
}

# The IESO server drops connections from the default python-requests User-Agent.
HEADERS = {"User-Agent": "ontario-demand-forecast/0.1 (student research project)"}

FIRST_YEAR = 2018
LAST_YEAR = 2025


def _download(url: str, path: Path, params: dict | None = None, attempts: int = 4) -> Path:
    """Download url to path unless it is already cached, retrying with backoff."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(attempts):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=(15, 120))
            resp.raise_for_status()
            break
        except requests.RequestException:
            if attempt == attempts - 1:
                raise
            time.sleep(10 * 2**attempt)
    path.write_bytes(resp.content)
    time.sleep(2)
    return path


def download_ieso_year(year: int) -> Path:
    return _download(IESO_URL.format(year=year), RAW_DIR / f"PUB_Demand_{year}.csv")


def load_ieso_year(year: int) -> pd.DataFrame:
    """Parse one IESO yearly file into columns timestamp, ontario_demand, market_demand."""
    path = download_ieso_year(year)
    df = pd.read_csv(path, comment="\\")
    df = df.rename(columns={
        "Date": "date",
        "Hour": "hour_ending",
        "Ontario Demand": "ontario_demand",
        "Market Demand": "market_demand",
    })
    df["timestamp"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour_ending"] - 1, unit="h")
    return df[["timestamp", "ontario_demand", "market_demand"]]


def load_demand(first_year: int = FIRST_YEAR, last_year: int = LAST_YEAR) -> pd.DataFrame:
    return pd.concat(
        [load_ieso_year(y) for y in range(first_year, last_year + 1)], ignore_index=True
    )


def download_weather(city: str, start: str, end: str) -> Path:
    """Fetch hourly weather for one city in UTC and cache the raw JSON."""
    lat, lon = CITIES[city]
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(WEATHER_VARS),
        "timezone": "GMT",
    }
    return _download(WEATHER_URL, RAW_DIR / f"weather_{city}_{start}_{end}.json", params)


def load_weather(first_year: int = FIRST_YEAR, last_year: int = LAST_YEAR) -> pd.DataFrame:
    """Hourly weather averaged over CITIES, indexed in EST (UTC-5).

    The UTC request window is padded by a day on each side so that the EST
    conversion still covers every hour of the requested years.
    """
    start, end = f"{first_year - 1}-12-31", f"{last_year + 1}-01-01"
    frames = []
    for city in CITIES:
        raw = pd.read_json(download_weather(city, start, end), typ="series")
        hourly = pd.DataFrame(raw["hourly"])
        hourly["timestamp"] = pd.to_datetime(hourly["time"]) - pd.Timedelta(hours=5)
        frames.append(hourly.drop(columns="time").assign(city=city))
    weather = pd.concat(frames).groupby("timestamp")[WEATHER_VARS].mean().reset_index()
    weather = weather.rename(columns={
        "temperature_2m": "temp",
        "relative_humidity_2m": "humidity",
        "wind_speed_10m": "wind",
    })
    in_range = weather["timestamp"].dt.year.between(first_year, last_year)
    return weather[in_range].reset_index(drop=True)


def ontario_holidays(first_year: int = FIRST_YEAR, last_year: int = LAST_YEAR) -> set:
    """Set of datetime.date objects that are Ontario statutory holidays."""
    return set(holidays.Canada(subdiv="ON", years=range(first_year - 1, last_year + 2)).keys())


def build_dataset() -> pd.DataFrame:
    """Join demand and weather onto a complete hourly EST grid and cache the result.

    Hours missing from the IESO files keep their weather and get NaN demand.
    """
    demand = load_demand()
    weather = load_weather()
    grid = pd.DataFrame({"timestamp": pd.date_range(
        f"{FIRST_YEAR}-01-01", f"{LAST_YEAR}-12-31 23:00", freq="h")})
    df = grid.merge(demand, on="timestamp", how="left").merge(weather, on="timestamp", how="left")
    df.to_parquet(PROCESSED_PATH, index=False)
    return df


if __name__ == "__main__":
    df = build_dataset()
    print(df.head())
    print(f"{len(df)} rows, {df['timestamp'].min()} to {df['timestamp'].max()}")
