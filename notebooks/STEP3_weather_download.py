

import requests
import pandas as pd
from datetime import datetime

# CONFIGURATION 
OUTPUT_DIR = "data"

# City coordinates and date range
CITIES = {
    'Delhi': {
        'lat': 28.6139,
        'lon': 77.2090,
        'start': '2015-01-01',
        'end':   '2020-12-31'
    },
    'Bengaluru': {
        'lat': 12.9716,
        'lon': 77.5946,
        'start': '2015-01-01',
        'end':   '2020-12-31'
    }
}
 


def download_weather(city_name, lat, lon, start_date, end_date):
    """
    Downloads daily weather data from Open-Meteo API.

    Open-Meteo is a free weather API that gives historical data
    for any location using latitude and longitude.
    No API key or login needed!

    Parameters:
        city_name  : name of the city (for labeling)
        lat, lon   : GPS coordinates
        start_date : YYYY-MM-DD format
        end_date   : YYYY-MM-DD format

    Returns:
        pandas DataFrame with daily weather
    """
    print(f"\nDownloading weather for {city_name}...")
    print(f"  Coordinates: {lat}, {lon}")
    print(f"  Period: {start_date} to {end_date}")

    
    url = (
        "https://archive.open-meteo.com/v1/archive"
        f"?latitude={lat}"
        f"&longitude={lon}"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
        "&daily=temperature_2m_mean"
        "&daily=wind_speed_10m_max"
        "&daily=relative_humidity_2m_mean"
        "&daily=precipitation_sum"
        "&timezone=Asia%2FKolkata"
    )

    try:
        # Make the API request
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Extract the daily data from the response
        daily = data.get('daily', {})

        df = pd.DataFrame({
            'Date':          daily.get('time', []),
            'Temperature':   daily.get('temperature_2m_mean', []),
            'Wind_Speed':    daily.get('wind_speed_10m_max', []),
            'Humidity':      daily.get('relative_humidity_2m_mean', []),
            'Precipitation': daily.get('precipitation_sum', []),
        })

        df['City']  = city_name
        df['Date']  = pd.to_datetime(df['Date'])
        df['Year']  = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month

        def get_season(m):
            if m in [12, 1, 2]:   return 'Winter'
            if m in [3, 4, 5]:    return 'Spring'
            if m in [6, 7, 8, 9]: return 'Monsoon'
            return 'Post-Monsoon'

        df['Season'] = df['Month'].apply(get_season)

        print(f"   Downloaded {len(df)} days of weather data")
        return df

    except Exception as e:
        print(f"   Download failed: {e}")
        print("  Creating synthetic weather data as fallback...")
        return create_synthetic_weather(city_name, start_date, end_date)


def create_synthetic_weather(city_name, start_date, end_date):
    """
    Creates realistic synthetic weather data as a fallback.
    Used when API is unavailable or network is slow.

    The synthetic data uses realistic seasonal patterns:
    - Delhi: hot summers, cold winters, monsoon rains
    - Bengaluru: mild year-round, two monsoon seasons
    """
    import numpy as np
    np.random.seed(42)

    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n = len(dates)

    months = dates.month.to_numpy()

    if city_name == 'Delhi':
        # Delhi has extreme temperature variations
        base_temp = 25 + 15 * np.sin((months - 4) * np.pi / 6)
        wind = 8 + 4 * np.random.randn(n)
        humidity = 50 + 20 * np.sin((months - 7) * np.pi / 6)
        precip = np.where((months >= 6) & (months <= 9),
                          np.random.exponential(5, n), 0)
    else:
        # Bengaluru is milder and more uniform
        base_temp = 24 + 4 * np.sin((months - 4) * np.pi / 6)
        wind = 10 + 3 * np.random.randn(n)
        humidity = 60 + 15 * np.sin((months - 8) * np.pi / 6)
        precip = np.where(((months >= 6) & (months <= 9)) |
                          ((months >= 10) & (months <= 11)),
                          np.random.exponential(3, n), 0)

    temp = base_temp + 2 * np.random.randn(n)

    def get_season(m):
        if m in [12, 1, 2]:   return 'Winter'
        if m in [3, 4, 5]:    return 'Spring'
        if m in [6, 7, 8, 9]: return 'Monsoon'
        return 'Post-Monsoon'

    df = pd.DataFrame({
        'Date':          dates,
        'Temperature':   temp.round(1),
        'Wind_Speed':    wind.clip(0).round(1),
        'Humidity':    humidity.clip(20, 100).round(1),
        'Precipitation': precip.round(2),
        'City':          city_name,
        'Year':          dates.year,
        'Month':         months,
        'Season':        [get_season(m) for m in months],
    })

    print(f"  Created {len(df)} days of synthetic weather (fallback)")
    return df


# ── MAIN: Download weather for both cities ─
print("DOWNLOADING WEATHER DATA")


all_weather = []
for city, info in CITIES.items():
    df = download_weather(
        city, info['lat'], info['lon'],
        info['start'], info['end']
    )
    all_weather.append(df)

weather = pd.concat(all_weather, ignore_index=True)
weather.to_csv(f"{OUTPUT_DIR}/weather_data.csv", index=False)

print(f"\n Total weather rows: {len(weather)}")
print(f"Saved to: {OUTPUT_DIR}/weather_data.csv")

