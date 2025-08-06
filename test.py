
import requests
import pandas as pd
def get_monthly_weather(latitude , longitude, year, month):
    """
    Fetch hourly TS (skin temp) and PRECTOTCORR (precipitation) in one API call.
    Returns:
        - Monthly average TS (°C)
        - Monthly total precipitation (mm)
    """
    # Set date range for the month
    start_date = f"{year}{month:02d}01"
    end_date = (pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(1)).strftime("%Y%m%d")
    
    # NASA POWER API request
    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    params = {
        "parameters": "TS,PRECTOTCORR",  # Request both variables
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "start": start_date,
        "end": end_date,
        "format": "JSON",
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if response.status_code != 200:
        raise ValueError(f"API Error: {data.get('message', 'Unknown error')}")
    
    # Extract hourly data
    ts_values = [
        float(val) 
        for val in data["properties"]["parameter"]["TS"].values() 
        if val != -999  # Skip missing TS
    ]
    precip_values = [
        float(val) 
        for val in data["properties"]["parameter"]["PRECTOTCORR"].values() 
        if val != -999  # Skip missing precipitation
    ]
    
    # Compute monthly stats
    avg_ts = sum(ts_values) / len(ts_values)  # TS in °C (already in °C)
    total_precip = sum(precip_values)  # PRECTOTCORR in mm
    
    return total_precip, avg_ts

print(get_monthly_weather(6.866667,10.116667, 2025, 3))
