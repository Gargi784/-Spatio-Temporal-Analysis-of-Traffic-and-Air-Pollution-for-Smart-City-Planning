

import pandas as pd
import os


AQ_FILE    = "data/station_day.csv"          
TRAF_FILE = "data/Banglore_traffic_Dataset.csv"
OUTPUT_DIR = "data"

print("LOADING AIR QUALITY DATA")

aq = pd.read_csv(AQ_FILE)

print("\nColumn names:")
print(list(aq.columns))

print("\nFirst 5 rows (preview):")
print(aq.head())

print("\nData types of each column:")
print(aq.dtypes)

print("\nHow many missing values per column?")
print(aq.isnull().sum())


print("\nUnique StationIds (first 30):")
print(sorted(aq['StationId'].unique())[:30])

print(f"\nTotal unique stations: {aq['StationId'].nunique()}")

print("FILTERING DELHI AND BENGALURU STATIONS")

aq['StationId'] = aq['StationId'].astype(str)

# Filter Delhi stations
delhi_mask = aq['StationId'].str.startswith('DL')
delhi_aq = aq[delhi_mask].copy()

# Filter Bengaluru stations
blr_mask = aq['StationId'].str.startswith('KA')
blr_aq = aq[blr_mask].copy()

print(f"\n Delhi stations found: {delhi_aq['StationId'].nunique()}")
print("Delhi station names:")
for s in sorted(delhi_aq['StationId'].unique()):
    print(f"   - {s}")

print(f"\n Bengaluru stations found: {blr_aq['StationId'].nunique()}")
print("Bengaluru station names:")
for s in sorted(blr_aq['StationId'].unique()):
    print(f"   - {s}")


delhi_aq['City'] = 'Delhi'
blr_aq['City'] = 'Bengaluru'


combined_aq = pd.concat([delhi_aq, blr_aq], ignore_index=True)

print(f"\n Combined dataset: {combined_aq.shape[0]} rows")
print(f"   Delhi rows: {len(delhi_aq)}")
print(f"   Bengaluru rows: {len(blr_aq)}")

print("CLEANING DATE COLUMN")

# pd.to_datetime() converts string dates to datetime objects
combined_aq['Date'] = pd.to_datetime(combined_aq['Date'])
combined_aq['Year']  = combined_aq['Date'].dt.year
combined_aq['Month'] = combined_aq['Date'].dt.month
combined_aq['Day']   = combined_aq['Date'].dt.day


def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8, 9]:
        return 'Monsoon'
    else:
        return 'Post-Monsoon'

combined_aq['Season'] = combined_aq['Month'].apply(get_season)

print(f"\nDate range in dataset:")
print(f"  From: {combined_aq['Date'].min()}")
print(f"  To:   {combined_aq['Date'].max()}")

print("\nSeason distribution:")
print(combined_aq['Season'].value_counts())

print("HANDLING MISSING VALUES")

pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'AQI']

print("Missing values BEFORE filling:")
print(combined_aq[pollutant_cols].isnull().sum())

for col in pollutant_cols:
    if col in combined_aq.columns:
        mean_val = combined_aq[col].mean()
        combined_aq[col] = combined_aq[col].fillna(mean_val)

print("\nMissing values AFTER filling:")
print(combined_aq[pollutant_cols].isnull().sum())

print("LOADING TRAFFIC DATA")

traffic = pd.read_csv(TRAF_FILE)

print("\nColumn names:")
print(list(traffic.columns))

print("\nFirst 5 rows:")
print(traffic.head())

print("\nData types:")
print(traffic.dtypes)

print("\nMissing values:")
print(traffic.isnull().sum())

print("\nUnique areas/junctions:")

area_col = 'Area Name'  
if area_col in traffic.columns:
    print(traffic[area_col].unique())

print("CLEANING TRAFFIC DATA")

traffic['Date'] = pd.to_datetime(traffic['Date'], errors='coerce')
traffic['Year']  = traffic['Date'].dt.year
traffic['Month'] = traffic['Date'].dt.month
traffic['Season'] = traffic['Month'].apply(get_season)
traffic['City'] = 'Bengaluru'


numeric_cols = traffic.select_dtypes(include='number').columns
for col in numeric_cols:
    traffic[col] = traffic[col].fillna(traffic[col].mean())

print("Traffic data after cleaning:")
print(traffic.dtypes)
print(f"\nDate range: {traffic['Date'].min()} to {traffic['Date'].max()}")

combined_aq.to_csv(f"{OUTPUT_DIR}/cleaned_air_quality.csv", index=False)
traffic.to_csv(f"{OUTPUT_DIR}/cleaned_traffic.csv", index=False)


