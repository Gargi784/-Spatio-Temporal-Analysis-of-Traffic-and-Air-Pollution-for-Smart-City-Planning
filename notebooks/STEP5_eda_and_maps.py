# ============================================================
# STEP 5: EXPLORATORY DATA ANALYSIS + SPATIAL MAPS
# ============================================================
# WHAT THIS FILE DOES:
#   - Creates temporal analysis plots (trends, patterns)
#   - Creates correlation heatmaps
#   - Creates Folium maps (interactive city maps)
#   - Saves all plots to the outputs/ folder
#
# WHAT IS EDA?
#   Exploratory Data Analysis = looking at data visually
#   before building models. You answer questions like:
#   - "Which station has the worst PM2.5?"
#   - "Is winter really more polluted?"
#   - "Does high traffic correlate with high AQI?"
#
# HOW TO RUN:
#   python notebooks/STEP5_eda_and_maps.py
#   Then open outputs/map_delhi.html in your browser!
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import folium
from folium.plugins import HeatMap
import warnings
warnings.filterwarnings('ignore')

# ── CONFIGURATION ──────────────────────────────────────────
DATA_FILE  = "data/model_ready_data.csv"
OUTPUT_DIR = "outputs"

# Map style settings
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size']      = 11
plt.style.use('seaborn-v0_8-whitegrid')
# ───────────────────────────────────────────────────────────

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS + MAPS")
print("=" * 60)

df = pd.read_csv(DATA_FILE)
df['Date'] = pd.to_datetime(df['Date'])

print(f"✅ Loaded: {df.shape[0]} rows")
delhi = df[df['City'] == 'Delhi']
blr   = df[df['City'] == 'Bengaluru']


# ============================================================
# PLOT 1: PM2.5 TREND OVER TIME (both cities)
# ============================================================
# This plot shows the TEMPORAL part of spatio-temporal analysis.
# You can see seasonal peaks (winter = high pollution in Delhi).

print("\nCreating Plot 1: PM2.5 trend over time...")

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

for ax, city_df, city_name, color in zip(
        axes,
        [delhi, blr],
        ['Delhi', 'Bengaluru'],
        ['#E74C3C', '#2980B9']):

    monthly = city_df.groupby('Date')['PM2.5'].mean().reset_index()
    ax.plot(monthly['Date'], monthly['PM2.5'],
            color=color, alpha=0.7, linewidth=1)

    # Add 30-day rolling average for trend line
    rolling = monthly['PM2.5'].rolling(30, min_periods=1).mean()
    ax.plot(monthly['Date'], rolling,
            color=color, linewidth=2.5, label='30-day average')

    # WHO guideline line
    ax.axhline(y=15, color='green', linestyle='--',
               linewidth=1.5, alpha=0.7, label='WHO guideline (15 µg/m³)')
    ax.axhline(y=60, color='orange', linestyle='--',
               linewidth=1.5, alpha=0.7, label='NAAQS standard (60 µg/m³)')

    ax.set_title(f'{city_name} — Daily PM2.5 Trend', fontsize=13, fontweight='bold')
    ax.set_ylabel('PM2.5 (µg/m³)')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(bottom=0)

axes[1].set_xlabel('Date')
plt.suptitle('Temporal Analysis: PM2.5 Over Time\n(Delhi vs Bengaluru)',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plot1_pm25_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: plot1_pm25_trend.png")


# ============================================================
# PLOT 2: SEASONAL BOX PLOTS
# ============================================================
# Box plots show the distribution of PM2.5 in each season.
# The box covers the middle 50% of values.
# The line in the middle = median.
# Dots outside = outliers (unusually high/low days).

print("\nCreating Plot 2: Seasonal box plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
season_order = ['Winter', 'Post-Monsoon', 'Spring', 'Monsoon']

for ax, city_df, city_name in zip(axes, [delhi, blr], ['Delhi', 'Bengaluru']):
    sns.boxplot(
        data=city_df, x='Season', y='PM2.5',
        order=season_order, palette='Set2', ax=ax
    )
    ax.set_title(f'{city_name}: PM2.5 by Season', fontsize=13, fontweight='bold')
    ax.set_xlabel('Season')
    ax.set_ylabel('PM2.5 (µg/m³)')
    ax.set_ylim(bottom=0)

plt.suptitle('Seasonal PM2.5 Distribution', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plot2_seasonal_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: plot2_seasonal_boxplot.png")


# ============================================================
# PLOT 3: STATION-WISE PM2.5 (SPATIAL!)
# ============================================================
# This is the SPATIAL part of spatio-temporal analysis!
# Different stations = different areas of the city.
# This shows which area (station) has worst air quality.

print("\nCreating Plot 3: Station-wise PM2.5...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, city_df, city_name in zip(axes, [delhi, blr], ['Delhi', 'Bengaluru']):
    station_avg = (city_df.groupby('StationId')['PM2.5']
                          .mean()
                          .sort_values(ascending=True)
                          .reset_index())

    # Clean station names for display
    station_avg['short_name'] = station_avg['StationId'].str.replace(
        r',.*', '', regex=True)

    colors = ['#E74C3C' if v > 60 else '#F39C12' if v > 30 else '#27AE60'
              for v in station_avg['PM2.5']]

    bars = ax.barh(station_avg['short_name'], station_avg['PM2.5'],
                   color=colors, edgecolor='white')

    ax.axvline(x=60, color='red', linestyle='--',
               linewidth=1.5, alpha=0.7, label='NAAQS (60)')
    ax.axvline(x=15, color='green', linestyle='--',
               linewidth=1.5, alpha=0.7, label='WHO (15)')

    ax.set_title(f'{city_name}: Average PM2.5 by Station/Area',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Average PM2.5 (µg/m³)')
    ax.legend(fontsize=9)

    # Add value labels
    for bar, val in zip(bars, station_avg['PM2.5']):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}', va='center', fontsize=9)

plt.suptitle('Spatial Analysis: PM2.5 by Area/Station\n(Red = Exceeds NAAQS Standard)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plot3_station_pm25.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: plot3_station_pm25.png")


# ============================================================
# PLOT 4: MONTHLY TRENDS (BOTH CITIES COMPARED)
# ============================================================
print("\nCreating Plot 4: Monthly comparison...")

monthly_city = df.groupby(['City', 'Month'])['PM2.5'].mean().reset_index()
month_names  = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']

fig, ax = plt.subplots(figsize=(12, 6))

for city, color, marker in zip(
        ['Delhi', 'Bengaluru'],
        ['#E74C3C', '#2980B9'],
        ['o', 's']):
    city_data = monthly_city[monthly_city['City'] == city]
    ax.plot(city_data['Month'], city_data['PM2.5'],
            color=color, marker=marker, linewidth=2.5,
            markersize=8, label=city)

ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_names)
ax.set_xlabel('Month')
ax.set_ylabel('Average PM2.5 (µg/m³)')
ax.set_title('Monthly PM2.5 Comparison: Delhi vs Bengaluru',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.axhline(y=60, color='red', linestyle='--',
           alpha=0.5, label='NAAQS standard')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plot4_monthly_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: plot4_monthly_comparison.png")


# ============================================================
# PLOT 5: CORRELATION HEATMAP
# ============================================================
# Correlation measures how strongly two variables move together.
# +1 = perfect positive correlation (both go up together)
# -1 = perfect negative correlation (one goes up, other goes down)
#  0 = no relationship
#
# For your project: you expect PM2.5 and AQI to be highly correlated.
# Wind speed should negatively correlate with PM2.5.

print("\nCreating Plot 5: Correlation heatmap...")

corr_cols = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'AQI',
             'Temperature', 'Wind_Speed', 'Humidity']
corr_cols = [c for c in corr_cols if c in df.columns]

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for ax, city_df, city_name in zip(axes, [delhi, blr], ['Delhi', 'Bengaluru']):
    corr_matrix = city_df[corr_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix, mask=mask, ax=ax,
        cmap='RdYlGn', center=0, vmin=-1, vmax=1,
        annot=True, fmt='.2f', annot_kws={'size': 8},
        square=True, linewidths=0.5
    )
    ax.set_title(f'{city_name}: Correlation Matrix',
                 fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('Correlation Heatmap: Pollution, Weather Variables\n'
             '(Green = positive, Red = negative correlation)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plot5_correlation_heatmap.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: plot5_correlation_heatmap.png")


# ============================================================
# FOLIUM MAP 1: DELHI STATION AQI HEATMAP
# ============================================================
# This creates an INTERACTIVE map you can open in a browser!
# Each station is shown as a colored circle.
# Color = pollution level (green = good, red = bad)
# Click any circle to see station name and average AQI.

print("\nCreating Folium Map 1: Delhi stations...")

DELHI_STATION_COORDS = {
    'Anand Vihar, Delhi - DPCC':          (28.6469, 77.3164),
    'IGI Airport, Delhi - IMD':            (28.5562, 77.0999),
    'ITO, Delhi - DPCC':                   (28.6289, 77.2398),
    'Punjabi Bagh, Delhi - DPCC':          (28.6710, 77.1313),
    'RK Puram, Delhi - DPCC':             (28.5640, 77.1855),
    'Rohini, Delhi - DPCC':               (28.7195, 77.1490),
    'Dwarka-Sector 8, Delhi - DPCC':      (28.5921, 77.0460),
    'Shadipur, Delhi - CPCB':             (28.6523, 77.1487),
    'Mandir Marg, Delhi - DPCC':          (28.6350, 77.2028),
    'Lodhi Road, Delhi - IMD':            (28.5918, 77.2273),
}

def get_color(aqi):
    if aqi <= 50:   return 'green'
    if aqi <= 100:  return 'lightgreen'
    if aqi <= 200:  return 'orange'
    if aqi <= 300:  return 'red'
    return 'darkred'

# Create Delhi map centered on Delhi
delhi_map = folium.Map(location=[28.6139, 77.2090], zoom_start=11)

station_avg_delhi = delhi.groupby('StationId').agg(
    avg_pm25=('PM2.5', 'mean'),
    avg_aqi=('AQI', 'mean')
).reset_index()

for _, row in station_avg_delhi.iterrows():
    station = row['StationId']
    coords  = None

    # Find matching coordinates
    for key, val in DELHI_STATION_COORDS.items():
        if any(part.lower() in station.lower()
               for part in key.split(',')[0].split()):
            coords = val
            break

    if coords is None:
        continue

    color   = get_color(row['avg_aqi'])
    radius  = max(8, min(20, row['avg_pm25'] / 10))

    folium.CircleMarker(
        location=coords,
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=folium.Popup(
            f"<b>{station}</b><br>"
            f"Avg PM2.5: {row['avg_pm25']:.1f} µg/m³<br>"
            f"Avg AQI: {row['avg_aqi']:.0f}",
            max_width=200
        ),
        tooltip=station.split(',')[0]
    ).add_to(delhi_map)

# Add a heatmap layer for visual effect
heatmap_data = []
for _, row in station_avg_delhi.iterrows():
    station = row['StationId']
    for key, coords in DELHI_STATION_COORDS.items():
        if any(part.lower() in station.lower()
               for part in key.split(',')[0].split()):
            heatmap_data.append([coords[0], coords[1], row['avg_pm25']])
            break

if heatmap_data:
    HeatMap(heatmap_data, radius=40, blur=25, min_opacity=0.3).add_to(delhi_map)

delhi_map.save(f'{OUTPUT_DIR}/map_delhi.html')
print("  ✅ Saved: map_delhi.html (open in browser!)")


# ============================================================
# FOLIUM MAP 2: BENGALURU STATION AQI MAP
# ============================================================
print("\nCreating Folium Map 2: Bengaluru stations...")

BLR_STATION_COORDS = {
    'BTM Layout':  (12.9141, 77.6100),
    'Hebbal':      (13.0358, 77.5970),
    'Jayanagar':   (12.9250, 77.5938),
    'Peenya':      (13.0285, 77.5194),
    'Silk Board':  (12.9177, 77.6228),
    'BWSSB':       (12.9784, 77.5908),
}

blr_map = folium.Map(location=[12.9716, 77.5946], zoom_start=11)

station_avg_blr = blr.groupby('StationId').agg(
    avg_pm25=('PM2.5', 'mean'),
    avg_aqi=('AQI', 'mean')
).reset_index()

for _, row in station_avg_blr.iterrows():
    station = row['StationId']
    coords  = None

    for key, val in BLR_STATION_COORDS.items():
        if key.lower() in station.lower():
            coords = val
            break

    if coords is None:
        coords = (12.9716 + np.random.uniform(-0.05, 0.05),
                  77.5946 + np.random.uniform(-0.05, 0.05))

    color  = get_color(row['avg_aqi'])
    radius = max(8, min(20, row['avg_pm25'] / 5))

    folium.CircleMarker(
        location=coords,
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=folium.Popup(
            f"<b>{station}</b><br>"
            f"Avg PM2.5: {row['avg_pm25']:.1f} µg/m³<br>"
            f"Avg AQI: {row['avg_aqi']:.0f}",
            max_width=200
        ),
        tooltip=station.split(',')[0]
    ).add_to(blr_map)

blr_map.save(f'{OUTPUT_DIR}/map_bengaluru.html')
print("  ✅ Saved: map_bengaluru.html (open in browser!)")

print("\n" + "=" * 60)
print("STEP 5 COMPLETE! ✅")
print(f"All plots saved in: {OUTPUT_DIR}/")
print("Open map_delhi.html and map_bengaluru.html in your browser!")
print("Next: Run STEP6_ml_models.py")
print("=" * 60)
