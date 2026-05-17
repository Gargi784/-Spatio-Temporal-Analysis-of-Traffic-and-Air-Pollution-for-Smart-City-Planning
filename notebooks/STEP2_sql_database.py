import pandas as pd
import sqlite3
import os
import os
os.makedirs("data", exist_ok=True)     
os.makedirs("outputs", exist_ok=True)   
os.makedirs("outputs/models", exist_ok=True)


DB_PATH  = "data/project_database.db"   
AQ_FILE  = "data/cleaned_air_quality.csv"
TRAF_FILE= "data/cleaned_traffic.csv"

print("CREATING SQL DATABASE")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print(f" Database created/connected: {DB_PATH}")

print("\nCreating tables")

cursor.executescript("""
    DROP TABLE IF EXISTS location_dim;
    DROP TABLE IF EXISTS pollution_fact;
    DROP TABLE IF EXISTS traffic_fact;
    DROP TABLE IF EXISTS weather_fact;
""")

cursor.execute("""
    CREATE TABLE location_dim (
        location_id   INTEGER PRIMARY KEY AUTOINCREMENT,
        station_name  TEXT NOT NULL,
        city          TEXT NOT NULL,
        latitude      REAL,
        longitude     REAL
    )
""")
print(" Created table: location_dim")

# Create pollution_fact table
cursor.execute("""
    CREATE TABLE pollution_fact (
        pollution_id  INTEGER PRIMARY KEY AUTOINCREMENT,
        location_id   INTEGER NOT NULL,
        date          TEXT NOT NULL,
        year          INTEGER,
        month         INTEGER,
        season        TEXT,
        pm25          REAL,
        pm10          REAL,
        no2           REAL,
        co            REAL,
        so2           REAL,
        o3            REAL,
        aqi           REAL,
        aqi_bucket    TEXT,
        FOREIGN KEY (location_id) REFERENCES location_dim(location_id)
    )
""")
print("Created table: pollution_fact")

# Create traffic_fact table
cursor.execute("""
    CREATE TABLE traffic_fact (
        traffic_id            INTEGER PRIMARY KEY AUTOINCREMENT,
        location_id           INTEGER NOT NULL,
        date                  TEXT,
        year                  INTEGER,
        month                 INTEGER,
        season                TEXT,
        area_name             TEXT,
        road_intersection     TEXT,
        traffic_volume        REAL,
        average_speed         REAL,
        congestion_level      TEXT,
        road_capacity_util    REAL,
        environmental_impact  TEXT,
        weather_conditions    TEXT,
        FOREIGN KEY (location_id) REFERENCES location_dim(location_id)
    )
""")
print("Created table: traffic_fact")

# Create weather_fact table
cursor.execute("""
    CREATE TABLE weather_fact (
        weather_id    INTEGER PRIMARY KEY AUTOINCREMENT,
        location_id   INTEGER NOT NULL,
        date          TEXT NOT NULL,
        city          TEXT,
        temperature   REAL,
        wind_speed    REAL,
        humidity      REAL,
        precipitation REAL,
        FOREIGN KEY (location_id) REFERENCES location_dim(location_id)
    )
""")
print("Created table: weather_fact")

conn.commit()

STATION_COORDS = {
    # Delhi stations
    'Anand Vihar, Delhi':        (28.6469, 77.3164),
    'IGI Airport, Delhi':        (28.5562, 77.0999),
    'ITO, Delhi':                (28.6289, 77.2398),
    'Punjabi Bagh, Delhi':       (28.6710, 77.1313),
    'RK Puram, Delhi':           (28.5640, 77.1855),
    'Rohini, Delhi':             (28.7195, 77.1490),
    'Dwarka, Delhi':             (28.5921, 77.0460),
    'Shadipur, Delhi':           (28.6523, 77.1487),
    'Mandir Marg, Delhi':        (28.6350, 77.2028),
    'Lodhi Road, Delhi':         (28.5918, 77.2273),
    # Bengaluru stations
    'BTM Layout, Bengaluru':     (12.9141, 77.6100),
    'Hebbal, Bengaluru':         (13.0358, 77.5970),
    'Jayanagar, Bengaluru':      (12.9250, 77.5938),
    'Peenya, Bengaluru':         (13.0285, 77.5194),
    'Silk Board, Bengaluru':     (12.9177, 77.6228),
    'BWSSB, Bengaluru':          (12.9784, 77.5908),
}

print("LOADING LOCATION DATA")

# Load cleaned air quality data to get station names
aq = pd.read_csv(AQ_FILE)

# Get unique station names
stations = aq['StationId'].unique()

location_map = {}  # maps station_name → location_id

for station in stations:
    city = aq[aq['StationId'] == station]['City'].iloc[0]

    # Look up coordinates; use default if not found
    coords = STATION_COORDS.get(station, (None, None))
    lat, lon = coords

    cursor.execute("""
        INSERT INTO location_dim (station_name, city, latitude, longitude)
        VALUES (?, ?, ?, ?)
    """, (station, city, lat, lon))

    location_map[station] = cursor.lastrowid

conn.commit()
print(f"Loaded {len(location_map)} stations into location_dim")

print("LOADING POLLUTION DATA INTO SQL")

rows_loaded = 0
for _, row in aq.iterrows():
    station = row['StationId']
    loc_id  = location_map.get(station)
    if loc_id is None:
        continue

    cursor.execute("""
        INSERT INTO pollution_fact
            (location_id, date, year, month, season,
             pm25, pm10, no2, co, so2, o3, aqi, aqi_bucket)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        loc_id,
        str(row.get('Date', '')),
        int(row.get('Year', 0)),
        int(row.get('Month', 0)),
        str(row.get('Season', '')),
        float(row.get('PM2.5', 0) or 0),
        float(row.get('PM10', 0) or 0),
        float(row.get('NO2', 0) or 0),
        float(row.get('CO', 0) or 0),
        float(row.get('SO2', 0) or 0),
        float(row.get('O3', 0) or 0),
        float(row.get('AQI', 0) or 0),
        str(row.get('AQI_Bucket', '')),
    ))
    rows_loaded += 1

conn.commit()
print(f"Loaded {rows_loaded} rows into pollution_fact")

print("LOADING TRAFFIC DATA INTO SQL")

traffic = pd.read_csv(TRAF_FILE)

# Add a Bengaluru location entry for traffic
cursor.execute("""
    INSERT INTO location_dim (station_name, city, latitude, longitude)
    VALUES (?, ?, ?, ?)
""", ('Bengaluru_Traffic_Zone', 'Bengaluru', 12.9716, 77.5946))
blr_traffic_loc_id = cursor.lastrowid

rows_loaded = 0
for _, row in traffic.iterrows():
    cursor.execute("""
        INSERT INTO traffic_fact
            (location_id, date, year, month, season,
             area_name, road_intersection, traffic_volume,
             average_speed, congestion_level, road_capacity_util,
             environmental_impact, weather_conditions)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        blr_traffic_loc_id,
        str(row.get('Date', '')),
        int(row.get('Year', 0) if pd.notna(row.get('Year')) else 0),
        int(row.get('Month', 0) if pd.notna(row.get('Month')) else 0),
        str(row.get('Season', '')),
        str(row.get('Area Name', '')),
        str(row.get('Road/Intersection Name', '')),
        float(row.get('Traffic Volume', 0) or 0),
        float(row.get('Average Speed', 0) or 0),
        str(row.get('Congestion Level', '')),
        float(row.get('Road Capacity Utilization', 0) or 0),
        str(row.get('Environmental Impact', '')),
        str(row.get('Weather Conditions', '')),
    ))
    rows_loaded += 1

conn.commit()
print(f" Loaded {rows_loaded} rows into traffic_fact")

print("RUNNING SQL ANALYSIS QUERIES")

# 1: Average PM2.5 by station 
print("\n Query 1: Average PM2.5 by Station")
print("(This answers: Which areas have worst air quality?)")
q1 = pd.read_sql_query("""
    SELECT
        l.station_name,
        l.city,
        ROUND(AVG(p.pm25), 2) AS avg_pm25,
        ROUND(AVG(p.aqi), 2)  AS avg_aqi,
        COUNT(*)               AS num_readings
    FROM pollution_fact p
    JOIN location_dim l ON p.location_id = l.location_id
    GROUP BY l.station_name, l.city
    ORDER BY avg_pm25 DESC
""", conn)
print(q1.to_string(index=False))

#  2: Monthly pollution trend 
print("\n Query 2: Monthly Average PM2.5 (Delhi vs Bengaluru)")
print("(This answers: How does pollution change by month/season?)")
q2 = pd.read_sql_query("""
    SELECT
        l.city,
        p.month,
        p.season,
        ROUND(AVG(p.pm25), 2) AS avg_pm25
    FROM pollution_fact p
    JOIN location_dim l ON p.location_id = l.location_id
    GROUP BY l.city, p.month, p.season
    ORDER BY l.city, p.month
""", conn)
print(q2.to_string(index=False))

#  3: Worst pollution season 
print("\n Query 3: Average PM2.5 by Season and City")
print("(This answers: Which season is most polluted?)")
q3 = pd.read_sql_query("""
    SELECT
        l.city,
        p.season,
        ROUND(AVG(p.pm25), 2) AS avg_pm25,
        ROUND(MAX(p.pm25), 2) AS max_pm25
    FROM pollution_fact p
    JOIN location_dim l ON p.location_id = l.location_id
    GROUP BY l.city, p.season
    ORDER BY l.city, avg_pm25 DESC
""", conn)
print(q3.to_string(index=False))

#  4: Traffic congestion by area
print("\n Query 4: Average Traffic Volume by Area (Bengaluru)")
print("(This answers: Which areas have highest traffic?)")
q4 = pd.read_sql_query("""
    SELECT
        area_name,
        ROUND(AVG(traffic_volume), 0) AS avg_traffic_volume,
        ROUND(AVG(average_speed), 1)  AS avg_speed,
        COUNT(*) AS num_records
    FROM traffic_fact
    GROUP BY area_name
    ORDER BY avg_traffic_volume DESC
    LIMIT 10
""", conn)
print(q4.to_string(index=False))

# 5: Yearly pollution trend 
print("\n Query 5: Yearly Average PM2.5 by City")
print("(This answers: Is pollution getting better or worse over years?)")
q5 = pd.read_sql_query("""
    SELECT
        l.city,
        p.year,
        ROUND(AVG(p.pm25), 2) AS avg_pm25,
        ROUND(AVG(p.aqi), 2)  AS avg_aqi
    FROM pollution_fact p
    JOIN location_dim l ON p.location_id = l.location_id
    WHERE p.year > 0
    GROUP BY l.city, p.year
    ORDER BY l.city, p.year
""", conn)
print(q5.to_string(index=False))

# Save query results for report
q1.to_csv("outputs/sql_query1_station_pm25.csv", index=False)
q2.to_csv("outputs/sql_query2_monthly_trend.csv", index=False)
q3.to_csv("outputs/sql_query3_seasonal.csv", index=False)
q4.to_csv("outputs/sql_query4_traffic_areas.csv", index=False)
q5.to_csv("outputs/sql_query5_yearly.csv", index=False)

print("\n Query results saved to outputs/ folder")

conn.close()


