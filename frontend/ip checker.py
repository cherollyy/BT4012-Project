import argparse
from pathlib import Path
import pandas as pd
import folium
from folium.plugins import HeatMap
import requests
import time
import sys

#File Path
ROOT = Path(__file__).resolve().parents[1]
FILE_NAME = ROOT / "data" / "Fraudulent_E-Commerce_Transaction_Data_2.csv"
IP_COLUMN = "IP Address"
OUTPUT_FILE = Path(__file__).resolve().parent / "ip_heatmap.html"

SLEEP_TIME = 1.5
SAMPLE_DEFAULT = 50

def get_lat_lon(ip):
    """Fetches latitude and longitude for a given IP address."""
    try:
        url = f"http://ip-api.com/json/{ip}"
        response = requests.get(url, timeout=5)
        data = response.json()
        if data['status'] == 'success':
            return data['lat'], data['lon']
    except Exception as e:
        print(f"Error fetching {ip}: {e}")
    return None

def main():
    print("Loading data...")
    df = pd.read_csv(FILE_NAME)

    sample_df = df.head(50) 
    
    print(f"Processing {len(sample_df)} IP addresses...")
    locations = []

    for index, row in sample_df.iterrows():
        ip = row[IP_COLUMN]
        coords = get_lat_lon(ip)
        
        if coords:
            locations.append(coords)
            print(f"[{index+1}] {ip} -> {coords}")
        else:
            print(f"[{index+1}] {ip} -> Not found")
        
        # Pause to respect API limits
        time.sleep(SLEEP_TIME)

    if locations:
        print(f"Generating map with {len(locations)} locations...")
        # Center map on the average location
        avg_lat = sum(x[0] for x in locations) / len(locations)
        avg_lon = sum(x[1] for x in locations) / len(locations)
        
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=2)
        HeatMap(locations, radius=15, blur=10).add_to(m)
        
        m.save(OUTPUT_FILE)
        print(f"Success! Open '{OUTPUT_FILE}' in your browser to see the heatmap.")
    else:
        print("No valid locations found.")

if __name__ == "__main__":
    main()