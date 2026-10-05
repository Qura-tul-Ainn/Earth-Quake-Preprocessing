"""Task 1: fetch and parse USGS earthquake data."""

import requests
import pandas as pd


def fetch_feed(feed="all_week"):
    """Fetch a USGS earthquake GeoJSON feed."""
    url = f"https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/{feed}.geojson"

    response = requests.get(url, timeout=15)
    response.raise_for_status()

    return response.json()


def geojson_to_df(payload):
    """Convert USGS GeoJSON payload into a pandas DataFrame."""

    rows = []

    for feature in payload["features"]:
        properties = feature["properties"]
        coordinates = feature["geometry"]["coordinates"]

        row = {
            "id": feature["id"],
            **properties,
            "lon": coordinates[0],
            "lat": coordinates[1],
            "depth_km": coordinates[2],
        }

        rows.append(row)

    return pd.DataFrame(rows)