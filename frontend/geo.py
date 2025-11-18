import time
import json
from pathlib import Path
from typing import Dict, List, Optional

import requests
import pandas as pd
import pycountry
import plotly.express as px
import streamlit as st

# Default cache path: prefer mounted /app/data for container, fallback to repo ./data
DEFAULT_CACHE_PATH = Path("/app/data/ip_country_cache.json")
if not DEFAULT_CACHE_PATH.parent.exists():
    DEFAULT_CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "ip_country_cache.json"


def load_cache(path: Path = DEFAULT_CACHE_PATH) -> Dict[str, Optional[str]]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(cache: Dict[str, Optional[str]], path: Path = DEFAULT_CACHE_PATH):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        st.warning(f"Could not save IP cache: {e}")


def resolve_ip_batch(
    ips: List[str],
    cache_path: Path = DEFAULT_CACHE_PATH,
    max_per_run: int = 500,
    delay: float = 0.4
) -> Dict[str, Optional[str]]:
    """
    Resolve IP -> country using ip-api.com with on-disk caching.
    - ips: list of IP strings (order preserved in returned dict)
    - max_per_run: number of NEW lookups to perform this run (protects from rate-limits)
    - delay: seconds sleep between requests
    Returns a dict mapping each ip in `ips` to country string or None.
    """
    cache = load_cache(cache_path)
    unique_ips = []
    for ip in ips:
        if ip not in unique_ips:
            unique_ips.append(ip)

    to_query = [ip for ip in unique_ips if ip and ip not in cache]
    if len(to_query) > max_per_run:
        to_query = to_query[:max_per_run]

    progress = None
    if st:
        progress = st.progress(0)

    for i, ip in enumerate(to_query):
        try:
            url = f"http://ip-api.com/json/{ip}?fields=country,status,message"
            r = requests.get(url, timeout=4)
            if r.status_code == 200:
                j = r.json()
                if j.get("status") == "success":
                    cache[ip] = j.get("country") or "Unknown"
                else:
                    cache[ip] = None
            else:
                cache[ip] = None
        except Exception:
            cache[ip] = None

        if progress:
            progress.progress((i + 1) / max(1, len(to_query)))
        time.sleep(delay)

    save_cache(cache, cache_path)
    # Return mapping for the original ips list
    return {ip: cache.get(ip) for ip in ips}


def country_name_to_iso3(name: Optional[str]) -> Optional[str]:
    """
    Convert verbose country names to ISO3 codes. Returns None for unknown/local entries.
    """
    if not name:
        return None
    low = name.strip().lower()
    if low in ("unknown", "local network", "nan", ""):
        return None

    mapping = {
        "united states": "USA",
        "united states of america": "USA",
        "u.s.": "USA",
        "uk": "GBR",
        "russia": "RUS",
        "south korea": "KOR",
        "viet nam": "VNM",
        "czechia": "CZE"
    }
    if name in mapping:
        return mapping[name]

    try:
        c = pycountry.countries.lookup(name)
        return c.alpha_3
    except Exception:
        # try fuzzy match by name lower-case
        try:
            for c in pycountry.countries:
                if c.name.lower() == low:
                    return c.alpha_3
        except Exception:
            return None
    return None


def plot_country_choropleth(country_fraud_df: pd.DataFrame, country_col: str, count_col: str = "FraudCount"):
    """
    country_fraud_df: DataFrame with columns [country_col, count_col]
    country_col should be human readable country names.
    """
    df = country_fraud_df.copy()
    df["iso_a3"] = df[country_col].apply(country_name_to_iso3)
    df = df.dropna(subset=["iso_a3"])
    if df.empty:
        st.info("No mappable countries to plot on the world map.")
        return

    fig = px.choropleth(
        df,
        locations="iso_a3",
        color=count_col,
        hover_name=country_col,
        color_continuous_scale="Reds",
        labels={count_col: "Fraud Count"},
    )
    fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)