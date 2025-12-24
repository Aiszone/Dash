from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

CSV_FILES: Dict[str, str] = {
    "SP500": "SP500.csv",
    "NASDAQ100": "Nasdaq100.csv",
    "CAC40": "CAC40.csv",
    "STOXX600": "EUR600.csv",
}


def _parse_holdings_line(line: str) -> list[str] | None:
    s = line.strip()
    if not s:
        return None
    if s.startswith("Ticker") or s.startswith("\ufeffTicker"):
        return None
    if "Fund Holdings as of" in s:
        return None

    s = s.lstrip("\ufeff").lstrip('"').rstrip('"').replace('""', '"')

    import csv

    row = next(csv.reader([s]))
    if len(row) < 2:
        return None
    row += [""] * (12 - len(row))
    return row[:12]


def _yf_suffix(exchange: str, location: str) -> str:
    ex = (exchange or "").lower()
    loc = (location or "").lower()

    if "paris" in ex or ("euronext" in ex and "france" in loc):
        return ".PA"
    if "brussels" in ex:
        return ".BR"
    if "amsterdam" in ex:
        return ".AS"
    if "borsa italiana" in ex or "italiana" in ex:
        return ".MI"
    if "xetra" in ex or "frankfurt" in ex or "boerse" in ex:
        return ".DE"
    if "london" in ex:
        return ".L"
    if "madrid" in ex:
        return ".MC"
    if "swiss" in ex:
        return ".SW"
    if "oslo" in ex:
        return ".OL"
    if "lisbon" in ex:
        return ".LS"
    if "helsinki" in ex:
        return ".HE"
    if "copenhagen" in ex:
        return ".CO"
    if "stockholm" in ex or "omx" in ex:
        return ".ST"
    if "nasdaq" in ex or "new york" in ex or "nyse" in ex:
        return ""
    return ""


def to_yf_ticker(ticker: str, exchange: str, location: str) -> str:
    t = (ticker or "").strip()
    if not t:
        return ""
    if "." in t:
        return t
    return f"{t}{_yf_suffix(exchange, location)}"


@st.cache_data(show_spinner=False)
def load_equity_universe() -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parents[1]
    rows: List[dict] = []

    for index_name, filename in CSV_FILES.items():
        path = base_dir / filename
        if not path.exists():
            continue

        with path.open(encoding="utf-8") as f:
            for raw_line in f:
                parsed = _parse_holdings_line(raw_line)
                if parsed is None:
                    continue

                (
                    ticker,
                    name,
                    sector,
                    asset_class,
                    _market_value,
                    _weight,
                    _notional,
                    _shares,
                    _price,
                    location,
                    exchange,
                    currency,
                ) = parsed

                rows.append(
                    {
                        "ticker": (ticker or "").strip(),
                        "name": (name or "").strip(),
                        "sector": (sector or "").strip(),
                        "asset_class": (asset_class or "").strip(),
                        "location": (location or "").strip(),
                        "exchange": (exchange or "").strip(),
                        "currency": (currency or "").strip(),
                        "index": index_name,
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "ticker",
                "name",
                "indices",
                "sector",
                "asset_class",
                "location",
                "exchange",
                "currency",
                "yf_ticker",
            ]
        )

    df = pd.DataFrame(rows)

    df = (
        df.groupby(["ticker", "name", "sector", "asset_class", "location", "exchange", "currency"])["index"]
        .apply(lambda s: ", ".join(sorted(set(s))))
        .reset_index()
        .rename(columns={"index": "indices"})
    )

    df["yf_ticker"] = df.apply(lambda r: to_yf_ticker(r["ticker"], r["exchange"], r["location"]), axis=1)
    df = df.sort_values(["ticker", "name"]).reset_index(drop=True)
    return df