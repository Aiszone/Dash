# lib/__init__.py
from .data import fetch_series, to_monthly_bm
from .portfolio import (
    build_prices_usd_monthly,
    weights_vector,
    nav_base_10k,
    compute_stats_from_nav,
    benchmarks_base_10k,
    relative_stats,
    BASE,
)