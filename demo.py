"""Generate one reproducible 1,000,000-row synthetic Parquet dataset, then read it.

Run: uv run demo.py
Outputs: demo.parquet and report/run_.../ (using config.yaml).
The synthetic data demonstrates the mechanics; it is not evidence about your data.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from data_mining import audit


def generate(path, rows=1_000_000, seed=42):
    rng = np.random.default_rng(seed)
    groups = np.arange(rows) // 5
    days = rng.integers(0, 84, groups.max() + 1)[groups]
    dates = pd.Timestamp("2026-01-01") + pd.to_timedelta(days, unit="D")
    rank = (np.arange(rows) % 5 + 1).astype(np.int16)
    apy = np.clip(rng.normal(4.2, .65, rows), .1, 8)
    term = rng.choice([3, 6, 12, 24, 36, 60], rows)
    device = rng.choice([1, 2, 3], rows, p=[.6, .35, .05])
    listing_set = rng.integers(0, 2, rows)
    inventory = rng.integers(0, 2, rows)
    publisher = rng.integers(100, 180, rows)
    account = rng.integers(1000, 1400, rows)
    channel = rng.choice(["organic", "paid", "email", "referral"], rows, p=[.45, .3, .15, .1])
    state = rng.choice(["WI", "CA", "TX", "NY", "FL", "IL", "MN", "WA"], rows)
    session_effect = rng.normal(0, .45, groups.max() + 1)[groups]
    logit = (-3.7 - .45 * (rank - 1) + .65 * (apy - 4.2)
             + .35 * (device == 2) + .2 * (channel == "email")
             + .85 * (listing_set != inventory)
             + .7 * ((apy > 4.7) & (term <= 12))
             + .4 * ((rank <= 2) & (device == 2))
             + .15 * np.sin(publisher) + .25 * days / 84 + session_effect)
    clicks = rng.binomial(1, expit(logit)).astype(np.int8)
    df = pd.DataFrame({
        "rank_pos": rank,
        "apy": apy.astype(np.float32),
        "apy_rank_on_widget": np.clip(np.rint(7 - apy + rng.normal(0, .4, rows)), 1, 6).astype(np.int16),
        "min_investment_amount": rng.choice([0, 500, 1000, 5000, 10000, 25000], rows).astype(np.int32),
        "term_months": term.astype(np.int16),
        "listing_set": listing_set.astype(np.int8),
        "mbt_star_rating": rng.choice([1, 2, 3, 4, 5], rows).astype(np.float32),
        "assets_usd": rng.lognormal(21, 2, rows),
        "popularity_tier_national": rng.choice(["low", "medium", "high"], rows),
        "trust_tier_1to5": rng.integers(1, 6, rows).astype(np.int8),
        "widget_pos": rank.copy(),  # Intentional exact redundant feature.
        "inventory_type": inventory.astype(np.int8),
        "widget_id": rng.integers(10000, 25000, rows).astype(np.int32),
        "device_type_id": device.astype(np.int8),
        "compounding_method": rng.choice(["daily", "monthly", "quarterly", "annually"], rows),
        "state_code": state,
        "advertiser_customer_id": (account // 4).astype(np.int32),
        "advertiser_account_id": account.astype(np.int32),
        "listing_style_id": rng.integers(1, 5, rows).astype(np.int8),
        "publisher.accountid": publisher.astype(np.int16),
        "media_channel": channel,
        "publisher_customer_id": (publisher // 2).astype(np.int16),
        "service_code": rng.choice(["CD", "SAVINGS", "MONEY_MARKET"], rows),
        "week_day": dates.dayofweek.to_numpy().astype(np.int8),
        "clickoccured": clicks,
        "searchid": groups,
        "search_date": dates,
    })
    df.loc[rng.random(rows) < .03, "apy"] = np.nan
    df.loc[rng.random(rows) < .08, "mbt_star_rating"] = np.nan
    df.loc[rng.random(rows) < .04, "state_code"] = None
    df.to_parquet(path, index=False, compression="zstd")
    print(f"Saved {len(df):,} synthetic rows to {path}", flush=True)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    path = root / "demo.parquet"
    generate(path)
    # Deliberately read the saved Parquet through the public two-input interface.
    audit(path, root / "config.yaml")
