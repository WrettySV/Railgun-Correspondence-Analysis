"""Distributions of amounts, inter-event times, and pool activity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from railgun_analysis.models import ShieldRecord, UnshieldRecord


def records_to_frame(
    shields: list[ShieldRecord],
    unshields: list[UnshieldRecord],
) -> pd.DataFrame:
    rows: list[dict] = []
    for s in shields:
        rows.append(
            {
                "kind": "shield",
                "chain_id": s.pool.chain_id,
                # Public proxy/contract address of the Railgun instance on this chain.
                "contract_address": s.pool.contract,
                "token": s.token.lower(),
                "amount_raw": s.amount_raw,
                "timestamp": s.timestamp,
                "block": s.block_number,
            }
        )
    for u in unshields:
        rows.append(
            {
                "kind": "unshield",
                "chain_id": u.pool.chain_id,
                "contract_address": u.pool.contract,
                "token": u.token.lower(),
                "amount_raw": u.amount_raw,
                "timestamp": u.timestamp,
                "block": u.block_number,
            }
        )
    return pd.DataFrame(rows)


def inter_event_times_seconds(df: pd.DataFrame) -> pd.Series:
    """Per (chain_id, contract_address, token), sorted by time, delta seconds between consecutive events."""
    if df.empty:
        return pd.Series(dtype=float)
    parts: list[pd.Series] = []
    for _, g in df.sort_values("timestamp").groupby(
        ["chain_id", "contract_address", "token"], sort=False
    ):
        t = g["timestamp"].astype(np.int64)
        parts.append(t.diff().dropna())
    return pd.concat(parts, ignore_index=True)


def summarize_amounts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "amount_raw" not in df.columns:
        return pd.DataFrame()
    return df.groupby(["kind", "chain_id", "contract_address", "token"])["amount_raw"].describe()


def activity_counts(df: pd.DataFrame, freq: str = "1D") -> pd.DataFrame:
    """Event counts per pool/token resampled by pandas offset alias (e.g. 1D, 1h)."""
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return (
        df.set_index("ts")
        .groupby(["chain_id", "pool", "token", "kind"])
        .resample(freq)
        .size()
        .unstack(fill_value=0)
    )
