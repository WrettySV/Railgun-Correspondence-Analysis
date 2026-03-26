from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChainRpc:
    chain_id: int
    rpc_url: str


def load_chain_rpcs() -> dict[int, str]:
    """Map chain_id -> RPC URL. Uses ETH_RPC, BSC_RPC, POLYGON_RPC if set."""
    out: dict[int, str] = {}
    if url := os.environ.get("ETH_RPC"):
        out[1] = url
    if url := os.environ.get("BSC_RPC"):
        out[56] = url
    if url := os.environ.get("POLYGON_RPC"):
        out[137] = url
    return out


def load_contracts_json(path: str | Path | None = None) -> list[dict]:
    """Optional JSON list of {chain_id, address, pool_index?, label?}."""
    p = path or os.environ.get("RAILGUN_CONTRACTS_JSON")
    if not p:
        return []
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("contracts", [])
