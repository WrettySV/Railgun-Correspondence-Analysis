"""
Fetch Shield / Unshield logs from an archive RPC.

Contract addresses and event ABIs must match your target deployment (see
Railgun-Privacy/contract and railgun-community/deployments on GitHub). Pass
JSON with `abi` and `address` or set RAILGUN_CONTRACTS_JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import time

import requests
from web3 import Web3
from web3.exceptions import Web3RPCError
from web3.middleware import ExtraDataToPOAMiddleware
from railgun_analysis.config import load_chain_rpcs
from railgun_analysis.models import PoolId, ShieldRecord, UnshieldRecord
from railgun_analysis.railgun_contracts import (
    DEFAULT_CHAIN_CONFIGS,
    ETHEREUM,
    SHIELD_EVENT_ABI,
    UNSHIELD_EVENT_ABI,
    ethereum_event_abis,
)

_CHAINS_POA_STYLE_BLOCK = frozenset({56, 137})


def _w3(chain_id: int, rpc_url: str | None = None) -> Web3:
    if rpc_url:
        provider = Web3.HTTPProvider(rpc_url)
    else:
        rpcs = load_chain_rpcs()
        if chain_id not in rpcs:
            raise ValueError(
                f"No RPC for chain_id={chain_id}. Set ETH_RPC, BSC_RPC, or POLYGON_RPC."
            )
        provider = Web3.HTTPProvider(rpcs[chain_id])
    w3 = Web3(provider)
    if chain_id in _CHAINS_POA_STYLE_BLOCK:
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    return w3


def load_abi(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "abi" in raw:
        return raw["abi"]
    return raw


def fetch_logs_for_event(
    chain_id: int,
    contract: str,
    event_name: str,
    abi: list[dict[str, Any]],
    from_block: int,
    to_block: int,
) -> list[Any]:
    w3 = _w3(chain_id)
    c = w3.eth.contract(
        address=Web3.to_checksum_address(contract), abi=abi
    )
    ev = getattr(c.events, event_name)
    return list(ev.get_logs(from_block=from_block, to_block=to_block))


def _lg(log: Any, name: str) -> Any:
    if isinstance(log, dict):
        return log[name]
    return getattr(log, name)


def _token_address_from_arg(value: Any) -> str:
    """ABI may emit `token` as address or as TokenData tuple (tokenAddress)."""
    if isinstance(value, str):
        return Web3.to_checksum_address(value)
    if isinstance(value, (bytes, bytearray)):
        return Web3.to_checksum_address(value)
    if isinstance(value, dict):
        inner = value.get("tokenAddress")
    else:
        inner = getattr(value, "tokenAddress", None)
    if inner is None:
        raise TypeError(f"Cannot resolve token address from {type(value)!r}")
    return Web3.to_checksum_address(inner)


def _rpc_error_retriable(exc: BaseException) -> bool:
    """True for rate limits, oversized ranges, and transient provider failures."""
    msg = str(exc).lower()
    if any(
        s in msg
        for s in (
            "too many requests",
            "429",
            "block range",
            "-32005",
            "temporarily unavailable",
            "service unavailable",
            "rate limit",
            "timeout",
            "timed out",
        )
    ):
        return True
    if "-32603" in msg:
        return True
    if isinstance(exc, Web3RPCError) and exc.args and isinstance(exc.args[0], dict):
        code = exc.args[0].get("code")
        if code in (-32603, -32005):
            return True
    return False


def _chunked_get_logs(
    event_proxy, start_block: int, end_block: int, *, step: int = 1000, max_retries: int = 6
):
    """
    Some providers reject wide ranges; fetch in chunks with simple backoff.
    """
    current = start_block
    if end_block == "latest":
        latest = int(event_proxy.w3.eth.block_number)
    else:
        latest = int(end_block)
    current_step = step
    while current <= latest:
        hi = min(latest, current + current_step - 1)
        retries = 0
        while True:
            try:
                yield from event_proxy.get_logs(from_block=current, to_block=hi)
                current = hi + 1
                current_step = step  # reset after success
                break
            except (Web3RPCError, requests.HTTPError) as e:
                if retries < max_retries and _rpc_error_retriable(e):
                    retries += 1
                    current_step = max(100, current_step // 2)
                    time.sleep(min(2.0 * retries, 15.0))
                    continue
                raise


def _get_block_with_backoff(
    w3: Web3, block_number: int, *, max_retries: int = 5
) -> dict[str, Any]:
    """
    Fetch a block with retries so timestamp enrichment survives RPC rate limits.
    """
    retries = 0
    while True:
        try:
            return dict(w3.eth.get_block(block_number))
        except (Web3RPCError, requests.HTTPError) as e:
            if retries < max_retries and _rpc_error_retriable(e):
                retries += 1
                time.sleep(min(2**retries, 10))
                continue
            raise


def _timestamp_cache_path(chain_id: int) -> Path:
    return Path(__file__).resolve().parents[1] / ".cache" / f"timestamps_{chain_id}.json"


def _load_timestamp_cache(path: Path) -> dict[int, int]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    return {int(block): int(ts) for block, ts in raw.items()}


def _save_timestamp_cache(path: Path, cache: dict[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    serializable = {str(block): ts for block, ts in sorted(cache.items())}
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, separators=(",", ":"))
    tmp_path.replace(path)


def parse_event_to_record(
    chain_id: int,
    contract: str,
    pool_index: int,
    event_name: str,
    log: Any,
    *,
    token_field: str = "token",
    amount_field: str = "amount",
    ts_from_block: dict[int, int] | None = None,
) -> ShieldRecord | UnshieldRecord:
    """Map decoded args to our records; adjust field names to your ABI."""
    args = _lg(log, "args")
    if not isinstance(args, dict):
        args = dict(args)
    token = _token_address_from_arg(args[token_field])
    amount_raw = int(args[amount_field])
    block_number = int(_lg(log, "blockNumber"))
    th = _lg(log, "transactionHash")
    tx_hash = th.hex() if hasattr(th, "hex") else str(th)
    log_index = int(_lg(log, "logIndex"))
    ts = 0
    if ts_from_block and block_number in ts_from_block:
        ts = ts_from_block[block_number]
    pool = PoolId(chain_id, contract.lower(), pool_index)
    if event_name.lower().startswith("shield"):
        return ShieldRecord(
            tx_hash=tx_hash,
            log_index=log_index,
            block_number=block_number,
            timestamp=ts,
            pool=pool,
            token=token,
            amount_raw=amount_raw,
        )
    return UnshieldRecord(
        tx_hash=tx_hash,
        log_index=log_index,
        block_number=block_number,
        timestamp=ts,
        pool=pool,
        token=token,
        amount_raw=amount_raw,
    )


def enrich_timestamps(
    w3: Web3, block_numbers: list[int], *, cache_path: Path | None = None
) -> dict[int, int]:
    """block_number -> unix timestamp (batch with care for RPC limits)."""
    wanted = sorted(set(block_numbers))
    if not wanted:
        return {}

    out = _load_timestamp_cache(cache_path) if cache_path else {}
    dirty = False
    for bn in wanted:
        if bn in out:
            continue
        blk = _get_block_with_backoff(w3, bn)
        out[bn] = int(blk["timestamp"])
        dirty = True

    if cache_path and dirty:
        _save_timestamp_cache(cache_path, out)

    out = {bn: out[bn] for bn in wanted}
    return out


# ---------- Railgun-specific helpers ----------


def fetch_railgun_eth_logs(
    from_block: int | None = None,
    to_block: int | None = None,
    *,
    chain_id: int = ETHEREUM["chain_id"],
    pool_index: int = 0,
    proxy_address: str | None = None,
    rpc_url: str | None = None,
) -> tuple[list[ShieldRecord], list[UnshieldRecord]]:
    """
    Fetch Shield / Unshield logs for a Railgun proxy on supported chains.
    """
    cfg = DEFAULT_CHAIN_CONFIGS.get(chain_id, {})

    if from_block is None:
        if "deployment_block" not in cfg:
            raise ValueError(
                f"from_block is required for chain_id={chain_id} when default deployment is unknown."
            )
        start_block = int(cfg["deployment_block"])
    else:
        start_block = from_block
    end_block = to_block or "latest"

    w3 = _w3(chain_id, rpc_url=rpc_url)
    proxy = proxy_address or cfg.get("proxy_address")
    if not proxy:
        raise ValueError(
            f"proxy_address is required for chain_id={chain_id} when default proxy is unknown."
        )
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(proxy),
        abi=ethereum_event_abis(),
    )

    shields_logs = list(
        _chunked_get_logs(contract.events.Shield, start_block, end_block)
    )
    unshields_logs = list(
        _chunked_get_logs(contract.events.Unshield, start_block, end_block)
    )

    # Fetch timestamps once for all blocks seen
    blocks = {int(log["blockNumber"]) for log in shields_logs + unshields_logs}
    ts_map = enrich_timestamps(
        w3,
        list(blocks),
        cache_path=_timestamp_cache_path(chain_id),
    )

    shields: list[ShieldRecord] = []
    for log in shields_logs:
        recs = parse_shield_commitments(
            chain_id,
            proxy,
            pool_index,
            log,
            ts_from_block=ts_map,
        )
        shields.extend(recs)

    unshields = [
        parse_event_to_record(
            chain_id,
            proxy,
            pool_index,
            "Unshield",
            log,
            token_field="token",
            amount_field="amount",
            ts_from_block=ts_map,
        )
        for log in unshields_logs
    ]
    return shields, unshields


def check_proxy_event_counts(
    *,
    chain_id: int,
    proxy_address: str,
    from_block: int,
    to_block: int,
    rpc_url: str | None = None,
) -> tuple[int, int]:
    """
    Fast sanity check for candidate proxy: count raw Shield/Unshield logs only.
    """
    w3 = _w3(chain_id, rpc_url=rpc_url)
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(proxy_address),
        abi=ethereum_event_abis(),
    )
    shields_logs = list(
        _chunked_get_logs(contract.events.Shield, from_block, to_block)
    )
    unshields_logs = list(
        _chunked_get_logs(contract.events.Unshield, from_block, to_block)
    )
    return len(shields_logs), len(unshields_logs)


def parse_shield_commitments(
    chain_id: int,
    contract: str,
    pool_index: int,
    log: Any,
    *,
    ts_from_block: dict[int, int] | None = None,
) -> list[ShieldRecord]:
    """
    Shield event contains an array of commitments; we emit one ShieldRecord per commitment.
    """
    args = _lg(log, "args")
    if not isinstance(args, dict):
        args = dict(args)
    commitments = args.get("commitments", [])

    block_number = int(_lg(log, "blockNumber"))
    ts = 0
    if ts_from_block and block_number in ts_from_block:
        ts = ts_from_block[block_number]
    tx_hash_hex = _lg(log, "transactionHash")
    tx_hash = tx_hash_hex.hex() if hasattr(tx_hash_hex, "hex") else str(tx_hash_hex)
    base_pool = PoolId(chain_id, contract.lower(), pool_index)

    records: list[ShieldRecord] = []
    for idx, cm in enumerate(commitments):
        token_info = cm["token"]
        token_addr = Web3.to_checksum_address(token_info["tokenAddress"])
        value = int(cm["value"])
        records.append(
            ShieldRecord(
                tx_hash=tx_hash,
                log_index=int(_lg(log, "logIndex")),
                block_number=block_number,
                timestamp=ts,
                pool=base_pool,
                token=token_addr,
                amount_raw=value,
                extra={"commitment_index": idx},
            )
        )
    return records
