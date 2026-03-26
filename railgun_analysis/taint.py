
from __future__ import annotations

from collections.abc import Iterable

from railgun_analysis.models import ShieldRecord, UnshieldRecord


def feasible_shields_for_unshield(
    unshield: UnshieldRecord,
    shields: Iterable[ShieldRecord],
    *,
    require_same_pool: bool = True,
    require_same_token: bool = True,
) -> list[ShieldRecord]:
    """
    Return Shield events that strictly precede the Unshield in time and match
    pool (and token) when requested.
    """
    out: list[ShieldRecord] = []
    for s in shields:
        if require_same_pool and s.pool.key() != unshield.pool.key():
            continue
        if require_same_token and s.token.lower() != unshield.token.lower():
            continue
        if s.timestamp >= unshield.timestamp:
            continue
        out.append(s)
    return sorted(out, key=lambda x: (x.timestamp, x.block_number, x.log_index))


def build_time_ordered_edges(
    shields: list[ShieldRecord],
    unshields: list[UnshieldRecord],
    *,
    require_same_pool: bool = True,
    require_same_token: bool = True,
) -> dict[str, list[ShieldRecord]]:
    """
    For each Unshield tx key (tx_hash:log_index), list feasible predecessor Shields.
    """
    shield_list = list(shields)
    result: dict[str, list[ShieldRecord]] = {}
    for u in unshields:
        key = f"{u.tx_hash}:{u.log_index}"
        result[key] = feasible_shields_for_unshield(
            u,
            shield_list,
            require_same_pool=require_same_pool,
            require_same_token=require_same_token,
        )
    return result
