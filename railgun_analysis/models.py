from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PoolId:
    """Identifies a Railgun liquidity pool (chain + contract + optional pool index)."""

    chain_id: int
    contract: str
    pool_index: int = 0

    def key(self) -> tuple[int, str, int]:
        return (self.chain_id, self.contract.lower(), self.pool_index)


@dataclass
class ShieldRecord:
    """Public observables for a Shield (deposit) after indexing."""

    tx_hash: str
    log_index: int
    block_number: int
    timestamp: int
    pool: PoolId
    token: str
    amount_raw: int
    from_address: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnshieldRecord:
    """Public observables for an Unshield (withdrawal) after indexing."""

    tx_hash: str
    log_index: int
    block_number: int
    timestamp: int
    pool: PoolId
    token: str
    amount_raw: int
    to_address: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
