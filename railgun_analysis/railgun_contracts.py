from __future__ import annotations

from typing import Final, List, Dict, Any

ETHEREUM = {
    "chain_id": 1,
    # Proxy is the user-facing contract emitting events.
    "proxy_address": "0xfa7093cdd9ee6932b4eb2c9e1cde7ce00b1fa4b9",
    "deployment_block": 14693013,
}

BSC = {
    "chain_id": 56,
    "proxy_address": "0x590162bf4b50f6576a459b75309ee21d92178a10",
    "deployment_block": 17431925,
}

DEFAULT_CHAIN_CONFIGS: Final[Dict[int, Dict[str, Any]]] = {
    ETHEREUM["chain_id"]: ETHEREUM,
    BSC["chain_id"]: BSC,
}

# Minimal ABIs needed to decode the two public events we care about.
# Full ABI is large; for log decoding we only need the event fragments.
SHIELD_EVENT_ABI: Final[Dict[str, Any]] = {
    "anonymous": False,
    "inputs": [
        {"indexed": False, "internalType": "uint256", "name": "treeNumber", "type": "uint256"},
        {"indexed": False, "internalType": "uint256", "name": "startPosition", "type": "uint256"},
        {
            "indexed": False,
            "internalType": "struct CommitmentPreimage[]",
            "name": "commitments",
            "type": "tuple[]",
            "components": [
                {"internalType": "bytes32", "name": "npk", "type": "bytes32"},
                {
                    "internalType": "struct TokenData",
                    "name": "token",
                    "type": "tuple",
                    "components": [
                        {"internalType": "enum TokenType", "name": "tokenType", "type": "uint8"},
                        {"internalType": "address", "name": "tokenAddress", "type": "address"},
                        {"internalType": "uint256", "name": "tokenSubID", "type": "uint256"},
                    ],
                },
                {"internalType": "uint120", "name": "value", "type": "uint120"},
            ],
        },
        {
            "indexed": False,
            "internalType": "struct ShieldCiphertext[]",
            "name": "shieldCiphertext",
            "type": "tuple[]",
            "components": [
                {"internalType": "bytes32[3]", "name": "encryptedBundle", "type": "bytes32[3]"},
                {"internalType": "bytes32", "name": "shieldKey", "type": "bytes32"},
            ],
        },
    ],
    "name": "Shield",
    "type": "event",
}

UNSHIELD_EVENT_ABI: Final[Dict[str, Any]] = {
    "anonymous": False,
    "inputs": [
        {"indexed": False, "internalType": "address", "name": "to", "type": "address"},
        {
            "indexed": False,
            "internalType": "struct TokenData",
            "name": "token",
            "type": "tuple",
            "components": [
                {"internalType": "enum TokenType", "name": "tokenType", "type": "uint8"},
                {"internalType": "address", "name": "tokenAddress", "type": "address"},
                {"internalType": "uint256", "name": "tokenSubID", "type": "uint256"},
            ],
        },
        {"indexed": False, "internalType": "uint256", "name": "amount", "type": "uint256"},
        {"indexed": False, "internalType": "uint256", "name": "fee", "type": "uint256"},
    ],
    "name": "Unshield",
    "type": "event",
}


def ethereum_event_abis() -> List[Dict[str, Any]]:
    """Return list of event ABIs for constructing a contract in web3."""
    return [SHIELD_EVENT_ABI, UNSHIELD_EVENT_ABI]
