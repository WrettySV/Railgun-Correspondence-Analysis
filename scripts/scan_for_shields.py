"""
Сканирует интервалы блоков назад от указанного блока и ищет окно,
в котором есть хотя бы один Shield. Удобно, чтобы не писать команды вручную.

Пример запуска:
    export ETH_RPC="https://rpc.ankr.com/eth/7c7f61bbabfd160fb834b28c209a51537782640ff7fe2833b7e9cf003da188bd" 
    cd "/Users/tevaluuu/Blockchain Project 2"
    . .venv/bin/activate
    python scripts/scan_for_shields.py \
        --proxy-address 0xfa7093cdd9ee6932b4eb2c9e1cde7ce00b1fa4b9 \
        --chain-id 1 \
        --end-block latest \
        --window 200000 \
        --max-windows 20

По умолчанию: цепь 1 (Ethereum), окно 200k блоков, смотрим 10 окон назад от latest.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from railgun_analysis.collector import _w3, fetch_railgun_eth_logs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Скан назад по блокам для поиска Shield.")
    p.add_argument("--chain-id", type=int, default=1, help="Chain ID (default: 1, Ethereum)")
    p.add_argument("--proxy-address", required=True, help="Railgun proxy address")
    p.add_argument("--rpc-url", default=None, help="RPC override (если не задан, возьмётся ETH_RPC и др.)")
    p.add_argument(
        "--end-block",
        default="latest",
        help="С какого блока начинать назад (default: latest)",
    )
    p.add_argument(
        "--window",
        type=int,
        default=200_000,
        help="Размер окна (блоков) в одном запросе (default: 200k)",
    )
    p.add_argument(
        "--max-windows",
        type=int,
        default=10,
        help="Сколько окон смотреть назад (default: 10)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    chain_id: int = args.chain_id
    proxy_address: str = args.proxy_address
    rpc_url: Optional[str] = args.rpc_url
    window: int = args.window
    max_windows: int = args.max_windows
    end_block = args.end_block

    # Определить верхнюю границу (latest -> номер)
    if end_block == "latest":
        w3 = _w3(chain_id, rpc_url=rpc_url)
        end_block_int = int(w3.eth.block_number)
    else:
        end_block_int = int(end_block)

    current_hi = end_block_int
    first_found: Optional[tuple[int, int, int, int]] = None

    for i in range(max_windows):
        lo = max(0, current_hi - window)
        print(f"\n[scan] range {lo} -> {current_hi}")
        shields, unshields = fetch_railgun_eth_logs(
            from_block=lo,
            to_block=current_hi,
            chain_id=chain_id,
            proxy_address=proxy_address,
            rpc_url=rpc_url,
        )
        print(f"Shields={len(shields)}, Unshields={len(unshields)}")
        if shields:
            if first_found is None:
                first_found = (lo, current_hi, len(shields), len(unshields))
            print("[scan] Найдены Shield в этом диапазоне.")
        current_hi = lo - 1
        if current_hi <= 0:
            break

    if first_found is None:
        print(f"[scan] В {max_windows} окнах по {window} блоков Shield не нашли. Расширьте поиск назад.")
    else:
        lo, hi, nsh, nu = first_found
        print(f"[scan] Первое окно с Shield: {lo} -> {hi} (Shields={nsh}, Unshields={nu})")


if __name__ == "__main__":
    main()
