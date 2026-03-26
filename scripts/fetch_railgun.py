from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from railgun_analysis.collector import check_proxy_event_counts, fetch_railgun_eth_logs
from railgun_analysis.statistics import records_to_frame, summarize_amounts
from railgun_analysis.models import ShieldRecord, UnshieldRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-block", type=int, default=None, help="Start block (default: deploy block or latest-10000)")
    parser.add_argument("--to-block", default="latest", help="End block (default: latest)")
    parser.add_argument(
        "--chain-id",
        type=int,
        default=1,
        help="Chain ID: 1=Ethereum, 56=BSC, 137=Polygon (default: 1).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="If set, iterate by this block size from --from-block to --to-block.",
    )
    parser.add_argument(
        "--proxy-address",
        help="Override Railgun proxy address (default: hardcoded ETH mainnet proxy).",
    )
    parser.add_argument(
        "--pool-index",
        type=int,
        default=0,
        help="Pool index label to attach to records (default: 0).",
    )
    parser.add_argument(
        "--rpc-url",
        default=None,
        help="Optional RPC URL override. If omitted, uses ETH_RPC/BSC_RPC/POLYGON_RPC by chain.",
    )
    parser.add_argument(
        "--check-proxy",
        action="store_true",
        help="Only check raw Shield/Unshield log counts for the proxy and exit.",
    )
    parser.add_argument(
        "--check-window",
        type=int,
        default=10_000,
        help="Block window for --check-proxy when --from-block is not set (default: 10000).",
    )
    parser.add_argument(
        "--csv",
        help="Optional path to save combined Shield/Unshield events as CSV.",
    )
    parser.add_argument(
        "--json",
        help="Optional path to save combined Shield/Unshield events as JSON list.",
    )
    parser.add_argument(
        "--recent-blocks",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Use only the last N blocks up to chain head: sets from_block=latest-N "
            "and to_block=latest (numeric). Overrides --from-block. "
            "Useful for Polygon/BSC where full history is huge."
        ),
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help=(
            "Merge with existing --csv / --json if the file is already there "
            "(dedupe by tx_hash + log_index + kind), then rewrite both outputs."
        ),
    )
    return parser.parse_args()


def default_recent_window(w3) -> tuple[int, str]:
    latest = int(w3.eth.block_number)
    return max(0, latest - 10_000), "latest"


def main() -> None:
    args = parse_args()
    from_block = args.from_block
    to_block = args.to_block
    chain_id = args.chain_id
    step = args.step
    proxy_address = args.proxy_address
    pool_index = args.pool_index
    rpc_url = args.rpc_url
    check_proxy = args.check_proxy
    check_window = args.check_window
    csv_path = args.csv
    json_path = args.json
    recent_blocks = args.recent_blocks
    append_mode = args.append

    if recent_blocks is not None:
        from railgun_analysis.collector import _w3  # type: ignore
        w3 = _w3(chain_id, rpc_url=rpc_url)
        latest = int(w3.eth.block_number)
        from_block = max(0, latest - recent_blocks)
        to_block = latest
        print(
            f"[info] Recent window: last {recent_blocks} blocks "
            f"-> from_block={from_block}, to_block={to_block}"
        )
    # If user didn't provide from_block, pick a recent window.
    elif from_block is None:
        # cheap Web3 creation just to get latest; reuse inside helper not exposed,
        # so we import _w3 lazily to avoid exporting it.
        from railgun_analysis.collector import _w3  # type: ignore
        w3 = _w3(chain_id, rpc_url=rpc_url)
        latest = int(w3.eth.block_number)
        if check_proxy:
            from_block = max(0, latest - check_window)
            to_block = latest
        else:
            fb, tb = default_recent_window(w3)
            from_block = fb
            to_block = tb
        print(f"[info] Using recent window: from_block={from_block}, to_block={to_block}")

    if check_proxy:
        if not proxy_address:
            raise ValueError("--check-proxy requires --proxy-address.")
        if to_block == "latest":
            from railgun_analysis.collector import _w3  # type: ignore
            tb_int = int(_w3(chain_id, rpc_url=rpc_url).eth.block_number)
        else:
            tb_int = int(to_block)
        shield_count, unshield_count = check_proxy_event_counts(
            chain_id=chain_id,
            proxy_address=proxy_address,
            from_block=int(from_block),
            to_block=tb_int,
            rpc_url=rpc_url,
        )
        print(f"Raw Shield logs: {shield_count}")
        print(f"Raw Unshield logs: {unshield_count}")
        return

    all_rows: list[dict] = []

    def run_range(fb: int, tb: int) -> None:
        shields, unshields = fetch_railgun_eth_logs(
            from_block=fb,
            to_block=tb,
            chain_id=chain_id,
            pool_index=pool_index,
            proxy_address=proxy_address,
            rpc_url=rpc_url,
        )
        print(f"\nRange {fb} -> {tb}: Shields={len(shields)}, Unshields={len(unshields)}")
        df = records_to_frame(shields, unshields)
        if not df.empty:
            print("Amount summary:")
            print(summarize_amounts(df).head())
        else:
            print("No events in this window.")

        if csv_path or json_path:
            rows = _records_to_rows(shields, unshields)
            all_rows.extend(rows)

    if step is None or to_block == "latest":
        run_range(from_block, to_block)
    else:
        current = from_block
        end = int(to_block)
        while current <= end:
            hi = min(end, current + step)
            run_range(current, hi)
            current = hi + 1

    if append_mode and (csv_path or json_path):
        prior = _load_rows_for_append(csv_path, json_path)
        if prior:
            n0, n1 = len(prior), len(all_rows)
            all_rows = _merge_dedupe_rows(prior, all_rows)
            print(
                f"[append] Loaded {n0} existing rows, {n1} from this run "
                f"-> {len(all_rows)} total after dedupe"
            )

    if csv_path:
        _write_csv(csv_path, all_rows)
        print(f"[write] CSV saved to {csv_path} ({len(all_rows)} rows)")
    if json_path:
        _write_json(json_path, all_rows)
        print(f"[write] JSON saved to {json_path} ({len(all_rows)} rows)")


def _records_to_rows(shields: list[ShieldRecord], unshields: list[UnshieldRecord]) -> list[dict]:
    rows: list[dict] = []
    for s in shields:
        rows.append(
            {
                "kind": "shield",
                "tx_hash": s.tx_hash,
                "log_index": s.log_index,
                "block_number": s.block_number,
                "timestamp": s.timestamp,
                "chain_id": s.pool.chain_id,
                # Public identifier for the Railgun instance on this chain.
                "contract_address": s.pool.contract,
                "token": s.token,
                "amount_raw": s.amount_raw,
                "extra": s.extra,
            }
        )
    for u in unshields:
        rows.append(
            {
                "kind": "unshield",
                "tx_hash": u.tx_hash,
                "log_index": u.log_index,
                "block_number": u.block_number,
                "timestamp": u.timestamp,
                "chain_id": u.pool.chain_id,
                # Public identifier for the Railgun instance on this chain.
                "contract_address": u.pool.contract,
                "token": u.token,
                "amount_raw": u.amount_raw,
                "extra": u.extra,
            }
        )
    return rows


def _row_dedupe_key(row: dict) -> tuple:
    li = row.get("log_index")
    try:
        li = int(li) if li is not None and li != "" else -1
    except (TypeError, ValueError):
        li = -1
    return (row.get("tx_hash"), li, row.get("kind"))


def _merge_dedupe_rows(existing: list[dict], new: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    out: list[dict] = []
    for r in existing + new:
        k = _row_dedupe_key(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def _load_rows_for_append(csv_path: str | None, json_path: str | None) -> list[dict]:
    """Prefer CSV if present (same schema); else JSON."""
    if csv_path and Path(csv_path).is_file():
        with open(csv_path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    if json_path and Path(json_path).is_file():
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    return []


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    # Use union of keys so schema changes (e.g. added/removed fields) don't drop columns.
    fieldnames: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
