from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from railgun_analysis.anonymity import anonymity_metric
from railgun_analysis.models import PoolId, ShieldRecord, UnshieldRecord
from railgun_analysis.statistics import inter_event_times_seconds, records_to_frame
from railgun_analysis.taint import feasible_shields_for_unshield


CHAIN_LABELS: dict[int, str] = {
    1: "Ethereum mainnet",
    56: "BSC",
    137: "Polygon",
}

NETWORK_ALIASES: dict[str, tuple[int, str]] = {
    "eth": (1, "railgun_window_eth.json"),
    "ethereum": (1, "railgun_window_eth.json"),
    "bsc": (56, "railgun_window_bsc.json"),
    "polygon": (137, "railgun_window_poly.json"),
    "poly": (137, "railgun_window_poly.json"),
}

# Human-readable token metadata for common assets in collected windows.
TOKEN_METADATA: dict[int, dict[str, dict[str, Any]]] = {
    1: {
        "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": {"symbol": "WETH", "decimals": 18, "kind": "wrapped native"},
        "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": {"symbol": "USDC", "decimals": 6, "kind": "ERC-20"},
        "0x6b175474e89094c44da98b954eedeac495271d0f": {"symbol": "DAI", "decimals": 18, "kind": "ERC-20"},
        "0xdac17f958d2ee523a2206206994597c13d831ec7": {"symbol": "USDT", "decimals": 6, "kind": "ERC-20"},
        "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": {"symbol": "WBTC", "decimals": 8, "kind": "wrapped BTC"},
    },
    56: {
        "0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c": {"symbol": "WBNB", "decimals": 18, "kind": "wrapped native"},
        "0xe9e7cea3dedca5984780bafc599bd69add087d56": {"symbol": "BUSD", "decimals": 18, "kind": "ERC-20"},
        "0x55d398326f99059ff775485246999027b3197955": {"symbol": "USDT", "decimals": 18, "kind": "ERC-20"},
        "0x1af3f329e8be154074d8769d1ffa4ee058b1dbc3": {"symbol": "DAI", "decimals": 18, "kind": "ERC-20"},
        "0x8ac76a51cc950d9822d68b83fe1ad97b32cd580d": {"symbol": "USDC", "decimals": 18, "kind": "ERC-20"},
    },
    137: {
        "0x0d500b1d8e8ef31e21c99d1db9a6444d3adf1270": {"symbol": "WMATIC", "decimals": 18, "kind": "wrapped native"},
        "0x8f3cf7ad23cd3cadbd9735aff958023239c6a063": {"symbol": "DAI", "decimals": 18, "kind": "ERC-20"},
        "0x2791bca1f2de4661ed88a30c99a7a9449aa84174": {"symbol": "USDC", "decimals": 6, "kind": "ERC-20"},
        "0xc2132d05d31c914a87c6611c10748aeb04b58e8f": {"symbol": "USDT", "decimals": 6, "kind": "ERC-20"},
        "0x7ceb23fd6bc0add59e62ac25578270cff1b9f619": {"symbol": "WETH", "decimals": 18, "kind": "wrapped ETH"},
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze saved Railgun datasets offline")
    p.add_argument(
        "--input",
        default=None,
        help="Path to saved JSON/CSV dataset (default depends on --network)",
    )
    p.add_argument(
        "--network",
        choices=sorted(NETWORK_ALIASES),
        default="ethereum",
        help="Network shortcut: ethereum, bsc, polygon (default: ethereum)",
    )
    p.add_argument("--chain-id", type=int, default=None, help="Optional chain ID override")
    p.add_argument(
        "--contract-address",
        default=None,
        help="Optional Railgun contract/proxy address filter",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for figures and summary files (default: figures_saved_<chain_id>)",
    )
    p.add_argument(
        "--max-candidates",
        type=int,
        default=200,
        help="Maximum candidate Shields to show in the wide weights chart; 0/-1 means all.",
    )
    p.add_argument(
        "--amount-mode",
        choices=("raw", "normalized"),
        default="raw",
        help="Use raw blockchain integers or normalized token amounts in amount plots (default: raw)",
    )
    p.add_argument(
        "--subset-sum-max-k",
        type=int,
        default=1,
        help=(
            "Maximum number k of Shield events allowed in a joint subset-sum explanation "
            "for one Unshield. Default=1 (singleton explanations only)."
        ),
    )
    p.add_argument(
        "--entropy-weight",
        type=float,
        default=0.7,
        help="Weight of normalized entropy in final anonymity score (default 0.7; heuristic).",
    )
    p.add_argument(
        "--multiplicity-weight",
        type=float,
        default=0.3,
        help="Weight of subset-sum multiplicity bonus when k>1 (default 0.3; heuristic).",
    )
    p.add_argument(
        "--amount-sigma-fraction",
        type=float,
        default=0.05,
        help="Auto amount scale: sigma = max(|unshield amount| * fraction, 1). Default 0.05 (5%%).",
    )
    p.add_argument(
        "--time-decay-hours",
        type=float,
        default=96.0,
        help="Hours scale for Shield→Unshield time penalty (default 96).",
    )
    return p.parse_args()


def _parse_extra(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value in (None, "", "{}", "None"):
        return {}
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Expected JSON list of records")
        return [dict(row) for row in data]
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            return [dict(row) for row in csv.DictReader(f)]
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row["chain_id"] = int(row["chain_id"])
    row["log_index"] = int(row["log_index"])
    row["block_number"] = int(row["block_number"])
    row["timestamp"] = int(row["timestamp"])
    row["amount_raw"] = int(row["amount_raw"])
    row["extra"] = _parse_extra(row.get("extra"))
    return row


def _short_token(address: str) -> str:
    return f"{address[:8]}...{address[-4:]}"


def _token_meta(chain_id: int, token: str) -> dict[str, Any]:
    token_l = token.lower()
    meta = TOKEN_METADATA.get(chain_id, {}).get(token_l, {})
    symbol = meta.get("symbol", _short_token(token))
    decimals = meta.get("decimals")
    kind = meta.get("kind", "token address")
    return {
        "address": token,
        "symbol": symbol,
        "decimals": decimals,
        "kind": kind,
        "label": f"{symbol} ({token})" if symbol != _short_token(token) else token,
    }


def _display_amount(chain_id: int, token: str, amount_raw: int, amount_mode: str) -> float:
    if amount_mode == "raw":
        return float(amount_raw)
    meta = _token_meta(chain_id, token)
    decimals = meta.get("decimals")
    if decimals is None:
        return float(amount_raw)
    return float(amount_raw) / (10 ** int(decimals))


def _rows_to_records(rows: list[dict[str, Any]]) -> tuple[list[ShieldRecord], list[UnshieldRecord]]:
    shields: list[ShieldRecord] = []
    unshields: list[UnshieldRecord] = []
    for row in rows:
        pool = PoolId(row["chain_id"], str(row["contract_address"]).lower(), 0)
        kind = str(row["kind"]).lower()
        if kind == "shield":
            shields.append(
                ShieldRecord(
                    tx_hash=str(row["tx_hash"]),
                    log_index=row["log_index"],
                    block_number=row["block_number"],
                    timestamp=row["timestamp"],
                    pool=pool,
                    token=str(row["token"]),
                    amount_raw=row["amount_raw"],
                    extra=row["extra"],
                )
            )
        elif kind == "unshield":
            unshields.append(
                UnshieldRecord(
                    tx_hash=str(row["tx_hash"]),
                    log_index=row["log_index"],
                    block_number=row["block_number"],
                    timestamp=row["timestamp"],
                    pool=pool,
                    token=str(row["token"]),
                    amount_raw=row["amount_raw"],
                    extra=row["extra"],
                )
            )
    return shields, unshields


def _ts_iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _stats_dict(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.array(values, dtype=float)
    return {
        "min": float(arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def _anonymity_extra_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "entropy_component_weight": float(args.entropy_weight),
        "multiplicity_component_weight": float(args.multiplicity_weight),
        "amount_sigma_fraction": float(args.amount_sigma_fraction),
        "time_decay_hours": float(args.time_decay_hours),
    }


def _analyze_unshields(
    shields: list[ShieldRecord],
    unshields: list[UnshieldRecord],
    subset_sum_max_k: int,
    anonymity_extra: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered_unshields = sorted(unshields, key=lambda x: (x.timestamp, x.block_number, x.log_index))
    if not ordered_unshields:
        return rows

    last_key = (
        ordered_unshields[-1].tx_hash,
        ordered_unshields[-1].log_index,
        ordered_unshields[-1].block_number,
        ordered_unshields[-1].timestamp,
    )

    for u in ordered_unshields:
        feasible = feasible_shields_for_unshield(u, shields)
        score, diag = anonymity_metric(
            u,
            feasible,
            subset_sum_max_k=subset_sum_max_k,
            **anonymity_extra,
        )
        weights = [float(x) for x in diag.get("weights", [])]
        top3 = [float(x) for x in diag.get("weights_top3", [])]
        token_meta = _token_meta(u.pool.chain_id, str(u.token))
        rows.append(
            {
                "tx_hash": u.tx_hash,
                "log_index": u.log_index,
                "block_number": u.block_number,
                "timestamp": u.timestamp,
                "timestamp_iso": _ts_iso(u.timestamp),
                "chain_id": u.pool.chain_id,
                "contract_address": u.pool.contract,
                "token": u.token,
                "token_label": token_meta["label"],
                "amount_raw": u.amount_raw,
                "n_feasible": int(diag.get("n_feasible", 0)),
                "anonymity_score": float(score),
                "entropy_bits": float(diag.get("entropy_bits", 0.0)),
                "entropy_norm": float(diag.get("entropy_norm", 0.0)),
                "subset_sum_explanations": int(diag.get("subset_sum_explanations", 0)),
                "max_weight": float(max(weights)) if weights else 0.0,
                "weight_top1": top3[0] if len(top3) > 0 else 0.0,
                "weight_top2": top3[1] if len(top3) > 1 else 0.0,
                "weight_top3": top3[2] if len(top3) > 2 else 0.0,
                "is_selected_example": (
                    u.tx_hash,
                    u.log_index,
                    u.block_number,
                    u.timestamp,
                )
                == last_key,
                "amount_sigma_used": float(diag.get("amount_sigma_used", 0.0)),
                "amount_sigma_fraction": float(diag.get("amount_sigma_fraction", 0.02)),
                "time_decay_hours_used": float(diag.get("time_decay_hours_used", 72.0)),
                "entropy_weight_used": float(diag.get("entropy_component_weight", 0.7)),
                "multiplicity_weight_used": float(diag.get("multiplicity_component_weight", 0.3)),
            }
        )
    return rows


def _build_summary(
    rows: list[dict[str, Any]],
    shields: list[ShieldRecord],
    unshields: list[UnshieldRecord],
    unshield_anonymity_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    contract_counts = Counter(str(row["contract_address"]).lower() for row in rows)
    token_counts = Counter(str(row["token"]).lower() for row in rows)
    token_labels = Counter(_token_meta(row["chain_id"], str(row["token"]))["label"] for row in rows)

    feasible_sizes = [
        len(feasible_shields_for_unshield(u, shields))
        for u in unshields
    ]
    feasible_stats = {}
    if feasible_sizes:
        arr = np.array(feasible_sizes, dtype=float)
        feasible_stats = {
            "min": int(arr.min()),
            "median": float(np.median(arr)),
            "mean": float(arr.mean()),
            "max": int(arr.max()),
        }

    scored_rows = [row for row in unshield_anonymity_rows if row["n_feasible"] > 0]
    empty_feasible_rows = [row for row in unshield_anonymity_rows if row["n_feasible"] == 0]
    score_stats = _stats_dict([row["anonymity_score"] for row in scored_rows])
    max_weight_stats = _stats_dict([row["max_weight"] for row in scored_rows])
    selected_example = next((row for row in unshield_anonymity_rows if row["is_selected_example"]), None)

    summary = {
        "total_rows": len(rows),
        "shield_rows": len(shields),
        "unshield_rows": len(unshields),
        "block_range": {
            "from": min(row["block_number"] for row in rows),
            "to": max(row["block_number"] for row in rows),
        },
        "timestamp_range": {
            "from": min(row["timestamp"] for row in rows),
            "to": max(row["timestamp"] for row in rows),
            "from_iso": _ts_iso(min(row["timestamp"] for row in rows)),
            "to_iso": _ts_iso(max(row["timestamp"] for row in rows)),
        },
        "unique_contract_addresses": len(contract_counts),
        "contract_address_counts_top": contract_counts.most_common(10),
        "unique_tokens": len(token_counts),
        "token_counts_top": token_counts.most_common(10),
        "token_labels_top": token_labels.most_common(10),
        "feasible_candidate_sizes": feasible_stats,
        "anonymity_scoring": {
            "total_unshields": len(unshield_anonymity_rows),
            "scored_unshields": len(scored_rows),
            "empty_feasible_set_unshields": len(empty_feasible_rows),
            "score_stats": score_stats,
            "max_weight_stats": max_weight_stats,
        },
        "selected_example_unshield": (
            {
                "tx_hash": selected_example["tx_hash"],
                "log_index": selected_example["log_index"],
                "block_number": selected_example["block_number"],
                "timestamp": selected_example["timestamp"],
                "timestamp_iso": selected_example["timestamp_iso"],
                "token": selected_example["token"],
                "token_label": selected_example["token_label"],
                "n_feasible": selected_example["n_feasible"],
                "anonymity_score": selected_example["anonymity_score"],
                "max_weight": selected_example["max_weight"],
            }
            if selected_example
            else None
        ),
    }
    return summary


def _write_summary(out_dir: Path, summary: dict[str, Any]) -> None:
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        f"total_rows: {summary['total_rows']}",
        f"shield_rows: {summary['shield_rows']}",
        f"unshield_rows: {summary['unshield_rows']}",
        f"block_range: {summary['block_range']['from']} -> {summary['block_range']['to']}",
        f"timestamp_range: {summary['timestamp_range']['from_iso']} -> {summary['timestamp_range']['to_iso']}",
        f"unique_contract_addresses: {summary['unique_contract_addresses']}",
        f"unique_tokens: {summary['unique_tokens']}",
    ]
    if summary["feasible_candidate_sizes"]:
        fstats = summary["feasible_candidate_sizes"]
        lines.append(
            "feasible_candidates_per_unshield:"
            f" min={fstats['min']}, median={fstats['median']:.1f},"
            f" mean={fstats['mean']:.2f}, max={fstats['max']}"
        )
    ascoring = summary.get("anonymity_scoring", {})
    if ascoring:
        lines.append(
            "anonymity_scoring:"
            f" scored={ascoring.get('scored_unshields', 0)}"
            f", empty_feasible={ascoring.get('empty_feasible_set_unshields', 0)}"
        )
        score_stats = ascoring.get("score_stats", {})
        if score_stats:
            lines.append(
                "anonymity_score_per_unshield:"
                f" min={score_stats['min']:.3f}, p10={score_stats['p10']:.3f},"
                f" median={score_stats['median']:.3f}, mean={score_stats['mean']:.3f},"
                f" p90={score_stats['p90']:.3f}, max={score_stats['max']:.3f}"
            )
        max_weight_stats = ascoring.get("max_weight_stats", {})
        if max_weight_stats:
            lines.append(
                "max_candidate_weight_per_unshield:"
                f" min={max_weight_stats['min']:.3f}, p10={max_weight_stats['p10']:.3f},"
                f" median={max_weight_stats['median']:.3f}, mean={max_weight_stats['mean']:.3f},"
                f" p90={max_weight_stats['p90']:.3f}, max={max_weight_stats['max']:.3f}"
            )
    selected_example = summary.get("selected_example_unshield")
    if selected_example:
        lines.append(
            "selected_example_unshield:"
            f" token={selected_example['token_label']},"
            f" n={selected_example['n_feasible']},"
            f" score={selected_example['anonymity_score']:.3f},"
            f" max_weight={selected_example['max_weight']:.3f}"
        )
    lines.append("")
    lines.append("top_tokens:")
    for token, count in summary["token_labels_top"]:
        lines.append(f"  {token}: {count}")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_token_reference(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    by_token: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        chain_id = int(row["chain_id"])
        token = str(row["token"])
        key = (chain_id, token.lower())
        if key in by_token:
            continue
        meta = _token_meta(chain_id, token)
        by_token[key] = {
            "chain_id": chain_id,
            "token_address": token,
            "symbol": meta["symbol"],
            "decimals": "" if meta["decimals"] is None else meta["decimals"],
            "type": meta["kind"],
        }

    with (out_dir / "token_reference.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["chain_id", "token_address", "symbol", "decimals", "type"],
        )
        writer.writeheader()
        writer.writerows(by_token.values())


def _write_unshield_anonymity(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "tx_hash",
        "log_index",
        "block_number",
        "timestamp",
        "timestamp_iso",
        "chain_id",
        "contract_address",
        "token",
        "token_label",
        "amount_raw",
        "n_feasible",
        "anonymity_score",
        "entropy_bits",
        "entropy_norm",
        "subset_sum_explanations",
        "max_weight",
        "weight_top1",
        "weight_top2",
        "weight_top3",
        "is_selected_example",
        "amount_sigma_used",
        "amount_sigma_fraction",
        "time_decay_hours_used",
        "entropy_weight_used",
        "multiplicity_weight_used",
    ]
    with (out_dir / "unshield_anonymity.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_common_figures(
    out_dir: Path,
    chain_label: str,
    df: pd.DataFrame,
    shields: list[ShieldRecord],
    unshields: list[UnshieldRecord],
    unshield_anonymity_rows: list[dict[str, Any]],
    max_candidates: int,
    amount_mode: str,
    subset_sum_max_k: int,
    anonymity_extra: dict[str, Any],
) -> None:
    df = df.copy()
    df["token_label"] = [
        _token_meta(int(row.chain_id), str(row.token))["symbol"]
        for row in df.itertuples(index=False)
    ]
    df["amount_display"] = [
        _display_amount(int(row.chain_id), str(row.token), int(row.amount_raw), amount_mode)
        for row in df.itertuples(index=False)
    ]
    anonym_df = pd.DataFrame(unshield_anonymity_rows)

    # 1) Amount distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    for kind, color in (("shield", "#3498db"), ("unshield", "#e74c3c")):
        sub = df[df["kind"] == kind]["amount_display"].astype(float)
        sub = sub[sub > 0]
        if len(sub) == 0:
            continue
        ax.hist(np.log10(sub), bins=40, alpha=0.55, label=kind, color=color, edgecolor="white")
    if amount_mode == "normalized":
        ax.set_xlabel("log10(amount) in token units")
        ax.set_title(f"Amount distribution (normalized amounts, {chain_label})")
    else:
        ax.set_xlabel("log10(amount_raw)")
        ax.set_title(f"Amount distribution (raw values, {chain_label})")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "saved_amounts.png", dpi=150)
    plt.close(fig)

    # 2) Timeline
    fig, ax = plt.subplots(figsize=(10, 3))
    t0 = df["timestamp"].min()
    for kind, y, color in (("shield", 0, "#3498db"), ("unshield", 1, "#e74c3c")):
        sub = df[df["kind"] == kind]
        if sub.empty:
            continue
        ax.scatter(sub["timestamp"] - t0, [y] * len(sub), s=8, alpha=0.5, c=color, label=kind)
    ax.set_yticks([0, 1], ["Shield", "Unshield"])
    ax.set_xlabel("Seconds since first event in sample")
    ax.set_title("Railgun events from saved dataset")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "saved_timeline.png", dpi=150)
    plt.close(fig)

    # 3) Inter-event gaps
    gaps = inter_event_times_seconds(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    if len(gaps) > 0:
        series = gaps.dropna()
        clip_hi = np.percentile(series, 99) if len(series) > 5 else series.max()
        ax.hist(np.clip(series, 0, clip_hi), bins=50, color="#34495e", edgecolor="white")
    ax.set_xlabel("Inter-event gap (seconds)")
    ax.set_ylabel("Frequency")
    ax.set_title("Inter-event gaps (top 1% clipped)")
    fig.tight_layout()
    fig.savefig(out_dir / "saved_inter_event.png", dpi=150)
    plt.close(fig)

    # 4) Top tokens
    token_counts = (
        df.groupby(["token_label", "kind"]).size().unstack(fill_value=0).sort_values(
            by=["shield", "unshield"], ascending=False
        )
    )
    top_tokens = token_counts.head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(top_tokens))
    ax.bar(x - 0.2, top_tokens.get("shield", 0), width=0.4, label="shield", color="#3498db")
    ax.bar(x + 0.2, top_tokens.get("unshield", 0), width=0.4, label="unshield", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(list(top_tokens.index), rotation=30, ha="right")
    ax.set_ylabel("Event count")
    ax.set_title("Top tokens by event count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "saved_top_tokens.png", dpi=150)
    plt.close(fig)

    # 5) Cumulative activity
    df_sorted = df.sort_values("timestamp").copy()
    df_sorted["t_rel_hours"] = (df_sorted["timestamp"] - df_sorted["timestamp"].min()) / 3600.0
    fig, ax = plt.subplots(figsize=(10, 4))
    for kind, color in (("shield", "#3498db"), ("unshield", "#e74c3c")):
        sub = df_sorted[df_sorted["kind"] == kind].copy()
        if sub.empty:
            continue
        sub["cum_count"] = np.arange(1, len(sub) + 1)
        ax.plot(sub["t_rel_hours"], sub["cum_count"], label=kind, color=color, linewidth=2)
    ax.set_xlabel("Hours since first event in sample")
    ax.set_ylabel("Cumulative count")
    ax.set_title("Cumulative activity over time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "saved_cumulative_activity.png", dpi=150)
    plt.close(fig)

    # 6) Feasible set sizes across all unshields
    feasible_sizes = [len(feasible_shields_for_unshield(u, shields)) for u in unshields]
    if feasible_sizes:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(feasible_sizes, bins=min(40, max(10, len(set(feasible_sizes)))), color="#16a085", edgecolor="white")
        ax.set_xlabel("Number of feasible predecessor Shields")
        ax.set_ylabel("Unshield count")
        ax.set_title("Feasible predecessor set sizes")
        fig.tight_layout()
        fig.savefig(out_dir / "saved_feasible_sizes.png", dpi=150)
        plt.close(fig)

    # 7) Network-level anonymity metrics across all unshields
    if not anonym_df.empty:
        scored = anonym_df[anonym_df["n_feasible"] > 0].copy()
        if not scored.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(scored["anonymity_score"], bins=30, color="#5dade2", edgecolor="white")
            ax.set_xlabel("Anonymity score")
            ax.set_ylabel("Unshield count")
            ax.set_title(f"Anonymity score distribution across all scored Unshields ({chain_label})")
            fig.tight_layout()
            fig.savefig(out_dir / "saved_anonymity_hist.png", dpi=150)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.scatter(
                scored["n_feasible"],
                scored["anonymity_score"],
                s=18,
                alpha=0.55,
                color="#8e44ad",
            )
            ax.set_xlabel("Feasible predecessor set size")
            ax.set_ylabel("Anonymity score")
            ax.set_title("Anonymity score vs feasible set size")
            fig.tight_layout()
            fig.savefig(out_dir / "saved_score_vs_feasible.png", dpi=150)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(scored["max_weight"], bins=30, color="#f39c12", edgecolor="white")
            ax.set_xlabel("Maximum candidate weight")
            ax.set_ylabel("Unshield count")
            ax.set_title("Maximum candidate weight across all scored Unshields")
            fig.tight_layout()
            fig.savefig(out_dir / "saved_max_weight_hist.png", dpi=150)
            plt.close(fig)

    # 8) Candidate weights for the last unshield (illustrative example)
    if not unshields:
        return
    u = max(unshields, key=lambda x: (x.timestamp, x.block_number, x.log_index))
    full_candidates = feasible_shields_for_unshield(u, shields)
    if not full_candidates:
        return

    score, diag = anonymity_metric(
        u,
        full_candidates,
        subset_sum_max_k=subset_sum_max_k,
        **anonymity_extra,
    )
    full_weights = diag.get("weights", [])
    display_all = max_candidates <= 0
    if len(full_weights) != len(full_candidates):
        weights = full_weights
        note = ""
    elif (not display_all) and len(full_candidates) > max_candidates:
        weights = full_weights[-max_candidates:]
        note = f" (showing last {max_candidates} of {len(full_candidates)} candidates in time order)"
    else:
        weights = full_weights
        note = ""
    if not weights:
        return

    fig, ax = plt.subplots(figsize=(min(22, 0.06 * len(weights) + 6), 4))
    x = np.arange(len(weights))
    ax.bar(x, weights, color="#9b59b6")
    ax.set_ylabel("Weight")
    ax.set_xlabel("Candidate Shield index")
    ax.set_ylim(0, max(0.05, float(max(weights)) * 1.15))
    ax.set_title(
        f"Candidate model for selected example Unshield{note}\n"
        f"n≈{len(full_candidates)}, anonymity score ≈ {score:.2f}"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "saved_weights.png", dpi=150)
    plt.close(fig)

    top_k = min(20, len(weights))
    ranked = sorted(weights, reverse=True)[:top_k]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(top_k), ranked, color="#8e44ad")
    ax.set_xlabel("Top-ranked candidate")
    ax.set_ylabel("Weight")
    ax.set_title(
        f"Top {top_k} candidate weights for selected example Unshield\n"
        f"full feasible set n≈{len(full_candidates)}, anonymity score ≈ {score:.2f}"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "saved_weights_top20.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    network_chain_id, default_input = NETWORK_ALIASES[args.network]
    chain_id = args.chain_id if args.chain_id is not None else network_chain_id
    subset_sum_max_k = max(int(args.subset_sum_max_k), 1)

    input_path = Path(args.input or default_input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    rows = [_normalize_row(row) for row in _load_rows(input_path)]

    rows = [row for row in rows if row["chain_id"] == chain_id]
    if args.contract_address:
        want = args.contract_address.lower()
        rows = [row for row in rows if str(row["contract_address"]).lower() == want]

    if not rows:
        raise ValueError("No rows left after filtering. Check --input / --chain-id / --contract-address.")

    shields, unshields = _rows_to_records(rows)
    df = records_to_frame(shields, unshields)
    chain_label = CHAIN_LABELS.get(chain_id, f"chain {chain_id}")

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / f"figures_saved_{chain_id}"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    anonymity_extra = _anonymity_extra_from_args(args)
    unshield_anonymity_rows = _analyze_unshields(
        shields,
        unshields,
        subset_sum_max_k=subset_sum_max_k,
        anonymity_extra=anonymity_extra,
    )
    summary = _build_summary(rows, shields, unshields, unshield_anonymity_rows)
    _write_summary(out_dir, summary)
    _write_token_reference(out_dir, rows)
    _write_unshield_anonymity(out_dir, unshield_anonymity_rows)
    _plot_common_figures(
        out_dir,
        chain_label,
        df,
        shields,
        unshields,
        unshield_anonymity_rows,
        args.max_candidates,
        args.amount_mode,
        subset_sum_max_k=subset_sum_max_k,
        anonymity_extra=anonymity_extra,
    )

    print(f"[network] {args.network} (chain_id={chain_id})")
    print(f"[input] {input_path}")
    print(f"[rows] total={summary['total_rows']} shield={summary['shield_rows']} unshield={summary['unshield_rows']}")
    print(
        f"[blocks] {summary['block_range']['from']} -> {summary['block_range']['to']}"
    )
    print(
        f"[time] {summary['timestamp_range']['from_iso']} -> {summary['timestamp_range']['to_iso']}"
    )
    print(f"[tokens] unique={summary['unique_tokens']}")
    print(f"[amount_mode] {args.amount_mode}")
    print(f"[subset_sum_max_k] {subset_sum_max_k}")
    print(
        "[metric]"
        f" entropy_w={anonymity_extra['entropy_component_weight']:.3g}"
        f" multiplicity_w={anonymity_extra['multiplicity_component_weight']:.3g}"
        f" amount_sigma_fraction={anonymity_extra['amount_sigma_fraction']:.4g}"
        f" time_decay_h={anonymity_extra['time_decay_hours']:.3g}"
    )
    if summary["feasible_candidate_sizes"]:
        fstats = summary["feasible_candidate_sizes"]
        print(
            "[feasible]"
            f" min={fstats['min']} median={fstats['median']:.1f}"
            f" mean={fstats['mean']:.2f} max={fstats['max']}"
        )
    ascoring = summary.get("anonymity_scoring", {})
    score_stats = ascoring.get("score_stats", {})
    if score_stats:
        print(
            "[anonymity]"
            f" scored={ascoring.get('scored_unshields', 0)}"
            f" empty={ascoring.get('empty_feasible_set_unshields', 0)}"
            f" median={score_stats['median']:.3f}"
            f" mean={score_stats['mean']:.3f}"
            f" p90={score_stats['p90']:.3f}"
        )
    print(f"[write] summary, token reference, per-unshield anonymity, and figures saved to {out_dir}")


if __name__ == "__main__":
    main()
