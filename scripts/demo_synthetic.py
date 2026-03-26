"""Synthetic Shield/Unshield demo: anonymity metric and summary stats."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from railgun_analysis.anonymity import anonymity_metric
from railgun_analysis.models import PoolId, ShieldRecord, UnshieldRecord
from railgun_analysis.statistics import inter_event_times_seconds, records_to_frame


def main() -> None:
    pool = PoolId(1, "0xabc0000000000000000000000000000000000001", 0)
    token = "0xdac17f958d2ee523a2206206994597c13d831ec7"

    shields = [
        ShieldRecord(
            "0x01",
            0,
            100,
            1_700_000_000,
            pool,
            token,
            1_000_000_000_000_000,
        ),
        ShieldRecord(
            "0x02",
            0,
            101,
            1_700_000_100,
            pool,
            token,
            1_000_000_000_000_000,
        ),
        ShieldRecord(
            "0x03",
            0,
            102,
            1_700_000_200,
            pool,
            token,
            999_000_000_000_000,
        ),
    ]
    unshield = UnshieldRecord(
        "0x10",
        0,
        200,
        1_700_010_000,
        pool,
        token,
        1_000_000_000_000_000,
    )

    score, diag = anonymity_metric(
        unshield,
        shields,
        amount_sigma=5e14,
        time_decay_hours=48.0,
    )
    print("anonymity_metric:", round(score, 4), diag)

    df = records_to_frame(shields, [unshield])
    print("inter_event_times (sample):\n", inter_event_times_seconds(df).describe())


if __name__ == "__main__":
    main()
