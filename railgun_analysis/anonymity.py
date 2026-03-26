from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable

import numpy as np

from railgun_analysis.models import ShieldRecord, UnshieldRecord
from railgun_analysis.taint import feasible_shields_for_unshield


def _resolve_amount_sigma(
    unshield: UnshieldRecord,
    amount_sigma: float | None,
    *,
    sigma_fraction: float = 0.05,
) -> float:
    """
    If amount_sigma is None, use a scale relative to the Unshield amount so that
    typical differences in amount_raw (same token) affect logits. The previous
    default of 1e12 in effect made the softmax depend almost only on time.

    sigma_fraction: when amount_sigma is None, sigma = max(|amount_raw(u)| * sigma_fraction, 1).
    """
    if amount_sigma is not None and amount_sigma > 0:
        return float(amount_sigma)
    target = abs(float(unshield.amount_raw))
    frac = float(sigma_fraction)
    if frac <= 0:
        frac = 0.05
    return max(target * frac, 1.0)


def _softmax_weights(
    shields: list[ShieldRecord],
    unshield: UnshieldRecord,
    *,
    amount_sigma: float,
    time_decay_hours: float,
) -> np.ndarray:
    """Higher weight when amount matches and shield is not too old (plain softmax on logits)."""
    if not shields:
        return np.array([])
    amounts = np.array([float(s.amount_raw) for s in shields], dtype=np.float64)
    target = float(unshield.amount_raw)
    dt_hours = np.array(
        [
            max(0.0, (unshield.timestamp - s.timestamp) / 3600.0)
            for s in shields
        ],
        dtype=np.float64,
    )
    # Gaussian-ish amount score; exponential decay on waiting time
    amount_score = -0.5 * ((amounts - target) / max(amount_sigma, 1e-9)) ** 2
    time_score = -dt_hours / max(time_decay_hours, 1e-9)
    logits = amount_score + time_score
    logits -= np.max(logits)
    w = np.exp(logits)
    s = w.sum()
    if s <= 0:
        return np.ones(len(shields)) / len(shields)
    return w / s


def count_subset_sum_explanations(
    shields: list[ShieldRecord],
    target_amount: int,
    *,
    max_k: int = 5,
    cap: int | None = None,
) -> int:
    """
    Number of distinct non-empty subsets of `shields` whose amounts sum to
    `target_amount` (bounded subset size for tractability).

    Use when you hypothesize multiple Shields jointly funding one Unshield.
    Set max_k small; exact full subset-sum is exponential.
    """
    n = len(shields)
    if n == 0 or target_amount <= 0:
        return 0
    amounts = [s.amount_raw for s in shields]

    # Fast paths for k=2/3 using multiset combinatorics.
    # This matches the subset-by-indices semantics of the DFS implementation.
    if max_k in (2, 3):
        cnt = Counter(amounts)

        singles = cnt.get(target_amount, 0)
        total = singles

        # Helper: count unordered pairs i<j with amount_i + amount_j = target.
        def pair_count() -> int:
            pairs = 0
            for a, ca in cnt.items():
                b = target_amount - a
                cb = cnt.get(b, 0)
                if cb <= 0:
                    continue
                if a < b:
                    pairs += ca * cb
                elif a == b:
                    pairs += ca * (ca - 1) // 2
            return pairs

        if max_k >= 2:
            pairs = pair_count()
            total += pairs
            if cap is not None and total >= cap:
                return int(cap)

        if max_k == 3:
            # Count triples i<j<k with amounts summing to target, i.e. a<=b<=c and a+b+c=target.
            keys = sorted(cnt.keys())
            triples = 0
            for i, a in enumerate(keys):
                ca = cnt[a]
                for j in range(i, len(keys)):
                    b = keys[j]
                    cb = cnt[b]
                    c = target_amount - a - b
                    if c < b:
                        break
                    cc = cnt.get(c, 0)
                    if cc <= 0:
                        continue
                    if a == b == c:
                        triples += ca * (ca - 1) * (ca - 2) // 6
                    elif a == b != c:
                        triples += ca * (ca - 1) // 2 * cc
                    elif a != b == c:
                        triples += ca * (cb * (cb - 1) // 2)
                    else:
                        triples += ca * cb * cc
                    if cap is not None and singles + pairs + triples >= cap:
                        return int(cap)
            total += triples

        # Must cap the *true* count, not return cap when total is still below it (was a bug).
        return int(min(total, cap) if cap is not None else total)

    def dfs(i: int, remaining: int, used: int) -> int:
        if i == n:
            return 1 if (remaining == 0 and 0 < used <= max_k) else 0
        if remaining < 0 or used > max_k:
            return 0
        skip = dfs(i + 1, remaining, used)
        take = dfs(i + 1, remaining - amounts[i], used + 1) if used < max_k else 0
        return skip + take

    res = dfs(0, target_amount, 0)
    if cap is not None:
        return int(min(res, cap))
    return res


def feasible_shield_candidates(
    unshield: UnshieldRecord,
    shields: Iterable[ShieldRecord],
    **kwargs,
) -> list[ShieldRecord]:
    return feasible_shields_for_unshield(unshield, shields, **kwargs)


def anonymity_metric(
    unshield: UnshieldRecord,
    shields: Iterable[ShieldRecord],
    *,
    amount_sigma: float | None = None,
    amount_sigma_fraction: float = 0.05,
    time_decay_hours: float = 96.0,
    subset_sum_max_k: int = 1,
    reference_entropy_bits: float | None = None,
    entropy_component_weight: float = 0.7,
    multiplicity_component_weight: float = 0.3,
) -> tuple[float, dict]:
    """
    Return (score, diagnostics) with score in [0, 1].

    - Uses entropy of weights over feasible singleton explanations by default
      (subset_sum_max_k=1). If k>1, multiplicity from subset-sum count enters
      via log2(1 + explanations) bonus.
    - Normalizes entropy by log2(N) when N>1; if reference_entropy_bits is set,
      divides by that instead (cross-protocol comparison).

    amount_sigma: if None, uses max(|unshield.amount_raw| * amount_sigma_fraction, 1) so that
    amount similarity contributes meaningfully (avoids the legacy default 1e12
    that collapsed to time-only weighting). Default fraction is 0.05 (2%).

    entropy_component_weight / multiplicity_component_weight: convex mix of
    normalized entropy vs. combinatorial bonus (defaults 0.7/0.3 are heuristics,
    not derived from theory; normalize to sum 1).

    score near 0: dominated probability mass on one candidate (traceable under model).
    score near 1: many candidates with flatter weights.
    """
    cand = feasible_shields_for_unshield(unshield, shields)
    n = len(cand)
    diag: dict = {"n_feasible": n, "subset_sum_explanations": 0}

    if n == 0:
        return 0.0, diag

    sigma_used = _resolve_amount_sigma(
        unshield,
        amount_sigma,
        sigma_fraction=amount_sigma_fraction,
    )
    diag["amount_sigma_used"] = sigma_used
    diag["amount_sigma_fraction"] = float(amount_sigma_fraction)
    diag["time_decay_hours_used"] = float(time_decay_hours)
    ew = max(0.0, float(entropy_component_weight))
    mw = max(0.0, float(multiplicity_component_weight))
    sm = ew + mw
    if sm <= 0:
        ew, mw, sm = 0.7, 0.3, 1.0
    ew /= sm
    mw /= sm
    diag["entropy_component_weight"] = ew
    diag["multiplicity_component_weight"] = mw

    if subset_sum_max_k > 1:
        diag["subset_sum_explanations"] = count_subset_sum_explanations(
            cand,
            unshield.amount_raw,
            max_k=subset_sum_max_k,
            # Cap the count since multiplicity_bonus saturates at ~1 when explanations are O(n).
            cap=max(n, 1),
        )

    w = _softmax_weights(
        cand,
        unshield,
        amount_sigma=sigma_used,
        time_decay_hours=time_decay_hours,
    )
    p = np.clip(w, 1e-30, 1.0)
    h_bits = float(-np.sum(p * np.log2(p)))

    if n == 1:
        entropy_norm = 0.0
    else:
        denom = reference_entropy_bits if reference_entropy_bits is not None else math.log2(n)
        entropy_norm = h_bits / denom if denom > 0 else 0.0

    multiplicity_bonus = 0.0
    if subset_sum_max_k > 1 and diag["subset_sum_explanations"] > 0:
        multiplicity_bonus = min(
            1.0,
            math.log2(1 + diag["subset_sum_explanations"]) / max(math.log2(n), 1e-9),
        )

    diag["multiplicity_bonus"] = float(multiplicity_bonus)
    score = float(np.clip(ew * entropy_norm + mw * multiplicity_bonus, 0.0, 1.0))
    diag["entropy_bits"] = h_bits
    diag["entropy_norm"] = entropy_norm
    diag["weights"] = [float(x) for x in p]
    diag["weights_top3"] = [float(x) for x in np.sort(p)[-3:][::-1]]

    return score, diag
