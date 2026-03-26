# Railgun Correspondence Analysis

This project analyzes possible correspondences between `Shield` and `Unshield` events in Railgun using public on-chain data.

## What this repository contains

- data loading and normalization for Railgun event windows
- feasible-set construction under public constraints (time ordering, token match)
- candidate ranking with amount and time signals
- anonymity scoring with entropy and subset-sum multiplicity components
- scripts for offline analysis and figure generation across Ethereum, BSC, and Polygon

## Quick run (offline analysis)

```bash
python scripts/analyze_saved_dataset.py --network ethereum
```

You can override scoring parameters, for example:

```bash
python scripts/analyze_saved_dataset.py \
  --network ethereum \
  --subset-sum-max-k 3 \
  --amount-sigma-fraction 0.05 \
  --time-decay-hours 96
```
