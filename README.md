# Airline Pricing + Overbooking Simulation

This repo is a small “toy market” simulator for airline pricing and (simple) overbooking decisions. The core idea is:

- Consumers arrive over a booking horizon and decide whether/when to buy.
- Airlines post a day-by-day price path (cheap early → expensive late).
- Airlines can overbook, which increases revenue but risks bumps (with a penalty cost).
- After each episode, airlines do a lightweight hill-climb update to their parameters using realized profit.

There are two ways to run it:

1. **Synthetic experiments** (no external data needed)
2. **Data-driven experiments** (uses processed on-time + fare samples that live under `data/processed/`)

## Repo layout

- `simulation/airline_sim.py`: main simulation logic (consumers, pricing, choice model, overbooking/bumping, stats).
- `simulation/run_experiments.py`: synthetic monopoly/duopoly/triopoly runs; writes outputs to `simulation/results/`.
- `simulation/run_data_experiments.py`: same idea, but initializes airlines from real carrier fare/quality profiles.
- `data_prep/prepare_airline_profiles.py`: creates `data/processed/airline_quality_2026_01.json` from the BTS on-time CSV.
- `data_prep/prepare_price_samples.py`: creates `data/processed/db1b_fare_samples_public_21fields.json.gz` from extracted DB1B `.asc` files.

Note: `data/` and `simulation/results/` are gitignored since they’re large/generated.

## Setup

I’m using Python 3 (no special packages needed for the synthetic runs).

Optional (only if you want to run the data prep scripts): you’ll need `pandas`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install pandas
```

## Quick start (synthetic experiments)

Run from inside the `simulation/` folder (this keeps output paths simple):

```bash
cd simulation
python3 run_experiments.py
```

Outputs:

- `simulation/results/summary.txt`
- `simulation/results/summary.json`

## Data-driven experiments (uses `data/processed/`)

This script expects these files:

- `data/processed/airline_quality_2026_01.json`
- `data/processed/db1b_fare_samples_public_21fields.json.gz`

Because the simulation code is a single module (not a packaged install), the easiest way to run from the repo root is to set `PYTHONPATH` to `simulation/`:

```bash
PYTHONPATH=simulation python3 simulation/run_data_experiments.py
```

Outputs:

- `simulation/results/data_driven_summary.txt`
- `simulation/results/data_driven_summary.json`

## (Optional) Rebuilding `data/processed/`

### 1) Airline quality profile (on-time performance)

Requires the BTS on-time CSV at:

- `data/On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2026_1.csv`

Run:

```bash
python3 data_prep/prepare_airline_profiles.py
```

### 2) Fare samples (DB1B extracted `.asc` files)

This expects a directory of extracted DB1B `.asc` files under:

- `data/prices_extracted/`

Run:

```bash
python3 data_prep/prepare_price_samples.py
```

That script does reservoir sampling to keep at most 20,000 fares per carrier, and writes:

- `data/processed/db1b_fare_samples_public_21fields.json.gz`

## Notes / knobs I used

- Main configuration lives in `simulation/airline_sim.py` under `SimConfig` (capacity, booking horizon, consumer valuations, price/quality sensitivity, bump cost, etc.).
- “Learning” is intentionally simple: each airline mutates its parameters each episode and keeps changes that improve profit (with a small chance to accept worse moves).
