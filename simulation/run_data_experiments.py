from __future__ import annotations

import gzip
import json
import random
import statistics
from dataclasses import asdict
from pathlib import Path

from airline_sim import (
    Airline,
    AirlineDataProfile,
    AirlineParams,
    AirlineQuality,
    SimConfig,
    run_episode,
    summarize_stats,
)


def load_quality(path: Path) -> dict[str, AirlineQuality]:
    d = json.loads(path.read_text())
    out: dict[str, AirlineQuality] = {}
    for a in d["airlines"]:
        q = AirlineQuality(
            carrier=a["carrier"],
            flights=int(a["flights"]),
            cancel_rate=float(a["cancel_rate"]),
            ontime_rate=float(a["ontime_rate"]),
            avg_arr_delay_min=float(a["avg_arr_delay_min"]),
        )
        out[q.carrier] = q
    return out


def load_fare_samples(path: Path) -> dict[str, list[float]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        d = json.load(f)
    return {k: list(map(float, v)) for k, v in d["carriers"].items()}


def _init_airlines_from_data(
    carriers: list[str],
    quality: dict[str, AirlineQuality],
    fare_samples: dict[str, list[float]],
    rng: random.Random,
) -> list[Airline]:
    airlines: list[Airline] = []
    for c in carriers:
        samples = fare_samples[c]
        base = statistics.median(samples)
        params = AirlineParams(
            # Anchor the near-departure price to a multiple of empirical median fare.
            base_price=base * rng.uniform(0.95, 1.15),
            slope=rng.uniform(0.4, 1.2),
            overbook_rate=rng.uniform(0.00, 0.10),
        )
        profile = AirlineDataProfile(carrier=c, fare_samples=samples, quality=quality[c])
        airlines.append(Airline(name=c, params=params, rng=rng, data_profile=profile))
    return airlines


def run_market(cfg: SimConfig, carriers: list[str], episodes: int, seed: int, label: str):
    rng = random.Random(seed)
    quality = load_quality(Path("data/processed/airline_quality_2026_01.json"))
    fare_samples = load_fare_samples(Path("data/processed/db1b_fare_samples_public_21fields.json.gz"))

    missing = [c for c in carriers if c not in quality or c not in fare_samples]
    if missing:
        raise SystemExit(f"Missing carriers in processed data: {missing}")

    airlines = _init_airlines_from_data(carriers, quality, fare_samples, rng)

    all_stats = []
    for ep in range(episodes):
        stats = run_episode(cfg, airlines, rng, ep, label)
        all_stats.append(stats)
        # Same simple learning signal for draft: per-airline equal share
        for a in airlines:
            a.learn(stats.total_profit / max(1, len(airlines)))
    return all_stats


def main() -> int:
    cfg = SimConfig(
        # Adjust consumer valuations to be closer to empirical fares.
        valuation_mean=320.0,
        valuation_sd=140.0,
        price_sensitivity=0.018,
        quality_sensitivity=2.0,
    )

    out_dir = Path("simulation/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick real carrier sets (intersection of on-time carriers and DB1B sample carriers).
    experiments = [
        ("monopoly_WN", ["WN"], 120, 101),
        ("duopoly_WN_DL", ["WN", "DL"], 120, 102),
        ("triopoly_WN_DL_UA", ["WN", "DL", "UA"], 120, 103),
    ]

    payload = {"config": asdict(cfg), "experiments": {}}
    lines = []
    for label, carriers, episodes, seed in experiments:
        stats = run_market(cfg, carriers, episodes, seed, label)
        burn_in = 40
        summary = summarize_stats(stats[burn_in:])
        payload["experiments"][label] = {
            "carriers": carriers,
            "episodes": episodes,
            "burn_in": burn_in,
            "summary": summary,
        }
        lines.append(
            f"{label:18s} carriers={','.join(carriers):10s} "
            f"avg_price_paid=${summary['avg_price_paid']:.2f} "
            f"avg_profit=${summary['avg_profit']:.2f} "
            f"avg_bump_rate={100*summary['avg_bump_rate']:.2f}% "
            f"avg_quality={summary['avg_quality_score']:.3f} "
            f"avg_tickets_sold={summary['avg_tickets_sold']:.1f}"
        )

    (out_dir / "data_driven_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    (out_dir / "data_driven_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
