from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

from airline_sim import Airline, AirlineParams, SimConfig, run_episode, summarize_stats


def _init_airlines(n: int, rng: random.Random) -> list[Airline]:
    airlines: list[Airline] = []
    for i in range(n):
        params = AirlineParams(
            base_price=rng.uniform(260.0, 380.0),
            slope=rng.uniform(0.4, 1.3),
            overbook_rate=rng.uniform(0.00, 0.12),
        )
        airlines.append(Airline(name=f"A{i+1}", params=params, rng=rng))
    return airlines


def run_market(cfg: SimConfig, n_airlines: int, episodes: int, seed: int, label: str):
    rng = random.Random(seed)
    airlines = _init_airlines(n_airlines, rng)

    all_stats = []
    for ep in range(episodes):
        stats = run_episode(cfg, airlines, rng, ep, label)
        all_stats.append(stats)
        # each airline updates using per-airline profit share (simple split by revenue - comp share)
        # For the draft we just give each airline the same total-profit signal to encourage stable learning.
        for a in airlines:
            a.learn(stats.total_profit / max(1, n_airlines))
    return all_stats


def main() -> int:
    cfg = SimConfig()
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        ("monopoly", 1, 120, 7),
        ("duopoly", 2, 120, 8),
        ("triopoly", 3, 120, 9),
    ]

    payload = {"config": asdict(cfg), "experiments": {}}
    lines = []
    for label, n_airlines, episodes, seed in experiments:
        stats = run_market(cfg, n_airlines, episodes, seed, label)
        burn_in = 40
        summary = summarize_stats(stats[burn_in:])
        payload["experiments"][label] = {
            "airlines": n_airlines,
            "episodes": episodes,
            "burn_in": burn_in,
            "summary": summary,
        }
        lines.append(
            f"{label:9s}  airlines={n_airlines}  "
            f"avg_price_paid=${summary['avg_price_paid']:.2f}  "
            f"avg_profit=${summary['avg_profit']:.2f}  "
            f"avg_bump_rate={100*summary['avg_bump_rate']:.2f}%  "
            f"avg_tickets_sold={summary['avg_tickets_sold']:.1f}"
        )

    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

