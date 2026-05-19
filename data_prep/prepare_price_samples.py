from __future__ import annotations

import gzip
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class SampleSummary:
    carrier: str
    sampled: int
    kept: int


def _reservoir_add(reservoir: list[float], x: float, k: int, seen: int, rng: random.Random) -> None:
    if len(reservoir) < k:
        reservoir.append(x)
        return
    j = rng.randrange(0, seen)
    if j < k:
        reservoir[j] = x


def main() -> int:
    prices_root = Path("data/prices_extracted")
    if not prices_root.exists():
        raise SystemExit(f"Missing extracted price data dir: {prices_root}")

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "db1b_fare_samples_public_21fields.json.gz"

    # Keep at most K fares per carrier across all files (reservoir sampling).
    k_per_carrier = 20000
    rng = random.Random(12345)

    fares_by_carrier: dict[str, list[float]] = {}
    seen_by_carrier: dict[str, int] = {}
    sampled_by_carrier: dict[str, int] = {}

    files = sorted(prices_root.rglob("*.asc"))
    if not files:
        raise SystemExit(f"No .asc files found under {prices_root}")

    for file_i, path in enumerate(files, start=1):
        print(f"[{file_i}/{len(files)}] Scanning {path} ...", flush=True)
        with open(path, "rb") as f:
            for raw in f:
                s = raw.decode("latin-1", "replace").rstrip("\n").rstrip("\r")
                parts = s.split("|")
                if len(parts) != 21:
                    continue
                try:
                    fare = float(parts[0])
                except ValueError:
                    continue
                if not (0.0 < fare < 5000.0):
                    continue
                carrier = parts[1].strip()
                if not carrier or len(carrier) > 3:
                    continue

                sampled_by_carrier[carrier] = sampled_by_carrier.get(carrier, 0) + 1
                seen = seen_by_carrier.get(carrier, 0) + 1
                seen_by_carrier[carrier] = seen
                reservoir = fares_by_carrier.setdefault(carrier, [])
                _reservoir_add(reservoir, fare, k_per_carrier, seen, rng)

    summaries: list[SampleSummary] = []
    for carrier, fares in sorted(fares_by_carrier.items()):
        summaries.append(
            SampleSummary(
                carrier=carrier,
                sampled=int(sampled_by_carrier.get(carrier, 0)),
                kept=len(fares),
            )
        )

    payload = {
        "source_dir": str(prices_root),
        "filter": "DB1B .asc lines with exactly 21 fields; fare=field0; carrier=field1",
        "k_per_carrier": k_per_carrier,
        "carriers": {c: fares_by_carrier[c] for c in sorted(fares_by_carrier.keys())},
        "summaries": [asdict(s) for s in summaries],
    }

    with gzip.open(out_path, "wt", encoding="utf-8") as gz:
        json.dump(payload, gz)
    print(f"Wrote {out_path} ({len(fares_by_carrier)} carriers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

