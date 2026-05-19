from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class AirlineQuality:
    carrier: str
    flights: int
    cancel_rate: float
    ontime_rate: float
    avg_arr_delay_min: float


def main() -> int:
    src = Path("data/On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2026_1.csv")
    if not src.exists():
        raise SystemExit(f"Missing on-time dataset: {src}")

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "airline_quality_2026_01.json"

    usecols = [
        "Reporting_Airline",
        "Cancelled",
        "ArrDel15",
        "ArrDelayMinutes",
    ]
    df = pd.read_csv(src, usecols=usecols)

    df["Cancelled"] = df["Cancelled"].fillna(0.0)
    df["ArrDel15"] = df["ArrDel15"].fillna(0.0)
    # For average delay: treat missing delays as 0 for non-cancelled flights where delay is absent
    df["ArrDelayMinutes"] = df["ArrDelayMinutes"].fillna(0.0)

    # Define on-time as: not cancelled and not ArrDel15
    df["OnTime"] = ((df["Cancelled"] == 0.0) & (df["ArrDel15"] == 0.0)).astype(float)
    df["NotCancelled"] = (df["Cancelled"] == 0.0).astype(float)

    g = df.groupby("Reporting_Airline", as_index=False)
    agg = g.agg(
        flights=("Reporting_Airline", "size"),
        cancel_rate=("Cancelled", "mean"),
        ontime_rate=("OnTime", "mean"),
    )

    # average delay among not-cancelled flights
    not_cancelled = df[df["Cancelled"] == 0.0]
    g2 = not_cancelled.groupby("Reporting_Airline", as_index=False).agg(avg_arr_delay_min=("ArrDelayMinutes", "mean"))
    merged = agg.merge(g2, on="Reporting_Airline", how="left").fillna({"avg_arr_delay_min": 0.0})

    qualities: list[AirlineQuality] = []
    for _, row in merged.iterrows():
        qualities.append(
            AirlineQuality(
                carrier=str(row["Reporting_Airline"]),
                flights=int(row["flights"]),
                cancel_rate=float(row["cancel_rate"]),
                ontime_rate=float(row["ontime_rate"]),
                avg_arr_delay_min=float(row["avg_arr_delay_min"]),
            )
        )

    qualities.sort(key=lambda q: q.flights, reverse=True)

    payload = {
        "source_file": str(src),
        "month": "2026-01",
        "airlines": [asdict(q) for q in qualities],
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote {out_path} ({len(qualities)} airlines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

