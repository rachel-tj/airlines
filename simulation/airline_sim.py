from __future__ import annotations

import dataclasses
import math
import random
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class AirlineQuality:
    carrier: str
    flights: int
    cancel_rate: float
    ontime_rate: float
    avg_arr_delay_min: float


@dataclass(frozen=True)
class AirlineDataProfile:
    carrier: str
    fare_samples: list[float]
    quality: AirlineQuality

    def fare_median(self) -> float:
        xs = sorted(self.fare_samples)
        if not xs:
            return 300.0
        mid = len(xs) // 2
        return xs[mid] if len(xs) % 2 else 0.5 * (xs[mid - 1] + xs[mid])


@dataclasses.dataclass(frozen=True)
class SimConfig:
    horizon_days: int = 30
    capacity: int = 180
    show_prob: float = 0.93
    bump_cost: float = 400.0

    # Market / demand
    consumers: int = 600
    arrival_spread: int = 30  # arrival day uniform in [0, arrival_spread)
    valuation_mean: float = 320.0
    valuation_sd: float = 110.0
    outside_option: float = 0.0

    # Choice model: probability consumer selects airline i given prices p_i
    # softmax over (-beta * price)
    price_sensitivity: float = 0.020
    # Utility bonus for airline quality (higher = more weight on on-time / reliability)
    quality_sensitivity: float = 2.0


@dataclasses.dataclass
class EpisodeStats:
    episode: int
    market: str
    airlines: int
    total_tickets_sold: int
    total_bumped: int
    total_shows: int
    total_revenue: float
    total_comp: float
    total_profit: float
    avg_price_paid: float
    avg_quality_score: float

    def bump_rate(self) -> float:
        return 0.0 if self.total_tickets_sold == 0 else self.total_bumped / self.total_tickets_sold


@dataclasses.dataclass
class AirlineParams:
    base_price: float
    slope: float  # controls how prices change as departure nears
    overbook_rate: float  # max tickets sold = capacity * (1 + overbook_rate)


class Airline:
    def __init__(
        self,
        name: str,
        params: AirlineParams,
        rng: random.Random,
        data_profile: AirlineDataProfile | None = None,
    ):
        self.name = name
        self.params = params
        self._rng = rng
        self.data_profile = data_profile

        # Learning state
        self._last_profit: float | None = None
        self._last_params: AirlineParams | None = None

    def price_on_day(self, day: int, horizon_days: int) -> float:
        # day in [0, horizon_days-1], closer to departure => larger tau
        tau = (horizon_days - 1 - day) / max(1, horizon_days - 1)
        # Higher slope => cheaper early, more expensive late (standard airline pattern)
        # base_price anchors near departure.
        price = self.params.base_price * math.exp(-self.params.slope * tau)
        return max(30.0, price)

    def max_sellable(self, capacity: int) -> int:
        return max(capacity, int(round(capacity * (1.0 + self.params.overbook_rate))))

    def propose_mutation(self, scale: float = 0.08) -> AirlineParams:
        # Log-normal style perturbations for positivity.
        bp = self.params.base_price * math.exp(self._rng.gauss(0.0, scale))
        slope = max(0.0, self.params.slope + self._rng.gauss(0.0, scale))
        ob = min(0.40, max(0.0, self.params.overbook_rate + self._rng.gauss(0.0, scale / 2.0)))
        return AirlineParams(base_price=bp, slope=slope, overbook_rate=ob)

    def learn(self, realized_profit: float, temperature: float = 0.10) -> None:
        """
        Simple hill-climb / bandit update:
        - Mutate parameters each episode.
        - Accept if profit improves; otherwise accept with small probability.
        """
        if self._last_profit is None:
            self._last_profit = realized_profit
            self._last_params = self.params
            # start exploring next round
            self.params = self.propose_mutation()
            return

        assert self._last_params is not None
        if realized_profit >= self._last_profit:
            self._last_profit = realized_profit
            self._last_params = self.params
        else:
            # accept worse move with probability exp((p_new - p_old) / (temp * |p_old|))
            denom = temperature * max(1.0, abs(self._last_profit))
            accept_prob = math.exp((realized_profit - self._last_profit) / denom)
            if self._rng.random() < accept_prob:
                self._last_profit = realized_profit
                self._last_params = self.params
            else:
                # revert
                self.params = self._last_params

        # propose next exploration step
        self.params = self.propose_mutation()


@dataclasses.dataclass(frozen=True)
class Consumer:
    arrival_day: int
    deadline_day: int
    valuation: float
    kind: str  # "early", "late", "adaptive"


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else hi if x > hi else x


def generate_consumers(cfg: SimConfig, rng: random.Random) -> list[Consumer]:
    consumers: list[Consumer] = []
    for _ in range(cfg.consumers):
        arrival = rng.randrange(0, max(1, cfg.arrival_spread))
        # deadline: arrival + 3..15 days, but within horizon
        deadline = arrival + rng.randrange(3, 16)
        deadline = _clamp_int(deadline, 0, cfg.horizon_days - 1)

        # truncated normal for valuations (no numpy dependency)
        v = rng.gauss(cfg.valuation_mean, cfg.valuation_sd)
        v = max(40.0, v)

        r = rng.random()
        if r < 0.40:
            kind = "early"
        elif r < 0.80:
            kind = "late"
        else:
            kind = "adaptive"
        consumers.append(Consumer(arrival_day=arrival, deadline_day=deadline, valuation=v, kind=kind))
    return consumers


def _softmax_choice(weights: list[float], rng: random.Random) -> int | None:
    total = sum(weights)
    if total <= 0.0:
        return None
    x = rng.random() * total
    running = 0.0
    for i, w in enumerate(weights):
        running += w
        if x <= running:
            return i
    return len(weights) - 1


def quality_score(q: AirlineQuality) -> float:
    """
    Combine on-time performance fields into a single [~0,1]-ish score.
    This is intentionally simple for the draft.
    """
    # Penalize cancellations strongly, average delay mildly.
    delay_penalty = min(1.0, max(0.0, q.avg_arr_delay_min / 60.0))  # 0..1 for 0..60min
    score = q.ontime_rate - 2.0 * q.cancel_rate - 0.25 * delay_penalty
    return score


def _consumer_buy_now(kind: str, day: int, deadline: int, price: float, observed_min_to_date: float, valuation: float) -> bool:
    if price > valuation:
        return False

    if kind == "early":
        return True
    if kind == "late":
        # Only buy at deadline.
        return day >= deadline
    if kind == "adaptive":
        # Buy if price is close to best seen so far, or at deadline.
        return day >= deadline or price <= 1.05 * observed_min_to_date
    return False


def run_episode(
    cfg: SimConfig,
    airlines: list[Airline],
    rng: random.Random,
    episode_index: int,
    market_label: str,
) -> EpisodeStats:
    horizon = cfg.horizon_days
    cap = cfg.capacity

    consumers = generate_consumers(cfg, rng)
    consumers_by_day: list[list[Consumer]] = [[] for _ in range(horizon)]
    for c in consumers:
        consumers_by_day[_clamp_int(c.arrival_day, 0, horizon - 1)].append(c)

    sold = [0 for _ in airlines]
    revenue = [0.0 for _ in airlines]
    sold_quality = 0.0
    # track the min price seen so far for "adaptive" consumers
    min_price_so_far = [float("inf") for _ in range(horizon)]

    for day in range(horizon):
        prices = [a.price_on_day(day, horizon) for a in airlines]
        min_price_so_far[day] = min(prices) if day == 0 else min(min_price_so_far[day - 1], min(prices))

        for c in consumers_by_day[day]:
            chosen_day = None
            for d in range(day, c.deadline_day + 1):
                prices_d = [a.price_on_day(d, horizon) for a in airlines]
                best_seen = min_price_so_far[d] if d <= day else min(min_price_so_far[day], min(prices_d))

                # consumer chooses an airline probabilistically via logit, but may choose not to buy at all
                weights = []
                for i, p in enumerate(prices_d):
                    if sold[i] >= airlines[i].max_sellable(cap):
                        weights.append(0.0)
                        continue
                    q_bonus = 0.0
                    if airlines[i].data_profile is not None:
                        q_bonus = cfg.quality_sensitivity * quality_score(airlines[i].data_profile.quality)
                    w = math.exp(-cfg.price_sensitivity * p + q_bonus)
                    weights.append(w)
                idx = _softmax_choice(weights, rng)
                if idx is None:
                    continue
                p = prices_d[idx]

                if _consumer_buy_now(c.kind, d, c.deadline_day, p, best_seen, c.valuation):
                    chosen_day = d
                    sold[idx] += 1
                    revenue[idx] += p
                    if airlines[idx].data_profile is not None:
                        sold_quality += quality_score(airlines[idx].data_profile.quality)
                    break

            _ = chosen_day  # explicit: used only for readability in draft experiments

    total_tickets_sold = sum(sold)
    total_revenue = sum(revenue)

    total_bumped = 0
    total_shows = 0
    total_comp = 0.0
    for i, a in enumerate(airlines):
        shows = sum(1 for _ in range(sold[i]) if rng.random() < cfg.show_prob)
        total_shows += shows
        bumped = max(0, shows - cap)
        total_bumped += bumped
        comp = cfg.bump_cost * bumped
        total_comp += comp

    total_profit = total_revenue - total_comp
    avg_price_paid = 0.0 if total_tickets_sold == 0 else total_revenue / total_tickets_sold
    avg_quality_score = 0.0 if total_tickets_sold == 0 else sold_quality / total_tickets_sold

    return EpisodeStats(
        episode=episode_index,
        market=market_label,
        airlines=len(airlines),
        total_tickets_sold=total_tickets_sold,
        total_bumped=total_bumped,
        total_shows=total_shows,
        total_revenue=total_revenue,
        total_comp=total_comp,
        total_profit=total_profit,
        avg_price_paid=avg_price_paid,
        avg_quality_score=avg_quality_score,
    )


def summarize_stats(stats: Iterable[EpisodeStats]) -> dict[str, float]:
    stats = list(stats)
    if not stats:
        return {}
    n = len(stats)
    return {
        "episodes": float(n),
        "avg_tickets_sold": sum(s.total_tickets_sold for s in stats) / n,
        "avg_bump_rate": sum(s.bump_rate() for s in stats) / n,
        "avg_profit": sum(s.total_profit for s in stats) / n,
        "avg_price_paid": sum(s.avg_price_paid for s in stats) / n,
        "avg_quality_score": sum(s.avg_quality_score for s in stats) / n,
    }
