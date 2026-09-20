"""Generate a tiny MIND-shaped dataset.

CI cannot download MIND (the real dataset is gated behind a license), so the end-to-end
smoke test runs against a synthetic dataset written in exactly the MIND TSV format. Every
stage of the pipeline therefore runs in CI on every push, on data small enough to finish
in seconds, using the same code paths as the real run.

Layout produced (mirrors the real extraction):

    <raw_dir>/train/{behaviors.tsv,news.tsv}
    <raw_dir>/dev/{behaviors.tsv,news.tsv}
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

from news_recsys.data.parse import TIME_FORMAT

CATEGORIES = {
    "news": ["newsworld", "newspolitics", "newscrime"],
    "sports": ["football_nfl", "basketball_nba", "baseball_mlb"],
    "finance": ["markets", "personalfinance"],
    "lifestyle": ["lifestyleroyals", "lifestylefood"],
    "health": ["healthnews", "fitness"],
}

_WORDS = [
    "market",
    "rally",
    "storm",
    "season",
    "quarterback",
    "verdict",
    "recipe",
    "vaccine",
    "senate",
    "budget",
    "playoff",
    "hurricane",
    "recall",
    "lawsuit",
    "startup",
    "earnings",
    "royal",
    "wedding",
    "rookie",
    "trade",
    "deadline",
    "forecast",
    "outbreak",
    "championship",
    "inflation",
]


def _sentence(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(n)).capitalize()


def _write_news(path: Path, news_ids: list[str], rng: random.Random) -> None:
    lines = []
    categories = list(CATEGORIES)
    for news_id in news_ids:
        category = rng.choice(categories)
        subcategory = rng.choice(CATEGORIES[category])
        title = _sentence(rng, rng.randint(6, 12))
        abstract = _sentence(rng, rng.randint(12, 30)) if rng.random() > 0.05 else ""
        url = f"https://example.invalid/{news_id}.html"
        lines.append(f"{news_id}\t{category}\t{subcategory}\t{title}\t{abstract}\t{url}\t[]\t[]")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_behaviors(
    path: Path,
    *,
    users: list[str],
    news_ids: list[str],
    start: datetime,
    days: int,
    impressions_per_day: int,
    rng: random.Random,
    first_impression_id: int,
) -> int:
    """Write one behaviours file; returns the next free impression id."""
    lines = []
    impression_id = first_impression_id
    # A popularity skew so that time-aware popularity features are not degenerate.
    weights = [1.0 / (rank + 3) for rank in range(len(news_ids))]

    for day in range(days):
        day_start = start + timedelta(days=day)
        for _ in range(impressions_per_day):
            user = rng.choice(users)
            stamp = day_start + timedelta(seconds=rng.randint(0, 86_399))
            history_size = rng.randint(0, 8)
            history = rng.sample(news_ids, history_size)
            slate_size = rng.randint(4, 10)
            slate = rng.choices(news_ids, weights=weights, k=slate_size)
            slate = list(dict.fromkeys(slate))  # dedupe, keep order
            clicked_index = rng.randrange(len(slate)) if rng.random() < 0.8 else -1
            entries = [
                f"{news_id}-{1 if index == clicked_index else 0}"
                for index, news_id in enumerate(slate)
            ]
            lines.append(
                "\t".join(
                    [
                        str(impression_id),
                        user,
                        stamp.strftime(TIME_FORMAT).lstrip("0"),
                        " ".join(history),
                        " ".join(entries),
                    ]
                )
            )
            impression_id += 1

    # MIND files are not sorted by time; sorting here would hide ordering bugs.
    rng.shuffle(lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return impression_id


def write_synthetic_mind(
    raw_dir: Path,
    *,
    n_users: int = 120,
    n_news: int = 300,
    train_days: int = 3,
    impressions_per_day: int = 200,
    seed: int = 42,
) -> Path:
    """Write a synthetic MIND dataset under ``raw_dir``; returns ``raw_dir``.

    ``train`` spans ``train_days`` days (the last of which becomes the validation fold)
    and ``dev`` is the following day, mirroring the real MIND-small layout. Roughly a
    third of the articles appearing in ``dev`` are new, so the cold-start analysis has
    something to measure.
    """
    rng = random.Random(seed)
    users_train = [f"U{index}" for index in range(n_users)]
    users_dev = [f"U{index}" for index in range(n_users // 2, n_users + n_users // 2)]

    all_news = [f"N{index}" for index in range(n_news)]
    train_news = all_news[: int(n_news * 0.7)]
    dev_news = all_news[int(n_news * 0.45) :]  # overlaps train, plus fresh articles

    start = datetime(2019, 11, 9, 0, 0, 0)
    train_dir = raw_dir / "train"
    dev_dir = raw_dir / "dev"
    train_dir.mkdir(parents=True, exist_ok=True)
    dev_dir.mkdir(parents=True, exist_ok=True)

    _write_news(train_dir / "news.tsv", train_news, rng)
    _write_news(dev_dir / "news.tsv", dev_news, rng)

    next_id = _write_behaviors(
        train_dir / "behaviors.tsv",
        users=users_train,
        news_ids=train_news,
        start=start,
        days=train_days,
        impressions_per_day=impressions_per_day,
        rng=rng,
        first_impression_id=1,
    )
    _write_behaviors(
        dev_dir / "behaviors.tsv",
        users=users_dev,
        news_ids=dev_news,
        start=start + timedelta(days=train_days),
        days=1,
        impressions_per_day=impressions_per_day,
        rng=rng,
        first_impression_id=next_id,
    )
    return raw_dir
