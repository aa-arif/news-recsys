"""Record published MIND-small numbers, with citations, for the comparison table.

These are **not** reproduced here - they are transcribed from the papers' own result
tables, each with a link, exactly as the repo's integrity rule requires ("never estimate
published numbers; cite them with links or leave them out"). Keeping them in a script that
writes JSON means the README generator can render them without anyone retyping a number.

Comparability caveats are recorded alongside the numbers, because they matter more than
the numbers do:

* Both sources evaluate on the MIND-small ``dev`` split. The second one states the
  evaluation set contains 73,152 impressions, which is exactly the fold this repo seals as
  its test set - so the denominators match.
* Those models train on the whole MIND-small ``train`` split. This repo holds out its last
  calendar day as validation, so it trains on ~81% of those impressions.
* They are content-only neural rankers over the impression slate. The models here
  additionally use time-aware popularity and CTR counters, computed causally from earlier
  events. That is a different *information set*, not a better architecture, and any
  comparison has to say so.
"""

from __future__ import annotations

import argparse

from news_recsys.config import get_settings
from news_recsys.io_utils import write_json
from news_recsys.logging_utils import get_logger

logger = get_logger("scripts.published_baselines")

SOURCES = {
    "digat": {
        "title": "DIGAT: Modeling News Recommendation with Dual-Graph Interaction",
        "arxiv": "2210.05196",
        "url": "https://arxiv.org/abs/2210.05196",
        "table": "Table 1 (MIND-small)",
        "retrieved": "2026-09-20",
    },
    "pointwise_pairwise": {
        "title": "Efficient Pointwise-Pairwise Learning-to-Rank for News Recommendation",
        "arxiv": "2409.17711",
        "url": "https://arxiv.org/abs/2409.17711",
        "table": "main results table; evaluation set stated as 73,152 impressions",
        "retrieved": "2026-09-20",
    },
}

#: Values are percentages exactly as printed in the cited tables.
PUBLISHED = [
    {"model": "NAML", "auc": 66.12, "mrr": 31.53, "ndcg@5": 34.88, "ndcg@10": 41.09, "source": "digat"},
    {"model": "LSTUR", "auc": 65.87, "mrr": 30.78, "ndcg@5": 33.95, "ndcg@10": 40.15, "source": "digat"},
    {"model": "NRMS", "auc": 65.63, "mrr": 30.96, "ndcg@5": 34.13, "ndcg@10": 40.52, "source": "digat"},
    {"model": "DIGAT", "auc": 68.77, "mrr": 33.46, "ndcg@5": 37.14, "ndcg@10": 43.39, "source": "digat"},
    {
        "model": "BERT-NRMS",
        "auc": 68.60,
        "mrr": 32.97,
        "ndcg@5": 36.55,
        "ndcg@10": 42.78,
        "source": "pointwise_pairwise",
    },
    {
        "model": "Prompt4NR",
        "auc": 68.48,
        "mrr": 33.29,
        "ndcg@5": 37.12,
        "ndcg@10": 43.25,
        "source": "pointwise_pairwise",
    },
    {
        "model": "UniTRec",
        "auc": 68.59,
        "mrr": 33.76,
        "ndcg@5": 37.63,
        "ndcg@10": 43.74,
        "source": "pointwise_pairwise",
    },
]

CAVEATS = [
    "Published numbers are transcribed from the cited tables, not reproduced here.",
    "Both sources evaluate on MIND-small dev; one states the evaluation set has 73,152 "
    "impressions, matching this repo's sealed test fold exactly.",
    "Those models train on all of MIND-small train; this repo holds out its last calendar "
    "day for validation and therefore trains on less data.",
    "Those models are content-only neural rankers. The models here also use time-aware "
    "popularity/CTR counters computed causally from earlier events - a different "
    "information set, which is the most likely explanation for any gap in either direction.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, choices=["small", "large", "synthetic"])
    args = parser.parse_args()
    settings = get_settings(dataset=args.dataset) if args.dataset else get_settings()
    settings.ensure_dirs()

    payload = {
        "dataset": "MIND-small dev split",
        "units": "percent, as printed in the cited tables",
        "sources": SOURCES,
        "results": PUBLISHED,
        "caveats": CAVEATS,
    }
    path = write_json(settings.metrics_dir / "published_baselines.json", payload)
    logger.info("recorded %d published results -> %s", len(PUBLISHED), path)


if __name__ == "__main__":
    main()
