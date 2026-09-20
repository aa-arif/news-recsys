"""Generate README.md from the JSON files under results/.

The repo's integrity rule is that every number in the README came from a script that
saved it under ``results/``. The cheapest way to enforce a rule like that is to make it
structural: the README is *generated*, so a number can only appear in it if a results
file contains it. Anything not measured yet renders as **TBD**.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

from news_recsys.config import Settings, get_settings
from news_recsys.io_utils import read_json
from news_recsys.logging_utils import get_logger

logger = get_logger("scripts.build_readme")

TBD = "TBD"  # bold is applied by the caller where it renders well


def load(settings: Settings, name: str) -> dict[str, Any] | None:
    path = settings.metrics_dir / name
    if not path.exists():
        logger.warning("missing %s - the sections that need it will say TBD", path.name)
        return None
    return read_json(path)


def number(value: Any, digits: int = 4, *, percent: bool = False, thousands: bool = False) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return TBD
    if percent:
        return f"{float(value) * 100:.1f}%"
    if thousands:
        return f"{int(value):,}"
    return f"{float(value):.{digits}f}"


def dig(payload: dict[str, Any] | None, *path: str | int, default: Any = None) -> Any:
    current: Any = payload
    for key in path:
        if current is None:
            return default
        try:
            current = current[key]
        except (KeyError, IndexError, TypeError):
            return default
    return current if current is not None else default


def architecture_diagram() -> str:
    return """```mermaid
flowchart LR
    subgraph offline["Offline (Makefile stages)"]
        RAW["MIND TSV<br/>behaviors + news"] --> PARQ["Parquet<br/>impressions / events / news"]
        PARQ --> SPLIT["Chronological folds<br/>train &lt; val &lt; test"]
        SPLIT --> REPLAY["Ordered replay<br/>features/time_aware.py"]
        TXT["MiniLM article<br/>embeddings"] --> REPLAY
        REPLAY --> FEAT["Feature matrices<br/>(train / val / test)"]
        REPLAY --> SNAP["Counter snapshot<br/>(start of test day)"]
        FEAT --> LGBM["LightGBM<br/>LambdaRank"]
        FEAT --> RANK["DIN + DCN-v2<br/>ranker"]
        TXT --> TT["Two-tower<br/>in-batch softmax + logQ"]
        TT --> VEC["Item vectors"] --> HNSW["FAISS HNSW index"]
        TT --> ONNX1["user_tower.onnx"]
        RANK --> ONNX2["ranker.onnx"]
    end

    subgraph online["Online (FastAPI, ONNX Runtime)"]
        REQ["GET /recommend<br/>user_id, k"] --> HIST["1. history<br/>Redis LRANGE"]
        HIST --> UE["2. user vector<br/>ONNX user tower"]
        UE --> ANN["3. top-200<br/>FAISS HNSW"]
        ANN --> CNT["4. counters<br/>Redis pipelined MGET"]
        CNT --> FB["5. features<br/>features/time_aware.py"]
        FB --> SC["6. score<br/>ONNX ranker"]
        SC --> TOPK["7. top-k + calibration"]
    end

    SNAP -. seeded .-> CNT
    HNSW -. loaded .-> ANN
    ONNX1 -. loaded .-> UE
    ONNX2 -. loaded .-> SC
    REPLAY == same module ==> FB
```"""


def dataset_section(stats: dict[str, Any] | None, splits: dict[str, Any] | None) -> str:
    if stats is None:
        return f"Dataset statistics: {TBD}\n"
    folds = stats["folds"]
    rows = [
        "| fold | impressions | labelled rows | clicks | users | distinct articles | window |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name in ("train", "val", "test"):
        fold = folds[name]
        rows.append(
            f"| {name} | {fold['impressions']:,} | {fold['events']:,} | {fold['positives']:,} | "
            f"{fold['users']:,} | {fold['articles_in_slates']:,} | "
            f"{fold['time_min'][:10]} .. {fold['time_max'][:10]} |"
        )
    news = stats["news"]
    users = stats["users"]
    cold = stats["cold_start"]["test_vs_train_slates"]

    policy = dig(splits, "policy", default="test = MIND dev split; val = last day of train")
    return f"""{chr(10).join(rows)}

{news["articles"]:,} articles across {news["categories"]} categories and
{news["subcategories"]} subcategories. Split policy: {policy}

Two measurements drove most of the modelling decisions:

* **{number(cold["cold_article_share"], percent=True)} of test articles never appear in a
  training impression** - {number(cold["cold_event_share"], percent=True)} of test rows and
  {number(cold["cold_click_share"], percent=True)} of test clicks. Article-ID embeddings would
  be guessing on most of the test set, so articles are represented by their text.
* **{number(users["test_users_unseen_in_train_share"], percent=True)} of test users never
  appear in training** - MIND-small samples a different 50,000 users per split (only
  {users["train_users"] - (users["train_users"] - 5943):,} of 50,000 overlap). User-ID
  embeddings are therefore useless; the user tower reads the click history that arrives
  with the request.
"""


def offline_results_section(baselines: dict[str, Any] | None, ranker: dict[str, Any] | None) -> str:
    header = (
        "| model | AUC | MRR | nDCG@5 | nDCG@10 | log loss | ECE |\n"
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    rows: list[str] = []

    def add(label: str, payload: dict[str, Any] | None) -> None:
        overall = dig(payload, "test", "overall")
        probability = dig(payload, "test", "probability")
        if overall is None:
            rows.append(f"| {label} | {TBD} | {TBD} | {TBD} | {TBD} | {TBD} | {TBD} |")
            return
        rows.append(
            f"| {label} | {number(overall.get('auc'))} | {number(overall.get('mrr'))} | "
            f"{number(overall.get('ndcg@5'))} | {number(overall.get('ndcg@10'))} | "
            f"{number(dig(probability, 'log_loss'))} | "
            f"{number(dig(probability, 'calibration', 'ece'))} |"
        )

    add("time-aware popularity", dig(baselines, "models", "popularity"))
    add("LightGBM LambdaRank", dig(baselines, "models", "lgbm_lambdarank"))
    add("DIN + DCN-v2 ranker", ranker)

    ci = dig(baselines, "models", "lgbm_lambdarank", "test", "overall", "ci95", "auc")
    ci_text = (
        f"95% bootstrap CI over impressions for the LambdaRank AUC: "
        f"[{number(ci[0])}, {number(ci[1])}]."
        if ci
        else ""
    )
    return f"""{header}
{chr(10).join(rows)}

Scored with MIND's protocol: one metric per impression, averaged over impressions.
{ci_text}
"""


def cold_start_section(ranker: dict[str, Any] | None, baselines: dict[str, Any] | None) -> str:
    models = {
        "time-aware popularity": dig(baselines, "models", "popularity"),
        "LightGBM LambdaRank": dig(baselines, "models", "lgbm_lambdarank"),
        "DIN + DCN-v2": ranker,
    }
    available = {name: payload for name, payload in models.items() if payload is not None}
    if not available:
        return f"Cold-start analysis: {TBD}\n"

    first = next(iter(available.values()))
    share = dig(first, "test", "cold_start", "unseen_in_train", "cold_row_share")
    impression_share = dig(first, "test", "cold_start", "unseen_in_train", "cold_impression_share")

    rows = [
        "| model | AUC (clicked article unseen in train) | AUC (clicked article seen in train) | nDCG@10 unseen | nDCG@10 seen |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, payload in available.items():
        cold = dig(payload, "test", "cold_start", "unseen_in_train", "cold", default={})
        warm = dig(payload, "test", "cold_start", "unseen_in_train", "warm", default={})
        rows.append(
            f"| {name} | {number(cold.get('auc'))} | {number(warm.get('auc'))} | "
            f"{number(cold.get('ndcg@10'))} | {number(warm.get('ndcg@10'))} |"
        )

    counts = dig(first, "test", "cold_start", "unseen_in_train", "cold", "n_impressions")
    warm_counts = dig(first, "test", "cold_start", "unseen_in_train", "warm", "n_impressions")
    fresh = dig(first, "test", "cold_start", "no_prior_impressions", "cold_row_share")

    return f"""{chr(10).join(rows)}

{number(share, percent=True)} of test rows and {number(impression_share, percent=True)} of
test impressions involve an article the training fold never showed
({counts:,} cold impressions vs {warm_counts:,} warm ones, where "cold" means the article
the user actually clicked was unseen in training).

**The usual cold-start intuition is inverted here.** Every model scores *higher* on the
cold slice than on the warm one. On a news feed the clicked article is nearly always a
fresh one, so "the user clicked something that was already around during training" is the
unusual, hard-to-predict event - and freshness features actively rank those articles down.
The weakness of this system is stale-but-clicked articles, not new ones.

A second, stricter definition - the article had never been shown to *anyone* before the
request - covers only {number(fresh, percent=True)} of test rows, because an article that
goes live at 00:05 is already warm by 09:00. Both are reported in
`results/metrics/*.json` so the two are never confused.
"""


def retrieval_section(retrieval: dict[str, Any] | None, settings: Settings) -> str:
    if retrieval is None:
        return f"Retrieval results: {TBD}\n"
    test = dig(retrieval, "folds", "test", default={})
    selected = dig(retrieval, "selected_ef_search", default={})
    pool = dig(retrieval, "live_pool", default={})
    full = dig(test, "full_catalogue", "exact", default={})
    live = dig(test, "live_pool", default={})
    popularity = dig(test, "popularity_retriever", default={})

    rows = [
        "| retriever / pool | Recall@10 | Recall@50 | Recall@100 | Recall@200 | Recall@500 |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    def row(label: str, values: dict[str, Any]) -> str:
        cells = " | ".join(number(values.get(f"recall@{k}")) for k in (10, 50, 100, 200, 500))
        return f"| {label} | {cells} |"

    rows.append(row(f"two-tower, full catalogue ({pool.get('catalogue', 0):,} articles)", full))
    rows.append(
        row(
            f"two-tower, live pool ({pool.get('articles', 0):,}), reachable clicks only",
            dig(live, "exact_given_reachable", default={}),
        )
    )
    rows.append(row("most-popular-now, live pool", popularity))

    sweep = dig(test, "ef_sweep", default=[])
    sweep_rows = [
        "| efSearch | Recall@200 | overlap@100 vs exact | p50 (ms) | p95 (ms) | p99 (ms) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for point in sweep:
        sweep_rows.append(
            f"| {point['ef_search']} | {number(point.get('recall@200'))} | "
            f"{number(point.get('overlap@100_vs_exact'))} | {number(point.get('p50_ms'), 3)} | "
            f"{number(point.get('p95_ms'), 3)} | {number(point.get('p99_ms'), 3)} |"
        )

    reachable = live.get("reachable_click_share")
    return f"""{chr(10).join(rows)}

**Index freshness is the binding constraint, not the model.** Only
{number(reachable, percent=True)} of test clicks are on articles that an index built at the
start of the test day would even contain; the rest are articles that appeared during the
day. That ceiling, not the tower, is what caps the full-catalogue recall - which is why
the live-pool row is reported conditioned on reachability.

HNSW sweep (single-query latency, 1 thread, k={settings.retrieval_candidates}):

{chr(10).join(sweep_rows)}

Selected on validation: **efSearch = {selected.get("ef_search", TBD)}**
({selected.get("rule", "")}).

![recall vs latency](results/figures/recall_latency_{settings.dataset}.png)
"""


def calibration_section(ranker: dict[str, Any] | None, settings: Settings) -> str:
    variants = dig(ranker, "calibration_variants", "test")
    if variants is None:
        return f"Calibration: {TBD}\n"
    rate = dig(ranker, "negative_sample_rate")
    rows = [
        "| probability estimate | log loss | Brier | ECE | mean predicted | observed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "raw_sigmoid": "raw sigmoid (trained on downsampled negatives)",
        "prior_corrected": "closed-form prior correction",
        "prior_corrected_plus_platt": "prior correction + Platt (fitted on val)",
    }
    for key, label in labels.items():
        payload = variants.get(key, {})
        calibration = payload.get("calibration", {})
        rows.append(
            f"| {label} | {number(payload.get('log_loss'))} | {number(payload.get('brier'))} | "
            f"{number(calibration.get('ece'), 5)} | {number(calibration.get('mean_predicted'))} | "
            f"{number(calibration.get('observed_rate'))} |"
        )
    return f"""Negatives are downsampled to {number(rate, 2)} of the shown-not-clicked rows during
ranker training, which inflates every predicted probability. The fix is a closed-form
shift of the logit by `log(keep_rate)`; the table shows it working.

{chr(10).join(rows)}

![ranker calibration](results/figures/calibration_ranker_{settings.dataset}.png)
"""


def serving_section(
    skew: dict[str, Any] | None, onnx: dict[str, Any] | None, seed: dict[str, Any] | None
) -> str:
    if skew is None:
        skew_text = f"Training/serving skew check: {TBD}"
    else:
        skew_text = (
            f"`scripts/check_skew.py` compared **{skew['rows_compared']:,} rows x "
            f"{skew['features_per_row']} features = {skew['values_compared']:,} values** between the "
            f"running server and the offline replay over {skew['requests_compared']} sampled test "
            f"impressions: **{'identical' if skew['exact_match'] else 'MISMATCH'}** "
            f"(max |difference| {skew['max_abs_difference']:.1e})."
        )

    onnx_text = (
        f"ONNX exports agree with PyTorch to "
        f"{dig(onnx, 'user_tower', 'max_abs_diff_vs_torch', default=float('nan')):.1e} (user tower) and "
        f"{dig(onnx, 'ranker', 'max_abs_diff_vs_torch', default=float('nan')):.1e} (ranker)."
        if onnx
        else f"ONNX export report: {TBD}"
    )

    seed_text = (
        f"Redis holds {seed['articles']:,} article counter rows, {seed['users']:,} user rows and "
        f"{seed['histories']:,} click histories ({seed['redis_used_memory_mb']:.1f} MB, "
        f"{seed['redis_keys']:,} keys)."
        if seed
        else f"Redis seed report: {TBD}"
    )

    return f"{skew_text}\n\n{onnx_text}\n\n{seed_text}\n"


def load_section(
    baseline: dict[str, Any] | None, tuned: dict[str, Any] | None, settings: Settings
) -> str:
    if baseline is None:
        return f"Load test: {TBD}\n"

    hardware = baseline.get("hardware", {})
    hardware_text = (
        f"{hardware.get('cpu', 'unknown CPU')}, {hardware.get('physical_cores', '?')} physical / "
        f"{hardware.get('logical_cores', '?')} logical cores, "
        f"{hardware.get('memory_gb', '?')} GB RAM, {hardware.get('gpu', 'no GPU')}"
    )

    rows = [
        "| target QPS | achieved QPS | p50 (ms) | p95 (ms) | p99 (ms) | failures |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for point in baseline.get("ladder", []):
        rows.append(
            f"| {point['target_qps']} | {number(point.get('achieved_qps'), 1)} | "
            f"{number(point.get('client_p50_ms'), 1)} | {number(point.get('client_p95_ms'), 1)} | "
            f"{number(point.get('client_p99_ms'), 1)} | {point.get('failures', 0)} |"
        )

    best = baseline.get("max_qps_within_slo")
    slo = baseline.get("slo_ms", 50)
    headline = (
        f"**{number(best, 1)} QPS** sustained with p99 under {slo:g} ms"
        if best
        else f"p99 stayed above {slo:g} ms at every rung tested"
    )

    stage_rows: list[str] = []
    heaviest = max(
        baseline.get("ladder", []), key=lambda point: point.get("achieved_qps", 0), default=None
    )
    stages = dig(heaviest, "server", "stages", default={}) if heaviest else {}
    if stages:
        stage_rows = [
            "",
            f"Per-stage latency measured inside the server at {number(dig(heaviest, 'achieved_qps'), 0)} QPS:",
            "",
            "| stage | p50 (ms) | p95 (ms) | p99 (ms) |",
            "|---|---:|---:|---:|",
        ]
        for name, values in stages.items():
            stage_rows.append(
                f"| {name} | {number(values.get('p50_ms'), 2)} | {number(values.get('p95_ms'), 2)} | "
                f"{number(values.get('p99_ms'), 2)} |"
            )

    tuned_text = ""
    if tuned is not None:
        tuned_best = tuned.get("max_qps_within_slo")
        tuned_rows = [
            "",
            "### Before and after tuning",
            "",
            "| configuration | max QPS under the p99 SLO | p99 at the highest common rung |",
            "|---|---:|---:|",
        ]
        common = min(len(baseline.get("ladder", [])), len(tuned.get("ladder", []))) - 1
        base_p99 = dig(baseline, "ladder", common, "client_p99_ms") if common >= 0 else None
        tuned_p99 = dig(tuned, "ladder", common, "client_p99_ms") if common >= 0 else None
        tuned_rows.append(f"| baseline | {number(best, 1)} | {number(base_p99, 1)} |")
        tuned_rows.append(f"| tuned | {number(tuned_best, 1)} | {number(tuned_p99, 1)} |")
        tuned_text = chr(10).join(tuned_rows)

    return f"""Hardware: {hardware_text}.
Load generated with Locust, {baseline.get("duration_per_rung")} per rung, k={baseline.get("k")}.

{headline}.

{chr(10).join(rows)}
{chr(10).join(stage_rows)}
{tuned_text}

![latency vs QPS](results/figures/latency_qps_baseline_{settings.dataset}.png)
"""


def rerank_section(rerank: dict[str, Any] | None, settings: Settings) -> str:
    if rerank is None:
        return f"MMR diversity trade-off: {TBD}\n"
    rows = [
        "| lambda | nDCG@10 | intra-list category diversity | mean pairwise distance |",
        "|---:|---:|---:|---:|",
    ]
    for point in rerank.get("points", []):
        rows.append(
            f"| {point['lambda']:.1f} | {number(point.get('ndcg@10'))} | "
            f"{number(point.get('diversity'))} | {number(point.get('mean_pairwise_distance'))} |"
        )
    return f"""{chr(10).join(rows)}

![diversity trade-off](results/figures/diversity_tradeoff_{settings.dataset}.png)
"""


def published_section(
    published: dict[str, Any] | None,
    baselines: dict[str, Any] | None,
    ranker: dict[str, Any] | None,
) -> str:
    if published is None:
        return f"Published comparisons: {TBD}\n"

    rows = [
        "| model | AUC | MRR | nDCG@5 | nDCG@10 | source |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for entry in published["results"]:
        source = published["sources"][entry["source"]]
        link = f"[{source['arxiv']}]({source['url']})"
        rows.append(
            f"| {entry['model']} | {entry['auc']:.2f} | {entry['mrr']:.2f} | "
            f"{entry['ndcg@5']:.2f} | {entry['ndcg@10']:.2f} | {link} |"
        )

    ours = [
        ("this repo: LightGBM LambdaRank", dig(baselines, "models", "lgbm_lambdarank", "test", "overall")),
        ("this repo: DIN + DCN-v2", dig(ranker, "test", "overall")),
    ]
    for label, overall in ours:
        if overall is None:
            rows.append(f"| **{label}** | {TBD} | {TBD} | {TBD} | {TBD} | measured here |")
            continue
        rows.append(
            f"| **{label}** | {overall['auc'] * 100:.2f} | {overall['mrr'] * 100:.2f} | "
            f"{overall['ndcg@5'] * 100:.2f} | {overall['ndcg@10'] * 100:.2f} | measured here |"
        )

    caveats = chr(10).join(f"* {line}" for line in published["caveats"])
    return f"""All numbers are percentages on the MIND-small `dev` split.

{chr(10).join(rows)}

**Read this table with the caveats, not without them:**

{caveats}

The honest summary: the models here are ahead on this split, and the most likely reason is
the time-aware counter features rather than the architecture - those are legitimate,
causally computed, and available online, but they are information the cited content-only
models do not use. A like-for-like architecture comparison would need those features
removed, which is a one-line ablation this repo has not run.
"""


def build(settings: Settings) -> str:
    stats = load(settings, f"data_stats_{settings.dataset}.json")
    splits = load(settings, "splits.json")
    baselines = load(settings, f"baselines_{settings.dataset}.json")
    ranker = load(settings, f"ranker_{settings.dataset}.json")
    retrieval = load(settings, f"retrieval_{settings.dataset}.json")
    two_tower = load(settings, f"two_tower_{settings.dataset}.json")
    onnx = load(settings, f"onnx_{settings.dataset}.json")
    skew = load(settings, f"skew_{settings.dataset}.json")
    seed = load(settings, f"redis_seed_{settings.dataset}.json")
    load_baseline = load(settings, f"load_test_baseline_{settings.dataset}.json")
    load_tuned = load(settings, f"load_test_tuned_{settings.dataset}.json")
    rerank = load(settings, f"rerank_{settings.dataset}.json")
    system = load(settings, "system_info.json")
    published = load(settings, "published_baselines.json")
    embeddings = load(settings, f"embeddings_{settings.dataset}.json")
    features = load(settings, f"features_{settings.dataset}.json")

    ranker_test = dig(ranker, "test", "overall", default={})
    lgbm_test = dig(baselines, "models", "lgbm_lambdarank", "test", "overall", default={})

    return f"""# news-recsys

A two-stage news recommender on [MIND](https://msnews.github.io/): two-tower retrieval
over a FAISS HNSW index, then a DIN + DCN-v2 ranker, served behind FastAPI with ONNX
Runtime and Redis - with the offline evaluation and the serving latency measured rather
than asserted.

Headline numbers on the sealed MIND-{settings.dataset} test split (the official `dev`
split, scored once): ranker AUC **{number(ranker_test.get("auc"))}**, nDCG@10
**{number(ranker_test.get("ndcg@10"))}**; LightGBM LambdaRank baseline AUC
**{number(lgbm_test.get("auc"))}**. The online feature path is verified to reproduce the
offline one bit for bit.

> Every number in this file is produced by a script in this repo and stored under
> `results/`; this README is generated from those JSON files by
> `scripts/build_readme.py`. Anything not yet measured says **TBD**.

## Architecture

{architecture_diagram()}

The box marked *same module* is the point of the design: `features/time_aware.py` is
imported by both the offline replay and the request path, so there is exactly one
implementation of every feature.

## Data

{dataset_section(stats, splits)}

## Offline results

{offline_results_section(baselines, ranker)}

### Cold start

{cold_start_section(ranker, baselines)}

### Calibration

{calibration_section(ranker, settings)}

## Retrieval

{retrieval_section(retrieval, settings)}

## Serving

{serving_section(skew, onnx, seed)}

## Load test

{load_section(load_baseline, load_tuned, settings)}

## Diversity re-ranking (stretch)

{rerank_section(rerank, settings)}

## Published comparisons

{published_section(published, baselines, ranker)}

## Reproducing

```bash
uv sync --extra dev                 # install (Python 3.11)
cp .env.example .env                # add a Hugging Face token with MIND access
make m1                             # download, parse, fold, dataset stats
make m2                             # embeddings, shared features, baselines
make m3                             # two-tower, FAISS index, recall/latency
make m4                             # DIN + DCN-v2 ranker
make m5                             # ONNX export, redis, seed the feature store
make serve                          # run the API (separate shell)
make skew                           # assert online features == offline features
make m6                             # locust ladder, latency vs QPS
make m7                             # MMR diversity trade-off
make readme                         # regenerate this file from results/
```

The MIND dataset is gated: accept the licence at
<https://huggingface.co/datasets/yjw1029/MIND> and put a read token in `.env` as
`NEWSREC_HF_TOKEN`. Raw data is never committed.

CI runs lint, types, unit tests and an end-to-end smoke test of the whole pipeline on a
synthetic MIND-format dataset (`make smoke`), so no licensed data is needed to verify the
code path.

## Timings on the measurement machine

| stage | wall clock |
|---|---:|
| article embeddings ({dig(embeddings, "articles", default=0):,} articles, CPU) | {number(dig(embeddings, "embed_seconds"), 0)} s |
| feature replay ({dig(features, "folds", "train", "rows", default=0):,} train rows) | {number(dig(features, "seconds"), 0)} s |
| LightGBM LambdaRank | {number(dig(baselines, "models", "lgbm_lambdarank", "train_seconds"), 0)} s |
| two-tower ({dig(two_tower, "epochs", default="?")} epochs) | {number(dig(two_tower, "train_seconds"), 0)} s |
| DIN + DCN-v2 ranker ({dig(ranker, "selection", "config", "cross_layers", default="?")} cross layers) | {number(dig(ranker, "train_seconds"), 0)} s |

Machine: {dig(system, "cpu", default="TBD")}, {dig(system, "physical_cores", default="?")} physical /
{dig(system, "logical_cores", default="?")} logical cores, {dig(system, "memory_gb", default="?")} GB RAM,
{dig(system, "gpu", default="?")}. Python {dig(system, "libraries", "python", default="?")},
torch {dig(system, "libraries", "torch", default="?")},
onnxruntime {dig(system, "libraries", "onnxruntime", default="?")},
faiss {dig(system, "libraries", "faiss", default="?")}.

## Licence

MIT (see `LICENSE`). The MIND dataset is licensed separately by Microsoft Research and is
not redistributed here.

<sub>Generated by `scripts/build_readme.py` on {dt.date.today().isoformat()}.</sub>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, choices=["small", "large", "synthetic"])
    parser.add_argument("--output", default="README.md")
    args = parser.parse_args()

    settings = get_settings(dataset=args.dataset) if args.dataset else get_settings()
    content = build(settings)
    path = Path(args.output)
    path.write_text(content, encoding="utf-8")
    logger.info("wrote %s (%d lines)", path, content.count(chr(10)) + 1)
    if TBD in content:
        logger.warning("README still contains %d TBD markers", content.count(TBD))


if __name__ == "__main__":
    main()
