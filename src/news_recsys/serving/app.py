"""FastAPI app exposing the two-stage recommender.

``GET /recommend?user_id=U123&k=10`` runs the full path and returns the ranked articles
plus the per-stage timing breakdown for that request.

The data is from November 2019, so "now" is a *simulated* clock: by default the server
answers as of the feature-store snapshot timestamp (the start of the test day). Passing
``as_of`` moves the clock, which is what the skew test and the load test use. Wiring the
wall clock in would make every article 7 years old and every time-aware feature useless.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import anyio.to_thread
import numpy as np
import redis
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from news_recsys.config import Settings, get_settings
from news_recsys.logging_utils import get_logger
from news_recsys.serving.pipeline import STAGES, RecommendationPipeline
from news_recsys.serving.state import ServingArtifacts, missing_artifacts

logger = get_logger("serving.app")

_state: dict[str, Any] = {}


class RecommendationModel(BaseModel):
    news_id: str
    title: str
    category: str
    score: float
    probability: float = Field(description="calibrated click probability")
    retrieval_score: float
    is_cold: bool


class RecommendResponse(BaseModel):
    user_id: str
    k: int
    as_of: float
    history_length: int
    candidates: int
    items: list[RecommendationModel]
    timings_ms: dict[str, float]
    total_ms: float
    cache_hit: bool
    sources: dict[str, Any] = Field(default_factory=dict)


def build_pipeline(settings: Settings) -> RecommendationPipeline:
    missing = missing_artifacts(settings)
    if missing:
        raise RuntimeError(
            f"missing serving artifacts: {missing}. Run the pipeline (make m2 m3 m4 onnx) first."
        )
    artifacts = ServingArtifacts.load(settings)
    client = redis.from_url(settings.redis_url)
    return RecommendationPipeline(artifacts, client)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    _state["settings"] = settings
    # Sync endpoints run in anyio's thread pool; its 40-thread default oversubscribes a
    # 12-thread box badly once every request wants the GIL and 1-2 ORT threads.
    anyio.to_thread.current_default_thread_limiter().total_tokens = settings.serve_threadpool_size
    _state["pipeline"] = build_pipeline(settings)
    logger.info("serving %s", _state["pipeline"].artifacts.describe())
    yield
    _state.clear()


app = FastAPI(
    title="news-recsys",
    version="0.1.0",
    summary="Two-stage news recommender: two-tower retrieval + DIN/DCN-v2 ranking",
    lifespan=lifespan,
)


def get_pipeline() -> RecommendationPipeline:
    pipeline = _state.get("pipeline")
    if pipeline is None:  # pragma: no cover - only when startup failed
        raise HTTPException(status_code=503, detail="pipeline not initialised")
    return pipeline


@app.get("/health")
def health(pipeline: RecommendationPipeline = Depends(get_pipeline)) -> dict[str, Any]:
    try:
        pinged = bool(pipeline.store.client.ping())
    except Exception as error:  # pragma: no cover - exercised only when redis is down
        raise HTTPException(status_code=503, detail=f"redis unavailable: {error}") from error
    return {"status": "ok", "redis": pinged, **pipeline.artifacts.describe()}


@app.get("/stats")
def stats(pipeline: RecommendationPipeline = Depends(get_pipeline)) -> dict[str, Any]:
    return {
        "user_embedding_cache": pipeline.cache.stats(),
        "stages": list(STAGES),
        **pipeline.artifacts.describe(),
    }


@app.get("/recommend", response_model=RecommendResponse)
def recommend(
    user_id: str = Query(..., description="MIND user id, e.g. U13740"),
    k: int = Query(default=10, ge=1, le=100),
    as_of: float | None = Query(default=None, description="simulated clock, POSIX seconds"),
    ef_search: int | None = Query(default=None, ge=1, le=2048),
    candidates: int | None = Query(default=None, ge=1, le=2000),
    mmr_lambda: float = Query(
        default=1.0, ge=0.0, le=1.0, description="1.0 = pure relevance; lower diversifies"
    ),
    popularity_share: float | None = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description="fraction of the candidate budget taken from the trending list",
    ),
    pipeline: RecommendationPipeline = Depends(get_pipeline),
) -> RecommendResponse:
    result = pipeline.recommend(
        user_id,
        k,
        now=as_of,
        ef_search=ef_search,
        n_candidates=candidates,
        mmr_lambda=mmr_lambda,
        popularity_share=popularity_share,
    )
    return RecommendResponse(
        user_id=result.user_id,
        k=result.k,
        as_of=result.as_of,
        history_length=result.history_length,
        candidates=result.candidates,
        items=[RecommendationModel(**item.__dict__) for item in result.items],
        timings_ms={name: round(value, 3) for name, value in result.timings_ms.items()},
        total_ms=round(result.total_ms, 3),
        cache_hit=result.cache_hit,
        sources=result.diagnostics,
    )


@app.get("/features")
def features(
    user_id: str = Query(...),
    news_ids: str = Query(..., description="comma-separated article ids"),
    as_of: float | None = Query(default=None),
    pipeline: RecommendationPipeline = Depends(get_pipeline),
) -> dict[str, Any]:
    """Online features for an explicit candidate set.

    This endpoint exists for the training/serving skew test: it lets a test ask the
    *server* for the same feature rows the offline pipeline computed, and compare them
    bit for bit.
    """
    ids = [item.strip() for item in news_ids.split(",") if item.strip()]
    indices = pipeline.artifacts.vocabulary.news_indices(ids)
    now = as_of if as_of is not None else pipeline.artifacts.snapshot_as_of
    matrix = pipeline.feature_matrix(user_id, np.asarray(indices, dtype=np.int64), now)
    return {
        "user_id": user_id,
        "as_of": now,
        "news_ids": ids,
        "features": matrix.tolist(),
    }


def main() -> None:  # pragma: no cover - entry point for `python -m`
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "news_recsys.serving.app:app",
        host=settings.serve_host,
        port=int(os.environ.get("NEWSREC_SERVE_PORT", settings.serve_port)),
        workers=1,
        log_level="warning",
    )


if __name__ == "__main__":  # pragma: no cover
    main()
