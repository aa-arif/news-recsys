"""End-to-end serving test: artifacts on disk, ONNX sessions, FAISS, Redis, HTTP.

The models here are untrained - random weights are fine, because what is under test is
the *path*: that the artifacts load, that the ONNX signatures match what the pipeline
feeds them, that the candidate set survives the trip through FAISS and Redis, and that
the API returns a well-formed, correctly ordered response with per-stage timings.

Quality of the recommendations is measured offline by the evaluation scripts, not here.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from news_recsys.config import Settings
from news_recsys.features.build import load_snapshot
from news_recsys.features.text import load_embeddings
from news_recsys.features.vocab import load_vocabulary
from news_recsys.models.onnx_export import export_ranker, export_user_tower
from news_recsys.models.ranker import DinDcnRanker, RankerConfig
from news_recsys.models.two_tower import TwoTowerConfig, TwoTowerModel
from news_recsys.models.two_tower_train import SequenceBatcher, encode_all_items
from news_recsys.retrieval.faiss_index import ItemIndex
from news_recsys.serving.pipeline import STAGES, RecommendationPipeline
from news_recsys.serving.seed import seed_counters, seed_histories
from news_recsys.serving.state import ServingArtifacts, missing_artifacts

fakeredis = pytest.importorskip("fakeredis")


@pytest.fixture(scope="module")
def serving_stack(synthetic_artifacts: Settings):
    """Every artifact the server loads, built small and fast."""
    settings = synthetic_artifacts
    vocabulary = load_vocabulary(settings)
    embeddings = np.asarray(load_embeddings(settings, mmap=False), dtype=np.float32)

    two_tower = TwoTowerModel(
        TwoTowerConfig(text_dim=embeddings.shape[1], output_dim=32, hidden_dim=64),
        vocabulary.n_categories,
        vocabulary.n_subcategories,
    ).eval()
    two_tower.save(settings.artifact_dir, vocabulary, settings)

    batcher = SequenceBatcher(embeddings, vocabulary, np.zeros(vocabulary.n_news))
    item_vectors = encode_all_items(two_tower, batcher, vocabulary.n_news)
    np.save(settings.artifact_dir / "item_vectors.npy", item_vectors)
    ItemIndex.build(item_vectors, m=16, ef_construction=64).save(settings.artifact_dir)

    ranker = DinDcnRanker(
        RankerConfig(
            dense_dim=35,
            text_dim=embeddings.shape[1],
            item_dim=settings.ranker_item_dim,
            attention_dim=settings.ranker_attention_dim,
            n_categories=vocabulary.n_categories,
            n_subcategories=vocabulary.n_subcategories,
        )
    ).eval()
    ranker.save(settings.artifact_dir)

    export_user_tower(
        two_tower, settings.artifact_dir, settings, history_length=settings.max_history
    )
    export_ranker(
        ranker, settings.artifact_dir, settings, history_length=settings.ranker_max_history
    )

    store, _ = load_snapshot(settings, vocabulary=vocabulary)
    client = fakeredis.FakeStrictRedis()
    seed_counters(client, store, vocabulary, settings)
    seed_histories(client, settings, fold="test")

    artifacts = ServingArtifacts.load(settings)
    return settings, artifacts, client


def a_seeded_user(settings: Settings) -> str:
    users = (settings.artifact_dir / "serving_users.txt").read_text(encoding="utf-8").splitlines()
    return users[0]


def test_every_serving_artifact_is_present(serving_stack) -> None:
    settings, _, _ = serving_stack
    assert missing_artifacts(settings) == []


def test_recommend_returns_ranked_items_with_stage_timings(serving_stack) -> None:
    settings, artifacts, client = serving_stack
    pipeline = RecommendationPipeline(artifacts, client)
    result = pipeline.recommend(a_seeded_user(settings), k=5)

    assert len(result.items) == 5
    scores = [item.score for item in result.items]
    assert scores == sorted(scores, reverse=True)
    assert {item.news_id for item in result.items} == {item.news_id for item in result.items}
    assert set(result.timings_ms) == set(STAGES)
    assert result.total_ms >= sum(result.timings_ms.values()) * 0.9
    assert 0 < result.candidates <= settings.retrieval_candidates
    assert all(0.0 <= item.probability <= 1.0 for item in result.items)


def test_unknown_user_still_gets_recommendations(serving_stack) -> None:
    _, artifacts, client = serving_stack
    pipeline = RecommendationPipeline(artifacts, client)
    result = pipeline.recommend("U-never-seen", k=3)
    assert len(result.items) == 3
    assert result.history_length == 0


def test_user_embedding_cache_hits_on_repeat_requests(serving_stack) -> None:
    settings, artifacts, client = serving_stack
    cached_settings = settings.model_copy(update={"user_embedding_cache_size": 128})
    artifacts_with_cache = ServingArtifacts(**{**artifacts.__dict__, "settings": cached_settings})
    pipeline = RecommendationPipeline(artifacts_with_cache, client)

    user_id = a_seeded_user(settings)
    first = pipeline.recommend(user_id, k=5)
    second = pipeline.recommend(user_id, k=5)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert [item.news_id for item in first.items] == [item.news_id for item in second.items]
    assert pipeline.cache.stats()["hits"] == 1


def test_ef_search_is_honoured_per_request(serving_stack) -> None:
    _, artifacts, client = serving_stack
    pipeline = RecommendationPipeline(artifacts, client)
    pipeline.recommend(a_seeded_user(artifacts.settings), k=5, ef_search=200)
    assert artifacts.index.ef_search == 200


def test_http_api_serves_recommendations(serving_stack, monkeypatch: pytest.MonkeyPatch) -> None:
    settings, artifacts, client = serving_stack
    from news_recsys.serving import app as app_module

    pipeline = RecommendationPipeline(artifacts, client)
    # The app's lifespan builds a pipeline from the *ambient* settings and a real Redis;
    # point it at the fixture's stack instead so the HTTP layer is what is under test.
    monkeypatch.setattr(app_module, "build_pipeline", lambda _settings: pipeline)
    monkeypatch.setattr(app_module, "get_settings", lambda: settings)

    with TestClient(app_module.app) as http:
        health = http.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        user_id = a_seeded_user(settings)
        response = http.get("/recommend", params={"user_id": user_id, "k": 4})
        assert response.status_code == 200
        payload = response.json()
        assert len(payload["items"]) == 4
        assert payload["user_id"] == user_id
        assert set(payload["timings_ms"]) == set(STAGES)

        features = http.get(
            "/features",
            params={"user_id": user_id, "news_ids": ",".join(artifacts.news_ids[:5])},
        )
        assert features.status_code == 200
        matrix = np.asarray(features.json()["features"], dtype=np.float32)
        assert matrix.shape == (5, 35)
