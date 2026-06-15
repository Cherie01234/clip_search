"""Tests for the FAISS indexer (no CLIP model / no images needed -> CI-safe)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from indexer import FAISSIndexer  # noqa: E402


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def test_dim_mismatch_raises():
    idx = FAISSIndexer(dim=4)
    with pytest.raises(ValueError):
        idx.add(np.zeros((2, 3), dtype=np.float32), [{}, {}])


def test_search_ranks_nearest_first_with_cosine():
    idx = FAISSIndexer(dim=8)
    rng = np.random.default_rng(0)
    vecs = _unit(rng.normal(size=(10, 8)))
    idx.add(vecs, [{"filename": f"{i}.jpg"} for i in range(10)])

    res = idx.search(vecs[3:4], top_k=3)
    assert res[0]["filename"] == "3.jpg"              # nearest is the query itself
    assert res[0]["score"] == pytest.approx(1.0, abs=1e-4)  # cosine == 1
    assert res[0]["score"] >= res[1]["score"] >= res[2]["score"]  # descending


def test_total_and_save_load_roundtrip(tmp_path):
    idx = FAISSIndexer(dim=4)
    vecs = _unit(np.eye(4))
    idx.add(vecs, [{"filename": str(i)} for i in range(4)])
    assert idx.total == 4

    idx.save(str(tmp_path))
    reloaded = FAISSIndexer(dim=4)
    reloaded.load(str(tmp_path))
    assert reloaded.total == 4
    assert reloaded.search(vecs[0:1], top_k=1)[0]["filename"] == "0"
