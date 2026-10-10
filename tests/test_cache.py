import json

import numpy as np
import pytest

from adaptive_rag.cache import (MODEL_FILES, ADAPTER_FILES, make_embedding_fingerprint,
                                load_embedding_cache, save_embedding_cache)
from adaptive_rag.data import Document


@pytest.fixture
def inputs(tmp_path):
    model = tmp_path / "model"
    adapter = tmp_path / "adapter"
    model.mkdir()
    adapter.mkdir()
    for name in MODEL_FILES:
        (model / name).write_bytes(("fake-" + name).encode())
    for name in ADAPTER_FILES:
        (adapter / name).write_bytes(("fake-" + name).encode())
    documents = [Document("d1", "First", "original text"), Document("d2", "Second", "other text")]
    queries = {"q1": "first query", "q2": "second query"}
    settings = dict(model_path=model, model_id="fixture/model", model_revision="pinned-revision",
                    adapter_path=adapter, max_length=192, dimension=3)
    fingerprint = make_embedding_fingerprint(documents, queries, ["q1", "q2"], **settings)
    vectors = np.eye(2, 3, dtype=np.float32)
    return documents, queries, settings, fingerprint, vectors


def test_cache_roundtrip_is_strict_and_object_free(tmp_path, inputs):
    documents, queries, settings, fingerprint, vectors = inputs
    path = tmp_path / "index.npz"
    assert load_embedding_cache(path, fingerprint) is None
    save_embedding_cache(path, vectors, vectors, ["d1", "d2"], ["q1", "q2"], fingerprint)
    actual_documents, actual_queries = load_embedding_cache(path, fingerprint)
    np.testing.assert_array_equal(actual_documents, vectors)
    np.testing.assert_array_equal(actual_queries, vectors)
    with np.load(path, allow_pickle=False) as arrays:
        assert str(arrays["adapter_sha256"]) == fingerprint["adapter"]["adapter_model.safetensors"]
        assert json.loads(str(arrays["fingerprint_json"])) == fingerprint
    assert not path.with_suffix(".npz.tmp").exists()


def test_fingerprint_changes_for_text_order_and_inference_settings(inputs):
    documents, queries, settings, fingerprint, _ = inputs
    changed = [Document("d1", "First", "edited text"), documents[1]]
    variants = [
        make_embedding_fingerprint(changed, queries, ["q1", "q2"], **settings),
        make_embedding_fingerprint(list(reversed(documents)), queries, ["q1", "q2"], **settings),
        make_embedding_fingerprint(documents, {**queries, "q1": "changed"}, ["q1", "q2"], **settings),
        make_embedding_fingerprint(documents, queries, ["q2", "q1"], **settings),
        make_embedding_fingerprint(documents, queries, ["q1", "q2"], **{**settings, "max_length": 256}),
        make_embedding_fingerprint(documents, queries, ["q1", "q2"], **{**settings, "model_revision": "new"}),
    ]
    assert all(value != fingerprint for value in variants)
    assert make_embedding_fingerprint(documents, queries, ["q1", "q2"], **settings) == fingerprint


def test_actual_model_tokenizer_and_adapter_bytes_are_fingerprinted(inputs):
    documents, queries, settings, fingerprint, _ = inputs
    for path in [settings["model_path"] / "model.safetensors", settings["model_path"] / "tokenizer.json",
                 settings["adapter_path"] / "adapter_config.json", settings["adapter_path"] / "adapter_model.safetensors"]:
        original = path.read_bytes()
        path.write_bytes(original + b"changed")
        assert make_embedding_fingerprint(documents, queries, ["q1", "q2"], **settings) != fingerprint
        path.write_bytes(original)


def test_changed_text_cannot_reuse_cache(tmp_path, inputs):
    documents, queries, settings, fingerprint, vectors = inputs
    path = tmp_path / "index.npz"
    save_embedding_cache(path, vectors, vectors, ["d1", "d2"], ["q1", "q2"], fingerprint)
    changed = make_embedding_fingerprint(documents, {**queries, "q1": "new query"}, ["q1", "q2"], **settings)
    with pytest.raises(ValueError, match="changed"):
        load_embedding_cache(path, changed)


def test_legacy_or_misordered_arrays_require_reindex(tmp_path, inputs):
    _, _, _, fingerprint, vectors = inputs
    path = tmp_path / "index.npz"
    np.savez_compressed(path, documents=vectors, queries=vectors)
    with pytest.raises(ValueError, match="legacy"):
        load_embedding_cache(path, fingerprint)
    np.savez_compressed(path, documents=vectors, queries=vectors, document_ids=["d2", "d1"],
                        query_ids=["q1", "q2"], fingerprint_json=json.dumps(fingerprint))
    with pytest.raises(ValueError, match="IDs/order"):
        load_embedding_cache(path, fingerprint)


def test_bad_shapes_or_nonfinite_vectors_cannot_be_saved(tmp_path, inputs):
    _, _, _, fingerprint, vectors = inputs
    with pytest.raises(ValueError, match="shape"):
        save_embedding_cache(tmp_path / "bad.npz", vectors[:, :2], vectors, ["d1", "d2"], ["q1", "q2"], fingerprint)
    bad = vectors.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        save_embedding_cache(tmp_path / "bad.npz", bad, vectors, ["d1", "d2"], ["q1", "q2"], fingerprint)
    assert not (tmp_path / "bad.npz").exists()


def test_ambiguous_newlines_and_unicode_have_distinct_fingerprints(inputs):
    documents, queries, settings, fingerprint, _ = inputs
    with pytest.raises(ValueError, match="unique"):
        make_embedding_fingerprint([documents[0], documents[0]], queries, **settings)
    with pytest.raises(ValueError, match="Missing"):
        make_embedding_fingerprint(documents, queries, ["unknown"], **settings)
    left = make_embedding_fingerprint([Document("x", "", "北京\n杭州")], queries, **settings)
    right = make_embedding_fingerprint([Document("x", "北京", "杭州")], queries, **settings)
    assert left == right  # same actual Document.content implies same model input
    assert left != fingerprint
