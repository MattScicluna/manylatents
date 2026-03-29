"""Tests for API resolution without Hydra."""
import numpy as np
import pytest


def test_resolve_datamodule_by_name():
    """String name resolves via data registry, not Hydra."""
    from manylatents.api import _resolve_datamodule
    dm = _resolve_datamodule(data="swissroll")
    assert hasattr(dm, "setup")
    assert hasattr(dm, "train_dataloader")


def test_resolve_datamodule_from_array():
    """numpy array wraps in PrecomputedDataModule."""
    from manylatents.api import _resolve_datamodule
    arr = np.random.randn(50, 3).astype(np.float32)
    dm = _resolve_datamodule(input_data=arr)
    dm.setup()
    assert dm.train_dataset is not None


def test_resolve_algorithm_by_name():
    """String name resolves via algorithm registry."""
    from manylatents.api import _resolve_algorithm
    from manylatents.algorithms.latent.latent_module_base import LatentModule
    algo = _resolve_algorithm(algorithm="pca")
    assert isinstance(algo, LatentModule)


def test_resolve_algorithm_instance_passthrough():
    """Pre-built instance passes through unchanged."""
    from manylatents.api import _resolve_algorithm
    from manylatents.algorithms.latent.pca import PCAModule
    mod = PCAModule(n_components=3)
    result = _resolve_algorithm(algorithm=mod)
    assert result is mod


def test_resolve_algorithm_dict_config():
    """Dict with _target_ still works (Hydra fallback)."""
    from manylatents.api import _resolve_algorithm
    algo = _resolve_algorithm(algorithms={
        "latent": {
            "_target_": "manylatents.algorithms.latent.pca.PCAModule",
            "n_components": 5,
        }
    })
    assert algo.n_components == 5


def test_resolve_algorithm_dict_string():
    """Dict with string value uses registry."""
    from manylatents.api import _resolve_algorithm
    from manylatents.algorithms.latent.pca import PCAModule
    algo = _resolve_algorithm(algorithms={"latent": "pca"})
    assert isinstance(algo, PCAModule)


def test_api_run_string_names():
    """Full run() with string names."""
    from manylatents.api import run
    result = run(
        data="swissroll",
        algorithm="pca",
        metrics=["FractalDimension"],
    )
    assert "embeddings" in result
    assert "scores" in result
    assert "FractalDimension" in result["scores"]
