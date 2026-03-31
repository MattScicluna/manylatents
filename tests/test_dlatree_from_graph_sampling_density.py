"""Tests for DLATreeFromGraph sampling density controls."""

import numpy as np

from manylatents.data.synthetic_dataset import DLATreeFromGraph


def test_dlatree_from_graph_sampling_density_factors_adjust_counts():
    """sampling_density_factors should change per-edge sample counts post-generation."""
    graph_edges = [
        [1, 2, 1, 100],
        [2, 3, 2, 100],
        [2, 4, 3, 100],
        [4, 5, 4, 100],
    ]

    dataset = DLATreeFromGraph(
        graph_edges=graph_edges,
        excluded_edges=[],
        n_dim=8,
        rand_multiplier=2.0,
        random_state=42,
        sigma=0.1,
        save_graph_viz=False,
        sampling_density_factors={1: 3.0, 2: 0.5},
    )

    labels, counts = np.unique(dataset.metadata, return_counts=True)
    counts_by_label = {int(label): int(count) for label, count in zip(labels, counts)}

    assert counts_by_label == {1: 300, 2: 50, 3: 100, 4: 100}

    dists = dataset.get_gt_dists()
    assert dists.shape == (550, 550)
