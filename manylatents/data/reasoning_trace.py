"""DataModule for loading reasoning trace hidden states.

Loads pre-extracted hidden states from LLM reasoning traces (OLMo, Claude, etc.)
and exposes them as standard (N, D) tensors with trace IDs as labels.

Two input modes:
    1. From TraceStore (npz files saved by manyagents):
        dm = ReasoningTraceDataModule(trace_store_path="outputs/traces/olmo_run")

    2. From live hidden states (passed directly by LocalLLMAdapter):
        dm = ReasoningTraceDataModule(
            hidden_states=[arr1, arr2, ...],  # list of (n_steps_i, D) arrays
            trace_ids=["trace_0", "trace_1", ...],
        )

In both cases, the module flattens all traces into (N_total, D) where
N_total = sum of steps across all traces. Trace IDs are exposed as integer
labels so any existing LatentModule (PCA, PHATE, diffusion maps) works
unchanged, and metrics/callbacks can group by trace via the labels.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from .precomputed_dataset import InMemoryDataset

logger = logging.getLogger(__name__)


class ReasoningTraceDataModule(LightningDataModule):
    """DataModule that unpacks reasoning traces into the manylatents pipeline.

    Handles the conversion from per-trace hidden state arrays (variable-length
    sequences) to the flat (N, D) format expected by LatentModules. Trace
    membership is preserved as integer labels.

    Attributes:
        trace_ids: Original trace ID strings, one per trace.
        step_trace_ids: Integer label per step indicating which trace it belongs to.
        steps_per_trace: Number of steps in each trace.
    """

    def __init__(
        self,
        trace_store_path: Optional[str] = None,
        hidden_states: Optional[List[np.ndarray]] = None,
        trace_ids: Optional[List[str]] = None,
        tensor_key: str = "pooled_steps",
        layer_index: int = -1,
        batch_size: int = 128,
        num_workers: int = 0,
        mode: str = "full",
        seed: int = 42,
    ):
        """
        Args:
            trace_store_path: Path to TraceStore directory (contains traces.jsonl + tensors/).
            hidden_states: List of arrays, each (n_steps_i, [n_layers,] D). Alternative to path.
            trace_ids: Trace ID strings matching hidden_states. Optional.
            tensor_key: Key in npz files to load ("pooled_steps" or "token_level").
            layer_index: Which layer to use when hidden states have a layer dimension.
                -1 for last layer.
            batch_size: DataLoader batch size.
            num_workers: DataLoader workers.
            mode: "full" (train=test=all) or "split".
            seed: Random seed for splits.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["hidden_states"])

        if trace_store_path is None and hidden_states is None:
            raise ValueError("Provide either trace_store_path or hidden_states.")

        self._raw_hidden_states = hidden_states
        self._raw_trace_ids = trace_ids

        # Set after setup()
        self.trace_ids: List[str] = []
        self.step_trace_ids: Optional[np.ndarray] = None
        self.steps_per_trace: List[int] = []
        self.train_dataset = None
        self.test_dataset = None

    def setup(self, stage: str = None):
        if self._raw_hidden_states is not None:
            data, labels = self._from_hidden_states(
                self._raw_hidden_states, self._raw_trace_ids
            )
        else:
            data, labels = self._from_trace_store(
                self.hparams.trace_store_path,
                self.hparams.tensor_key,
                self.hparams.layer_index,
            )

        self.step_trace_ids = labels
        data_tensor = torch.from_numpy(data).float()
        label_tensor = torch.from_numpy(labels).long()

        self.train_dataset = InMemoryDataset(data_tensor, label_tensor)
        self.test_dataset = self.train_dataset

        logger.info(
            f"ReasoningTraceDataModule: {len(self.trace_ids)} traces, "
            f"{data.shape[0]} total steps, D={data.shape[1]}"
        )

    def _from_hidden_states(
        self, hidden_states: List[np.ndarray], trace_ids: Optional[List[str]]
    ) -> tuple:
        """Build flat arrays from live hidden state list."""
        if trace_ids is None:
            trace_ids = [f"trace_{i}" for i in range(len(hidden_states))]
        self.trace_ids = trace_ids

        all_steps = []
        all_labels = []

        for i, hs in enumerate(hidden_states):
            hs = np.asarray(hs)
            # If 3D (n_steps, n_layers, D), select layer
            if hs.ndim == 3:
                hs = hs[:, self.hparams.layer_index, :]
            self.steps_per_trace.append(hs.shape[0])
            all_steps.append(hs)
            all_labels.append(np.full(hs.shape[0], i, dtype=np.int64))

        data = np.concatenate(all_steps, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        return data, labels

    def _from_trace_store(
        self, store_path: str, tensor_key: str, layer_index: int
    ) -> tuple:
        """Load hidden states from TraceStore npz files."""
        store_dir = Path(store_path)
        tensors_dir = store_dir / "tensors"

        if not tensors_dir.exists():
            raise FileNotFoundError(
                f"No tensors/ directory in {store_dir}. "
                "Was the TraceStore created with hidden_states?"
            )

        npz_files = sorted(tensors_dir.glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files in {tensors_dir}")

        hidden_states = []
        trace_ids = []

        for npz_path in npz_files:
            data = np.load(npz_path)
            if tensor_key not in data:
                available = list(data.keys())
                raise KeyError(
                    f"Key '{tensor_key}' not in {npz_path.name}. Available: {available}"
                )

            hs = data[tensor_key]  # (n_steps, [n_layers,] D)
            hidden_states.append(hs)
            trace_ids.append(npz_path.stem)

        return self._from_hidden_states(hidden_states, trace_ids)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,  # preserve trace order
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def get_labels(self) -> Optional[np.ndarray]:
        """Return per-step trace IDs as integer labels."""
        return self.step_trace_ids
