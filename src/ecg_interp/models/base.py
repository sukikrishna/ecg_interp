"""Common interface for pretrained ECG model wrappers, so representation-extraction and
analysis code doesn't need to know which underlying model it's talking to."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import torch


class ECGModel(ABC):
    """Wraps a pretrained ECG foundation model behind a common interface."""

    @abstractmethod
    def load(self, weights_path: str) -> None:
        """Load pretrained weights from disk."""

    @abstractmethod
    def preprocess(self, signal) -> torch.Tensor:
        """Turn a raw waveform (as loaded from PTB-XL) into the model's expected input tensor."""

    @property
    @abstractmethod
    def layer_names(self) -> List[str]:
        """Names (as in `model.named_modules()`) of the layers to hook for representation
        extraction — early/mid/late depth, chosen per model."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model forward. Combine with `ecg_interp.representations.extract` to also
        capture intermediate activations at `layer_names` via forward hooks."""
