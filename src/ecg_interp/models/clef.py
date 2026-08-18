"""Wrapper around Nokia Bell Labs' CLEF (the Net1D architecture, single-lead).

CLEF isn't published as a pip package, so scripts/setup_clef.sh clones its source into
external/ecg-foundation-model and this wrapper imports from there.

Important caveat for cross-model comparisons (see docs/models.md): CLEF-medium and
ECGFounder both build the exact same Net1D configuration (identical filter_list/
m_blocks_list at every stage) — they are the same backbone architecture trained on
different data/objectives, not two independently designed architectures. Any shared
representation found between them is evidence about shared training data/objective effects,
not architecture-independence. A third, architecturally distinct model (e.g. a
transformer-based one) is needed before "shared across architectures" is a supportable claim.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import scipy.signal
import torch

from ecg_interp.models.base import ECGModel

_CLEF_REPO = Path(__file__).resolve().parents[3] / "external" / "ecg-foundation-model"
if str(_CLEF_REPO) not in sys.path:
    sys.path.insert(0, str(_CLEF_REPO))


class CLEF(ECGModel):
    """Single-lead CLEF (Net1D + SE blocks), small/medium/large. Preprocessing and hook
    points verified against the original repo's CLEF.py / contrastive_dataloader.py — see
    docs/models.md."""

    INPUT_LENGTH = 5000  # samples
    SAMPLING_RATE = 500  # Hz

    def __init__(self, size: str = "medium"):
        if size not in ("small", "medium", "large"):
            raise ValueError("CLEF ships small/medium/large checkpoints")
        self.size = size
        self.model = None  # built in load(); CLEF's own factory conflates instantiate+load

    def load(self, weights_path: str) -> None:
        try:
            from clef.baselines.models.CLEF import create_net1d_by_size
        except ImportError as e:
            raise ImportError(
                "CLEF source not found. Run scripts/setup_clef.sh first to clone "
                "github.com/Nokia-Bell-Labs/ecg-foundation-model into "
                "external/ecg-foundation-model."
            ) from e
        # n_classes is irrelevant here: create_net1d_by_size replaces the head with
        # nn.Identity, so forward() returns the pooled backbone feature, not logits.
        self.model = create_net1d_by_size(
            device=torch.device("cpu"),
            model_size=self.size,
            n_classes=4,
            linear_prob=False,
            pth=weights_path,
            in_channels=1,
        )
        self.model.eval()

    def preprocess(self, signal: np.ndarray, lead: int = 0) -> torch.Tensor:
        """`signal`: (samples, leads) in standard clinical lead order, as PTB-XL stores it.
        CLEF takes a single lead — Lead I (index 0) by default, chosen to match
        ECGFounder's 1-lead checkpoint (trained specifically on Lead I) so both models can
        be fed the literal same channel for a fair comparison; CLEF itself is roughly
        lead-invariant since pretraining randomly picked a lead per example.

        Resamples to 5000 samples (500Hz, 10s), applies a 0.67-40Hz Butterworth bandpass,
        then a per-sample z-score — CLEF's own pretraining path
        (clef/data/contrastive_dataloader.py), not its PTB-XL downstream loader (which uses
        sklearn StandardScaler instead); the pretraining path is the more faithful choice
        for representation extraction.
        """
        x = np.asarray(signal[:, lead], dtype=np.float64)
        if len(x) != self.INPUT_LENGTH:
            x_old = np.linspace(0, 1, len(x))
            x_new = np.linspace(0, 1, self.INPUT_LENGTH)
            x = np.interp(x_new, x_old, x)
        sos = scipy.signal.butter(4, [0.67, 40], btype="bandpass", fs=self.SAMPLING_RATE, output="sos")
        x = scipy.signal.sosfiltfilt(sos, x)
        x = (x - x.mean()) / (x.std() + 1e-8)
        return torch.from_numpy(x.copy()).float().view(1, 1, -1)  # (1, 1, 5000)

    @property
    def layer_names(self) -> List[str]:
        n_stages = {"small": 6, "medium": 7, "large": 9}[self.size]
        return ["first_conv"] + [f"stage_list.{i}" for i in range(n_stages)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
