"""Forward-hook based activation extraction — model-agnostic given a list of layer names."""
from __future__ import annotations

from typing import Dict, List

import torch


class ActivationExtractor:
    """Registers forward hooks on the given layer names and collects their outputs.

    Usage:
        with ActivationExtractor(model, ["layer1", "layer2.0"]) as extractor:
            model(x)
        activations = extractor.activations  # {name: tensor}
    """

    def __init__(self, model: torch.nn.Module, layer_names: List[str]):
        self.model = model
        self.layer_names = layer_names
        self.activations: Dict[str, torch.Tensor] = {}
        self._handles = []

    def __enter__(self) -> "ActivationExtractor":
        modules = dict(self.model.named_modules())
        for name in self.layer_names:
            if name not in modules:
                raise KeyError(
                    f"'{name}' is not a module of this model. "
                    f"Available names include: {sorted(modules)[:20]}"
                )
            handle = modules[name].register_forward_hook(self._make_hook(name))
            self._handles.append(handle)
        return self

    def __exit__(self, *exc_info) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, name: str):
        def hook(module, inputs, output):
            self.activations[name] = output.detach().cpu()

        return hook
