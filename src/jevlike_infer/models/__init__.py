"""Model registry. Each entry maps a model name to the module that builds it.

A model module exposes ``BACKENDS`` (tuple of backend names, first is the default) and
``load(settings) -> DecisionModel``. Modules are imported lazily so that listing models, running the
unit tests or serving one model does not import the heavy dependencies of the others.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from jevlike_infer.config import Settings
from jevlike_infer.models.base import DecisionModel

MODELS: dict[str, str] = {
    "clef-flash": "jevlike_infer.models.clef_flash",
}


def model_module(name: str) -> ModuleType:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; supported: {', '.join(sorted(MODELS))}")
    return importlib.import_module(MODELS[name])


def load_model(settings: Settings) -> DecisionModel:
    return model_module(settings.model).load(settings)
