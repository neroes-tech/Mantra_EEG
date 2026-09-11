"""Fontes de aquisição.

``make_source(cfg)`` é o único ponto onde se escolhe o backend. Trocar de fonte
é alterar ``device.source`` no YAML e mais nada (critério de aceitação 11).
"""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from .base import (
    ACCEL_SLICE,
    BATTERY_ROW,
    CANONICAL_ROW_NAMES,
    COUNTER_ROW,
    EEG_SLICE,
    GYRO_SLICE,
    N_CANONICAL_CHANNELS,
    VALIDATION_ROW,
    DeviceInfo,
    EegSource,
    SourceError,
    plausibility_warning,
)

__all__ = [
    "ACCEL_SLICE",
    "BATTERY_ROW",
    "CANONICAL_ROW_NAMES",
    "COUNTER_ROW",
    "EEG_SLICE",
    "GYRO_SLICE",
    "N_CANONICAL_CHANNELS",
    "VALIDATION_ROW",
    "DeviceInfo",
    "EegSource",
    "SourceError",
    "plausibility_warning",
    "make_source",
]


def make_source(cfg: Config, override: str | None = None, **kwargs: object) -> EegSource:
    """Instancia a fonte indicada por ``device.source`` (ou por ``override``)."""
    name = override or cfg.device.source

    if name == "brainflow":
        from .brainflow_src import BrainflowSource

        return BrainflowSource(cfg.device.brainflow, cfg.device.sfreq_nominal)

    if name == "unicorn_dll":
        from .unicorn_dll import UnicornDllSource

        return UnicornDllSource(
            cfg.device.unicorn_dll,
            cfg.device.sfreq_nominal,
            cfg.device.resolved_dll_paths(),
        )

    if name == "lsl":
        from .lsl import LSLSource

        return LSLSource(cfg.device.lsl, cfg.device.sfreq_nominal)

    if name == "replay":
        from .replay import ReplaySource

        path = kwargs.get("path")
        return ReplaySource(
            cfg.device.replay,
            cfg.device.sfreq_nominal,
            path=Path(str(path)) if path else None,
        )

    if name == "synthetic":
        # Implementada no M1 (SPEC 6.2). A mensagem é explícita para não parecer
        # um bug de configuração.
        raise SourceError(
            "a fonte 'synthetic' é entregue no M1 (SPEC 6.2); use 'replay' para "
            "trabalhar sobre gravações reais entretanto"
        )

    raise SourceError(f"device.source desconhecido: {name!r}")
