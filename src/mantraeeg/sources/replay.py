"""Reprodução de gravações. Puxada da SPEC (M1) para o M0 deliberadamente.

Uma fonte de replay sobre dados Unicorn **reais** permite construir e testar o
monitor de contacto, o recorder e o pipeline inteiro sem depender de o hardware
estar disponível naquele minuto — e é sobre ela que correm os testes
automáticos que envolvem sinal genuíno.

Formatos aceites:

``.csv``
    Saída do UnicornRecorder: 17 colunas, ``EEG 1..8``, ``Accelerometer X/Y/Z``,
    ``Gyroscope X/Y/Z``, ``Battery Level``, ``Counter``, ``Validation Indicator``
    — exatamente a ordem canónica da SPEC 1.1.

``.npy``
    ``(17, n)`` como gravado por ``recorder.py``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .base import (
    N_CANONICAL_CHANNELS,
    DeviceInfo,
    EegSource,
    SourceError,
    empty_chunk,
)

#: Ordem das colunas do UnicornRecorder. Coincide com o layout canónico.
UNICORN_CSV_COLUMNS: tuple[str, ...] = (
    "EEG 1", "EEG 2", "EEG 3", "EEG 4", "EEG 5", "EEG 6", "EEG 7", "EEG 8",
    "Accelerometer X", "Accelerometer Y", "Accelerometer Z",
    "Gyroscope X", "Gyroscope Y", "Gyroscope Z",
    "Battery Level", "Counter", "Validation Indicator",
)


def load_recording(path: str | Path) -> np.ndarray:
    """Carrega uma gravação para ``(17, n)`` float32, em ordem canónica."""
    path = Path(path)
    if not path.exists():
        raise SourceError(f"gravação não encontrada: {path}")

    if path.suffix.lower() == ".npy":
        data = np.load(path)
        if data.ndim != 2 or data.shape[0] != N_CANONICAL_CHANNELS:
            raise SourceError(
                f"{path}: esperado ({N_CANONICAL_CHANNELS}, n), obtido {data.shape}"
            )
        return np.ascontiguousarray(data, dtype=np.float32)

    if path.suffix.lower() != ".csv":
        raise SourceError(f"{path}: extensão não suportada (use .csv ou .npy)")

    header = path.read_text(encoding="utf-8-sig").split("\n", 1)[0].strip()
    names = [c.strip() for c in header.split(",")]
    if len(names) != len(UNICORN_CSV_COLUMNS):
        raise SourceError(
            f"{path}: {len(names)} colunas, esperadas "
            f"{len(UNICORN_CSV_COLUMNS)} (formato UnicornRecorder)"
        )
    if tuple(names) != UNICORN_CSV_COLUMNS:
        mismatch = [
            f"col {i}: {got!r} != {want!r}"
            for i, (got, want) in enumerate(zip(names, UNICORN_CSV_COLUMNS))
            if got != want
        ]
        raise SourceError(f"{path}: cabeçalho inesperado -> {mismatch[:3]}")

    table = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    if table.ndim == 1:
        table = table[None, :]
    return np.ascontiguousarray(table.T, dtype=np.float32)


class ReplaySource(EegSource):
    """Reproduz uma gravação como se fosse o dispositivo."""

    def __init__(
        self,
        cfg: Mapping[str, Any],
        sfreq_nominal: float,
        path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self._cfg = dict(cfg)
        self._sfreq_nominal = float(sfreq_nominal)
        raw_path = str(path or self._cfg.get("path", "") or "").strip()
        self._raw_path = raw_path
        self._path = Path(raw_path) if raw_path else None
        self._realtime = bool(self._cfg.get("realtime", False))
        self._data: np.ndarray = np.zeros((N_CANONICAL_CHANNELS, 0), dtype=np.float32)
        self._pos = 0
        self._t0 = 0.0

    def open(self) -> DeviceInfo:
        if self._path is None:
            raise SourceError(
                "device.replay.path está vazio; indique a gravação a reproduzir"
            )
        self._data = load_recording(self._path)
        self._pos = 0
        self._info = DeviceInfo(
            source_name="replay",
            sfreq_nominal=self._sfreq_nominal,
            n_channels=N_CANONICAL_CHANNELS,
            device_id=str(self._path.name),
            units_to_uv=1.0,
            extra={
                "path": str(self._path),
                "n_samples": int(self._data.shape[1]),
                "duration_s": self._data.shape[1] / self._sfreq_nominal,
                "realtime": self._realtime,
            },
        )
        return self._info

    def start(self) -> None:
        self._t0 = time.monotonic()
        self._pos = 0
        self._started = True

    def _read_raw(self) -> np.ndarray:
        if not self._started or self._pos >= self._data.shape[1]:
            return empty_chunk()

        if self._realtime:
            # Só as amostras que já "aconteceram" segundo o relógio.
            due = int((time.monotonic() - self._t0) * self._sfreq_nominal)
            end = min(due, self._data.shape[1])
        else:
            end = self._data.shape[1]

        if end <= self._pos:
            return empty_chunk()
        chunk = self._data[:, self._pos : end]
        self._pos = end
        return chunk

    @property
    def exhausted(self) -> bool:
        return self._pos >= self._data.shape[1]

    @property
    def n_samples(self) -> int:
        return int(self._data.shape[1])

    def stop(self) -> None:
        self._started = False

    def close(self) -> None:
        self._data = np.zeros((N_CANONICAL_CHANNELS, 0), dtype=np.float32)
        self._pos = 0
