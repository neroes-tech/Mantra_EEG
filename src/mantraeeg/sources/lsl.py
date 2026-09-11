"""Aquisição via LSL (UnicornLSL). Último recurso (SPEC 1.3).

**Estado:** o UnicornLSL não está instalado neste sistema — o Suite instalou só
o launcher, a ``Unicorn.dll`` e a camada de licenciamento. Este caminho fica
implementado mas indisponível até o UnicornLSL existir. O lançamento e vigilância
do UnicornLSL como subprocesso pertence ao M4 (SPEC 1.3), não a este módulo:
aqui só se resolve e lê um stream que já exista.

O stream do UnicornLSL publica os 17 canais do dispositivo na ordem canónica.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import N_CANONICAL_CHANNELS, DeviceInfo, EegSource, SourceError, empty_chunk

#: Amostras pedidas por chamada a ``pull_chunk``.
_MAX_SAMPLES_PER_PULL = 2048


class LSLSource(EegSource):
    """Unicorn através de um stream LSL publicado pelo UnicornLSL."""

    def __init__(self, cfg: Mapping[str, Any], sfreq_nominal: float) -> None:
        super().__init__()
        self._cfg = dict(cfg)
        self._sfreq_nominal = float(sfreq_nominal)
        self._inlet: Any = None

    def open(self) -> DeviceInfo:
        try:
            from pylsl import StreamInlet, resolve_byprop
        except ImportError as exc:  # pragma: no cover
            raise SourceError("pylsl não está instalado. `pip install pylsl`.") from exc

        name = str(self._cfg.get("stream_name", "UnicornLSL"))
        stream_type = str(self._cfg.get("stream_type", "EEG"))
        timeout = float(self._cfg.get("resolve_timeout_s", 10.0))

        streams = resolve_byprop("name", name, timeout=timeout)
        matched_by = "name"
        if not streams:
            streams = resolve_byprop("type", stream_type, timeout=timeout)
            matched_by = "type"
        if not streams:
            raise SourceError(
                f"nenhum stream LSL encontrado (nome {name!r} nem tipo "
                f"{stream_type!r}) em {timeout:.0f} s.\n"
                f"  - o UnicornLSL está a correr e com o dispositivo ligado?"
            )

        inlet = StreamInlet(streams[0], max_buflen=60)
        sinfo = inlet.info()
        n_ch = int(sinfo.channel_count())
        if n_ch != N_CANONICAL_CHANNELS:
            inlet.close_stream()
            raise SourceError(
                f"o stream LSL tem {n_ch} canais, esperados "
                f"{N_CANONICAL_CHANNELS} (SPEC 1.1)"
            )
        sfreq = float(sinfo.nominal_srate())
        if sfreq > 0 and abs(sfreq - self._sfreq_nominal) > 1e-6:
            inlet.close_stream()
            raise SourceError(
                f"o stream LSL reporta {sfreq} Hz, a config diz "
                f"{self._sfreq_nominal} Hz"
            )

        self._inlet = inlet
        self._info = DeviceInfo(
            source_name="lsl",
            sfreq_nominal=self._sfreq_nominal,
            n_channels=N_CANONICAL_CHANNELS,
            device_id=str(sinfo.source_id() or sinfo.name()),
            units_to_uv=1.0,
            extra={
                "stream_name": str(sinfo.name()),
                "stream_type": str(sinfo.type()),
                "matched_by": matched_by,
                "hostname": str(sinfo.hostname()),
            },
        )
        return self._info

    def start(self) -> None:
        if self._inlet is None:
            raise SourceError("start() antes de open()")
        self._inlet.flush()
        self._started = True

    def _read_raw(self) -> np.ndarray:
        if self._inlet is None or not self._started:
            return empty_chunk()
        samples, _timestamps = self._inlet.pull_chunk(
            timeout=0.0, max_samples=_MAX_SAMPLES_PER_PULL
        )
        if not samples:
            return empty_chunk()
        return np.asarray(samples, dtype=np.float32).T

    def stop(self) -> None:
        self._started = False

    def close(self) -> None:
        if self._inlet is not None:
            try:
                self._inlet.close_stream()
            except Exception:
                pass
        self._inlet = None
