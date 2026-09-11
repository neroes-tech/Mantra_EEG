"""Gravação da sessão para disco.

Escreve em *streaming* para ``raw.f32`` (binário cru) e só converte para
``raw.npy`` no fim. É deliberado: numa banca de festival, uma aplicação que
rebente a meio deixa na mesma um ficheiro recuperável, o que não aconteceria com
um ``.npy`` cujo cabeçalho só se escreve no fecho.

Sem dados pessoais em nomes de ficheiro — só um ID sequencial (SPEC 6.3).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .sources.base import CANONICAL_ROW_NAMES, N_CANONICAL_CHANNELS, DeviceInfo

RAW_STREAM_NAME = "raw.f32"
RAW_ARRAY_NAME = "raw.npy"
RAW_META_NAME = "raw_meta.json"
EVENTS_NAME = "events.json"


def _git_hash() -> str:
    """Hash do commit atual, ou ``"(sem git)"``. Vai para o raw_meta."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() or "(sem git)"
    except Exception:
        return "(sem git)"


def _code_version() -> str:
    try:
        from importlib.metadata import version

        return version("mantra-eeg")
    except Exception:
        return "0.0.0+dev"


def session_folder_name(device_short_id: str, when: datetime | None = None) -> str:
    """``<id do unicorn>_<HHMMSSDDMMAAAA>``, por exemplo ``16_03402010092026``.

    Sem dados pessoais no nome — só o número de série do equipamento e o
    instante (SPEC 6.3).
    """
    stamp = (when or datetime.now()).strftime("%H%M%S%d%m%Y")
    return f"{device_short_id}_{stamp}"


def next_session_id(sessions_dir: Path) -> str:
    """``p001``, ``p002``, ... Sequencial, sem qualquer identificador pessoal."""
    existing = [
        p.name.rsplit("_p", 1)[-1]
        for p in sessions_dir.glob("*_p*")
        if p.is_dir() and "_p" in p.name
    ]
    numbers = [int(s) for s in existing if s.isdigit()]
    return f"p{max(numbers, default=0) + 1:03d}"


@dataclass
class SessionEvent:
    """Uma transição, um aborto ou uma nota do operador."""

    t_s: float
    kind: str
    detail: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


class Recorder:
    """Escreve uma sessão para ``data/sessions/<timestamp>_<id>/``."""

    def __init__(
        self,
        cfg: Config,
        session_dir: Path | None = None,
        device_short_id: str | None = None,
    ) -> None:
        self._cfg = cfg
        # As sessões vivem dentro da pasta que é sincronizada para o Drive.
        sessions_root = cfg.paths.sessions_dir / cfg.paths.drive_folder_name
        sessions_root.mkdir(parents=True, exist_ok=True)
        if session_dir is None:
            if device_short_id:
                session_dir = sessions_root / session_folder_name(device_short_id)
            else:
                stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
                session_dir = sessions_root / f"{stamp}_{next_session_id(sessions_root)}"
        self.dir = Path(session_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

        self._stream_path = self.dir / RAW_STREAM_NAME
        self._fh = self._stream_path.open("wb")
        self._n_samples = 0
        self.events: list[SessionEvent] = []
        self._closed = False

    # -- escrita ------------------------------------------------------------- #
    def write(self, chunk: np.ndarray) -> None:
        """Acrescenta ``(17, k)``. Chamado pela thread de aquisição."""
        if self._closed:
            raise RuntimeError("recorder já fechado")
        if chunk.shape[0] != N_CANONICAL_CHANNELS:
            raise ValueError(
                f"esperado ({N_CANONICAL_CHANNELS}, k), obtido {chunk.shape}"
            )
        # Guardado transposto (amostras em linhas) para o append ser contíguo.
        np.ascontiguousarray(chunk.T, dtype=np.float32).tofile(self._fh)
        self._n_samples += chunk.shape[1]

    def add_event(self, kind: str, t_s: float, detail: str = "", **payload: Any) -> None:
        self.events.append(SessionEvent(t_s=t_s, kind=kind, detail=detail, payload=payload))

    @property
    def n_samples(self) -> int:
        return self._n_samples

    # -- fecho --------------------------------------------------------------- #
    def finalize(
        self,
        device_info: DeviceInfo,
        measured_sfreq: float,
        acquisition_stats: Any = None,
        montage_validation: dict[str, Any] | None = None,
        extra_meta: dict[str, Any] | None = None,
        keep_stream: bool = False,
    ) -> Path:
        """Converte para ``raw.npy`` e escreve os metadados. Devolve a pasta."""
        if not self._closed:
            self._fh.close()
            self._closed = True

        raw = np.fromfile(self._stream_path, dtype=np.float32)
        expected = self._n_samples * N_CANONICAL_CHANNELS
        if raw.size != expected:
            # Sessão truncada por crash: aproveita-se o que houver.
            usable = (raw.size // N_CANONICAL_CHANNELS) * N_CANONICAL_CHANNELS
            self.add_event(
                "truncated_recording",
                t_s=0.0,
                detail=f"{raw.size} valores lidos, {expected} esperados",
            )
            raw = raw[:usable]
            self._n_samples = usable // N_CANONICAL_CHANNELS
        array = raw.reshape(self._n_samples, N_CANONICAL_CHANNELS).T
        np.save(self.dir / RAW_ARRAY_NAME, np.ascontiguousarray(array))
        if not keep_stream:
            self._stream_path.unlink(missing_ok=True)

        cfg, proto = self._cfg, self._cfg.protocol
        adequacy = cfg.reference_adequacy()
        meta: dict[str, Any] = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "code_version": _code_version(),
            "git_hash": _git_hash(),
            "config_path": str(cfg.source_path) if cfg.source_path else None,
            "n_samples": int(self._n_samples),
            "n_channels": N_CANONICAL_CHANNELS,
            "row_names": list(CANONICAL_ROW_NAMES),
            "units_eeg": "uV",
            "sfreq_nominal": cfg.device.sfreq_nominal,
            "sfreq_measured": float(measured_sfreq),
            "source": {
                "name": device_info.source_name,
                "device_id": device_info.device_id,
                "units_to_uv": device_info.units_to_uv,
                # Guardados só para auditoria. A montagem NÃO vem daqui: o
                # BrainFlow reporta a montagem default do Unicorn, que não é a
                # nossa, e usá-la trocaria F3 com F4 em silêncio.
                "backend_channel_names": list(device_info.backend_channel_names),
                "backend_names_used_for_montage": False,
                "extra": _jsonable(device_info.extra),
            },
            "montage": {
                "channels": {str(k): v for k, v in cfg.montage.channels.items()},
                "metric_channels": list(cfg.montage.metric_channels),
                "eog_channels": list(cfg.montage.eog_channels),
                "unused_channels": list(cfg.montage.unused_channels),
                "reference": cfg.montage.reference,
                "car_applied": cfg.montage.apply_car,
                "validation": montage_validation or {},
            },
            # A condição ocular e as durações viajam com os dados: nunca são
            # inferidas, e comparar segmentos de condições diferentes levanta
            # exceção na análise (SPEC 2).
            "eye_condition": proto.eye_condition,
            "durations_s": {
                "calibration": proto.calibration_s,
                "calib_use_last": proto.calib_use_last_s,
                "mantra": proto.mantra_s,
                "settle": proto.settle_s,
            },
            "settle_used_as_baseline": proto.settle_is_baseline,
            "reference_adequacy": asdict(adequacy),
        }
        if acquisition_stats is not None:
            meta["acquisition"] = {
                "counter_gaps": [
                    {"at_sample": s, "from": a, "to": b}
                    for s, a, b in getattr(acquisition_stats, "counter_gaps", [])
                ],
                "overruns": getattr(acquisition_stats, "overruns", 0),
                "discarded_settle_samples": getattr(
                    acquisition_stats, "discarded_settle_samples", 0
                ),
                "plausibility_warnings": list(
                    getattr(acquisition_stats, "plausibility_warnings", [])
                ),
            }
        if extra_meta:
            meta.update(_jsonable(extra_meta))

        (self.dir / RAW_META_NAME).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (self.dir / EVENTS_NAME).write_text(
            json.dumps(
                [asdict(e) for e in self.events], indent=2, ensure_ascii=False
            ),
            encoding="utf-8",
        )
        return self.dir

    def abort(self) -> None:
        if not self._closed:
            self._fh.close()
            self._closed = True


def _jsonable(obj: Any) -> Any:
    """Converte tipos numpy/Path para algo que o ``json`` aceite."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj
