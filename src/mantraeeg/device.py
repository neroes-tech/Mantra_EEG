"""Gestão do dispositivo: ligar, desligar, vigiar e voltar a ligar.

Numa banca de festival a aplicação passa horas sem ninguém, o headset é
desligado para poupar bateria, e volta a ser ligado quando aparece alguém. A
aplicação tem de sobreviver a isso sem bloquear — e a versão anterior não
sobrevivia: ficava presa quando o dispositivo desaparecia.

Três decisões que resolvem isso:

**A aquisição só corre quando é precisa.** Fora da sessão (ecrã de boas-vindas,
relatório) o dispositivo é libertado. É também o que o protocolo pede: o
equipamento só recolhe sinal da calibração ao fim do repouso.

**Há um cão-de-guarda.** Se deixarem de chegar amostras durante
``stale_after_s``, o estado passa a ``STALE`` e a interface diz-o, em vez de
mostrar um traçado congelado que parece sinal.

**Ligar nunca bloqueia a interface.** A tentativa corre numa thread; a janela
continua a responder mesmo que o BrainFlow demore 15 s a desistir.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

from .acquisition import Acquisition
from .config import Config
from .sources import make_source
from .sources.base import DeviceInfo, EegSource, SourceError


class DeviceState(str, Enum):
    DISCONNECTED = "desligado"
    CONNECTING = "a ligar"
    READY = "ligado"
    STREAMING = "a adquirir"
    STALE = "sem sinal"
    ERROR = "erro"


@dataclass(frozen=True)
class DeviceStatus:
    state: DeviceState
    detail: str = ""
    device_id: str = ""
    source_name: str = ""
    seconds_since_data: float = float("nan")
    settled: bool = False
    settle_remaining_s: float = 0.0

    @property
    def usable(self) -> bool:
        return self.state in (DeviceState.READY, DeviceState.STREAMING)

    @property
    def needs_restart(self) -> bool:
        """O operador tem de mexer no headset (ligar, ou reiniciar)."""
        return self.state in (DeviceState.DISCONNECTED, DeviceState.STALE, DeviceState.ERROR)


class DeviceManager:
    """Ciclo de vida do dispositivo, isolado da interface."""

    CASCADE = ("brainflow", "unicorn_dll", "lsl")

    def __init__(
        self,
        cfg: Config,
        on_chunk: Callable[[np.ndarray], None] | None = None,
        source_override: str | None = None,
        replay_file: str | None = None,
    ) -> None:
        self._cfg = cfg
        self._on_chunk = on_chunk
        self._override = source_override
        self._replay_file = replay_file

        self._source: EegSource | None = None
        self._acq: Acquisition | None = None
        self._state = DeviceState.DISCONNECTED
        self._detail = "nunca ligado"
        self._info: DeviceInfo | None = None
        self._last_samples = 0
        self._last_data_at = 0.0
        self._lock = threading.Lock()
        self._connecting: threading.Thread | None = None

    # -- estado ---------------------------------------------------------------- #
    @property
    def acquisition(self) -> Acquisition | None:
        return self._acq

    @property
    def info(self) -> DeviceInfo | None:
        return self._info

    def status(self) -> DeviceStatus:
        with self._lock:
            state, detail = self._state, self._detail
            acq, info = self._acq, self._info

        since = float("nan")
        if state == DeviceState.STREAMING and acq is not None:
            samples = acq.stats.n_samples
            now = time.monotonic()
            if samples != self._last_samples:
                self._last_samples = samples
                self._last_data_at = now
            elif self._last_data_at == 0.0:
                self._last_data_at = now
            since = now - self._last_data_at
            # Durante o assentamento ainda não há amostras contadas: não é falha.
            if (
                since > self._cfg.device.stale_after_s
                and acq.settled
                and self._state == DeviceState.STREAMING
            ):
                self._set(DeviceState.STALE, f"sem amostras há {since:.0f} s")
                state = DeviceState.STALE

        return DeviceStatus(
            state=state,
            detail=detail,
            device_id=info.device_id if info else "",
            source_name=info.source_name if info else "",
            seconds_since_data=since,
            settled=bool(acq.settled) if acq else False,
            settle_remaining_s=float(acq.settle_remaining_s) if acq else 0.0,
        )

    def _set(self, state: DeviceState, detail: str) -> None:
        with self._lock:
            self._state = state
            self._detail = detail

    # -- ligação ---------------------------------------------------------------- #
    def connect_async(self, done: Callable[[DeviceStatus], None] | None = None) -> None:
        """Liga numa thread, para a interface nunca ficar presa."""
        if self._connecting is not None and self._connecting.is_alive():
            return
        if self._state in (DeviceState.READY, DeviceState.STREAMING):
            if done:
                done(self.status())
            return

        def worker() -> None:
            self.connect()
            if done:
                done(self.status())

        self._set(DeviceState.CONNECTING, "a procurar o dispositivo…")
        self._connecting = threading.Thread(target=worker, daemon=True)
        self._connecting.start()

    def connect(self) -> bool:
        """Percorre a cascata da SPEC 1.3. Devolve ``True`` se abriu."""
        self.disconnect()
        order = [self._override] if self._override else list(self.CASCADE)
        failures: list[str] = []
        for name in order:
            try:
                source = make_source(self._cfg, override=name, path=self._replay_file)
                info = source.open()
            except Exception as exc:
                failures.append(f"{name}: {exc}")
                continue
            self._source = source
            self._info = info
            self._acq = Acquisition(self._cfg, source, on_chunk=self._forward)
            self._set(DeviceState.READY, f"{info.source_name} · {info.device_id}")
            return True

        self._set(
            DeviceState.DISCONNECTED,
            "nenhum dispositivo encontrado. Ligue o headset (auto-desliga por "
            "inatividade) e confirme o dongle.",
        )
        self._info = None
        return False

    def disconnect(self) -> None:
        if self._acq is not None:
            try:
                self._acq.stop()
            except Exception:
                pass
            self._acq = None
        if self._source is not None:
            try:
                self._source.stop()
            finally:
                try:
                    self._source.close()
                except Exception:
                    pass
            self._source = None
        self._info = None
        self._last_samples = 0
        self._last_data_at = 0.0
        self._set(DeviceState.DISCONNECTED, "desligado")

    # -- aquisição ---------------------------------------------------------------- #
    def settle_remaining_s(self) -> float:
        """Segundos que faltam para o sinal assentar. Zero quando ja assentou."""
        return self._acq.settle_remaining_s if self._acq is not None else 0.0

    def start_streaming(self) -> bool:
        """Começa a adquirir. O LED do Unicorn passa a azul fixo."""
        if self._acq is None and not self.connect():
            return False
        assert self._acq is not None
        if self._acq.running:
            return True
        try:
            self._acq.start()
        except Exception as exc:
            self._set(DeviceState.ERROR, f"falha ao arrancar a aquisição: {exc}")
            return False
        self._last_samples = 0
        self._last_data_at = time.monotonic()
        self._set(DeviceState.STREAMING, "a adquirir")
        return True

    def stop_streaming(self) -> None:
        """Pára de adquirir mas mantém o dispositivo aberto (LED a piscar)."""
        if self._acq is not None and self._acq.running:
            try:
                self._acq.stop()
            except Exception:
                pass
        if self._state in (DeviceState.STREAMING, DeviceState.STALE):
            self._set(DeviceState.READY, "em espera")

    def _forward(self, chunk: np.ndarray) -> None:
        if self._on_chunk is not None:
            self._on_chunk(chunk)

    # -- utilitários ------------------------------------------------------------- #
    @property
    def short_id(self) -> str:
        """Sufixo do número de série, para nomear a pasta da sessão.

        ``UN-2022.01.16`` -> ``16``. Sem dispositivo, ``sem-id``.
        """
        raw = (self._info.device_id if self._info else "") or ""
        tail = raw.replace("-", ".").split(".")[-1].strip()
        # So aceitar algo que pareca um numero de serie. Sem isto, um
        # device_id como "(auto-detetado)" produzia a pasta "detetado)".
        if tail and all(c.isalnum() for c in tail):
            return tail
        return "sem-id"

    def peek_last(self, seconds: float) -> np.ndarray:
        if self._acq is None:
            return np.zeros((17, 0), dtype=np.float32)
        return self._acq.peek_last(seconds)

    @property
    def elapsed_s(self) -> float:
        return self._acq.elapsed_s if self._acq is not None else 0.0
