"""Thread de aquisição, ring buffer e monitor de contacto.

Durante a sessão correm **só duas coisas** (SPEC 1.4): a escrita da gravação e a
vigilância de elétrodos. Nenhum biomarcador, nenhum índice, nenhuma
normalização. A UI nunca lê do dispositivo — lê deste ring buffer.

O monitor de contacto avalia o sinal **depois de filtrar** (notch de 50 Hz +
passa-banda de 1-45 Hz), porque é esse o sinal que a análise vai usar. Julgar o
sinal bruto reprovava canais pelo ruído que o próprio pipeline remove a seguir,
e era isso que fazia com que nenhum elétrodo seco alguma vez ficasse verde.

A janela é sempre detrended primeiro: o Unicorn entrega µV com offset de
eletrodo de centenas de mV (medido: 185 a 711 mV) e, com contacto mau, o sinal
deriva — ao vivo, com os elétrodos soltos, mediu-se o DC a subir 5 660 µV/s.

O indicador de validação do dispositivo (linha 16) não serve: leu ``1`` em todas
as amostras com os elétrodos completamente soltos.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from .config import Config
from .montage import check_channel_independence
from .preprocess import bandpass, detrend, notch
from .ring_buffer import RingBuffer
from .sources.base import (
    COUNTER_ROW,
    EEG_SLICE,
    N_CANONICAL_CHANNELS,
    EegSource,
    plausibility_warning,
)
from .spectral import relative_power, spectral_slope, welch_psd

ContactLevel = Literal["green", "yellow", "red"]

#: Tolerância na deteção de falhas do contador (uint32 na prática; só é preciso
#: distinguir +1 de um salto).
_COUNTER_WRAP_TOLERANCE = 1


# --------------------------------------------------------------------------- #
# Monitor de contacto
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ChannelContact:
    """Estado de contacto de um elétrodo, com o motivo por extenso."""

    name: str
    index: int
    level: ContactLevel
    reason: str
    #: Medidos sobre a janela FILTRADA — o sinal que a análise vai usar.
    std_uv: float
    p2p_uv: float
    #: Razão 48-52 / 1-45 medida ANTES do notch. Informativa: a análise remove a
    #: rede, por isso nunca reprova um canal sozinha.
    line_rel: float
    #: Declive log-log em 2-40 Hz com o alfa mascarado. EEG real decai.
    slope: float
    #: Taxa de deriva no sinal cru, em uV/s. Apanha eletrodos a flutuar.
    drift_uv_per_s: float


@dataclass(frozen=True)
class ContactReport:
    channels: tuple[ChannelContact, ...]
    #: Referência ou terra soltos: todos os canais a ver o mesmo sinal.
    global_warning: str | None = None
    #: Ambiente com rede forte. Não reprova canais — o notch trata dela.
    line_warning: str | None = None

    @property
    def n_good(self) -> int:
        return sum(1 for c in self.channels if c.level != "red")

    def by_name(self, name: str) -> ChannelContact:
        for c in self.channels:
            if c.name == name:
                return c
        raise KeyError(f"canal {name!r} não está no relatório de contacto")


class ContactMonitor:
    """Semáforo por elétrodo, a cada ``update_period_s`` sobre ``window_s``.

    Os limiares são de **elétrodos secos**, não de gel em sala blindada. Mesmo
    assim o árbitro final não é este semáforo: é o teste de Berger
    (``scripts/compare_segments.py``), que pergunta se o canal consegue recuperar
    um efeito fisiológico conhecido. Um canal amarelo que mostra o alfa a subir
    ao fechar os olhos vale mais do que um verde que não mostra nada.
    """

    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._cm = cfg.contact_monitor
        self._montage = cfg.montage
        self._sfreq = cfg.device.sfreq_nominal
        self._pp = cfg.preprocess
        self._line_band = cfg.bands["line"]
        self._total_band = cfg.bands.total
        self._alpha_band = cfg.bands["alpha"]

    @property
    def window_samples(self) -> int:
        return int(round(self._cm.window_s * self._sfreq))

    def evaluate(self, window: np.ndarray) -> ContactReport:
        """Avalia uma janela ``(17, n)`` crua, tal como sai do dispositivo."""
        raw = np.asarray(window[EEG_SLICE], dtype=np.float64)
        if raw.shape[1] < int(self._sfreq):  # menos de 1 s não dá espectro
            return ContactReport(channels=(), global_warning="janela demasiado curta")

        detrended = (
            detrend(raw, self._pp.detrend) if self._cm.detrend_window else raw.copy()
        )
        # A deriva mede-se no sinal CRU: e exatamente o que o detrend remove, por
        # isso tem de ser lida antes, ou um eletrodo a flutuar passa despercebido.
        seconds = raw.shape[1] / self._sfreq
        drift = np.ptp(raw - detrended, axis=-1) / max(seconds, 1e-9)

        # Índice de rede ANTES do notch, senão seria sempre ~0 e nunca avisaria.
        freqs_raw, psd_raw = welch_psd(detrended, self._sfreq, self._cm.window_s)
        line_rel = relative_power(psd_raw, freqs_raw, self._line_band, self._total_band)

        graded = detrended
        if self._cm.grade_on_filtered:
            graded = notch(
                detrended,
                self._sfreq,
                self._pp.notch_freqs_hz,
                self._pp.notch_q,
                self._pp.notch_passes,
            )
            graded = bandpass(
                graded, self._sfreq, self._pp.bandpass_hz, self._pp.bandpass_order
            )

        std = graded.std(axis=-1)
        p2p = graded.max(axis=-1) - graded.min(axis=-1)

        # O declive sai do sinal com notch mas SEM passa-banda: o rolloff do
        # passa-banda impoe um declive negativo que faria ruido branco passar
        # por 1/f. O intervalo de ajuste fica bem dentro da banda.
        notched = notch(
            detrended,
            self._sfreq,
            self._pp.notch_freqs_hz,
            self._pp.notch_q,
            self._pp.notch_passes,
        )
        freqs, psd = welch_psd(notched, self._sfreq, self._cm.slope_welch_seg_s)
        slope = spectral_slope(
            psd,
            freqs,
            fit_range=self._cm.slope_fit_range,
            peak_mask_bands=(self._alpha_band,),
        )

        channels: list[ChannelContact] = []
        for idx in sorted(self._montage.channels):
            level, reason = self._grade(
                std=float(std[idx]),
                p2p=float(p2p[idx]),
                drift=float(drift[idx]),
                slope=float(slope[idx]),
            )
            channels.append(
                ChannelContact(
                    name=self._montage.channels[idx],
                    index=idx,
                    level=level,
                    reason=reason,
                    std_uv=float(std[idx]),
                    p2p_uv=float(p2p[idx]),
                    line_rel=float(line_rel[idx]),
                    slope=float(slope[idx]),
                    drift_uv_per_s=float(drift[idx]),
                )
            )

        return ContactReport(
            channels=tuple(channels),
            global_warning=self._independence_warning(detrended),
            line_warning=self._line_warning(line_rel),
        )

    def _grade(
        self, std: float, p2p: float, drift: float, slope: float
    ) -> tuple[ContactLevel, str]:
        """Classifica um canal. O declive e reportado mas NAO classifica: em 2 s o
        estimador e ruidoso demais para decidir (ver a nota na config)."""
        cm = self._cm
        if not np.isfinite(std) or std < cm.min_std_uv:
            return "red", f"elétrodo morto (sd {std:.2f} µV)"
        if drift > cm.max_drift_uv_per_s:
            return "red", f"elétrodo a flutuar (deriva {drift:.0f} µV/s)"

        g, y = cm.green, cm.yellow
        if std <= g.max_std_uv and p2p <= g.max_p2p_uv:
            return "green", "ok"

        hard: list[str] = []
        if std > y.max_std_uv:
            hard.append(f"ruído alto (sd {std:.0f} µV)")
        if p2p > y.max_p2p_uv:
            hard.append(f"amplitude alta (p2p {p2p:.0f} µV)")
        if hard:
            return "red", "; ".join(hard)

        soft: list[str] = []
        if std > g.max_std_uv:
            soft.append(f"sd {std:.0f} µV")
        if p2p > g.max_p2p_uv:
            soft.append(f"p2p {p2p:.0f} µV")
        return "yellow", "; ".join(soft) or "marginal"

    def _line_warning(self, line_rel: np.ndarray) -> str | None:
        """Rede alta é ambiente, não contacto — e o notch remove-a na análise."""
        finite = line_rel[np.isfinite(line_rel)]
        if finite.size == 0:
            return None
        median = float(np.median(finite))
        if median <= self._cm.line_warn_ratio:
            return None
        return (
            f"rede de {self._cfg.device.line_freq_hz:.0f} Hz forte em toda a touca "
            f"(mediana {median:.1f}x a banda 1-45). O notch remove-a na análise, "
            f"mas vale a pena afastar carregadores e fontes de alimentação."
        )

    def _independence_warning(self, eeg: np.ndarray) -> str | None:
        """Referência/terra soltos: delega em ``montage.check_channel_independence``."""
        v = self._montage.validation
        result = check_channel_independence(
            eeg, v.no_contact_min_interchannel_r, v.no_contact_max_pair_diff_ratio
        )
        return result.detail if result.passed is False else None


# --------------------------------------------------------------------------- #
# Thread de aquisição
# --------------------------------------------------------------------------- #
@dataclass
class AcquisitionStats:
    """O que correu durante a aquisição. Vai para ``events.json``."""

    n_samples: int = 0
    counter_gaps: list[tuple[int, float, float]] = field(default_factory=list)
    overruns: int = 0
    started_at: float = 0.0
    stopped_at: float = 0.0
    #: Instante em que o assentamento terminou e as amostras passaram a contar.
    #: A taxa mede-se a partir daqui: se o período descartado entrasse no
    #: denominador, a taxa saía ~34 % abaixo da real.
    counting_from: float = 0.0
    discarded_settle_samples: int = 0
    plausibility_warnings: list[str] = field(default_factory=list)

    @property
    def elapsed_s(self) -> float:
        """Tempo durante o qual as amostras foram efetivamente contadas."""
        start = self.counting_from or self.started_at
        if not start:
            return 0.0
        end = self.stopped_at or time.monotonic()
        return max(end - start, 0.0)

    @property
    def measured_sfreq(self) -> float:
        return self.n_samples / self.elapsed_s if self.elapsed_s > 0 else float("nan")


class Acquisition:
    """Thread dedicada: dispositivo -> ring buffer -> callback de escrita.

    O callback (tipicamente ``Recorder.write``) corre nesta thread. A UI lê o
    ring buffer com :meth:`peek_last` e nunca toca na fonte.
    """

    def __init__(
        self,
        cfg: Config,
        source: EegSource,
        on_chunk: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        self._cfg = cfg
        self._source = source
        self._on_chunk = on_chunk
        self._sfreq = cfg.device.sfreq_nominal
        self.buffer = RingBuffer(
            N_CANONICAL_CHANNELS, int(round(cfg.device.ring_buffer_s * self._sfreq))
        )
        self.stats = AcquisitionStats()
        self.monitor = ContactMonitor(cfg)

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_counter: float | None = None
        self._settle_remaining = int(round(cfg.device.settle_discard_s * self._sfreq))
        self._checked_plausibility = False

    # -- controlo ------------------------------------------------------------ #
    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("aquisição já está a correr")
        self._stop_event.clear()
        self.stats = AcquisitionStats(started_at=time.monotonic())
        self._source.start()
        self._thread = threading.Thread(
            target=self._run, name="mantraeeg-acquisition", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> AcquisitionStats:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        self._source.stop()
        self.stats.stopped_at = time.monotonic()
        self.stats.overruns = self.buffer.stats().overruns
        return self.stats

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def elapsed_s(self) -> float:
        """Segundos de sinal já contados (depois do assentamento)."""
        return self.stats.n_samples / self._sfreq

    @property
    def settled(self) -> bool:
        """O amplificador já assentou e o relógio da sessão está a correr.

        O relógio da sessão é o tempo de **sinal gravado**, para as fronteiras
        das fases caírem em amostras. Enquanto o assentamento decorre esse
        relógio está parado, portanto uma fase iniciada antes disso ficaria com
        duração indefinida — daí o arranque estar bloqueado até aqui.
        """
        return self._settle_remaining <= 0

    @property
    def settle_remaining_s(self) -> float:
        return max(self._settle_remaining, 0) / self._sfreq

    # -- laço ---------------------------------------------------------------- #
    def _run(self) -> None:
        period = self._cfg.device.read_chunk_s
        while not self._stop_event.is_set():
            chunk = self._source.read()
            if chunk.shape[1]:
                self._handle(chunk)
            else:
                time.sleep(period)

    def _handle(self, chunk: np.ndarray) -> None:
        self._track_counter(chunk)

        # Descartar o período de assentamento do amplificador: os offsets DC
        # levam segundos a convergir e essas amostras não são utilizáveis.
        if self._settle_remaining > 0:
            drop = min(self._settle_remaining, chunk.shape[1])
            self._settle_remaining -= drop
            self.stats.discarded_settle_samples += drop
            chunk = chunk[:, drop:]
            if self._settle_remaining == 0 and not self.stats.counting_from:
                self.stats.counting_from = time.monotonic()
            if chunk.shape[1] == 0:
                return

        if not self._checked_plausibility:
            warning = plausibility_warning(chunk[EEG_SLICE])
            if warning:
                self.stats.plausibility_warnings.append(warning)
            self._checked_plausibility = True

        self.buffer.write(chunk)
        self.stats.n_samples += chunk.shape[1]
        if self._on_chunk is not None:
            self._on_chunk(chunk)

    def _track_counter(self, chunk: np.ndarray) -> None:
        """Regista descontinuidades do contador (amostras perdidas)."""
        counter = np.asarray(chunk[COUNTER_ROW], dtype=np.float64)
        if counter.size == 0:
            return
        series = (
            counter
            if self._last_counter is None
            else np.concatenate(([self._last_counter], counter))
        )
        steps = np.diff(series)
        bad = np.nonzero(np.abs(steps - 1.0) > _COUNTER_WRAP_TOLERANCE)[0]
        base = self.stats.n_samples
        for i in bad:
            if steps[i] < 0:  # wrap-around do contador, não é perda
                continue
            self.stats.counter_gaps.append(
                (base + int(i), float(series[i]), float(series[i + 1]))
            )
        self._last_counter = float(counter[-1])

    # -- leitura para a UI --------------------------------------------------- #
    def contact_report(self) -> ContactReport:
        """Avalia o contacto sobre a janela mais recente. Não consome o buffer."""
        window = self.buffer.peek_last(self.monitor.window_samples)
        if window.shape[1] < self.monitor.window_samples:
            return ContactReport(channels=(), global_warning="a acumular dados")
        return self.monitor.evaluate(window)

    def peek_last(self, seconds: float) -> np.ndarray:
        return self.buffer.peek_last(int(round(seconds * self._sfreq)))
