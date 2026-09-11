"""Passagem de análise post-hoc: do sinal gravado às séries de marcadores.

Corre uma vez, sobre a gravação completa, depois de a sessão terminar. Itera o
registo de ``markers.py`` — acrescentar um marcador lá faz com que apareça
aqui, no gráfico e na tabela, sem tocar neste ficheiro.

**Qual é a estatística de resumo.** Para reduzir uma fase a um número usa-se a
**mediana**, não a média nem o Q3. Três razões medidas nestes dados: as séries
de potência de banda têm cauda pesada (uma época com artefacto muda a média e
não mexe na mediana); o gate deixa buracos, e a mediana lida com eles sem
interpolar; e o delta de Cliff, que é a dimensão de efeito, é ele próprio
baseado em ordens — usar a mediana mantém a coerência entre o número que se
mostra e o teste que o sustenta.

A variação é sempre **face à calibração da própria pessoa**. Nunca limiares
absolutos entre pessoas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np

from .config import Config
from .config import Config as _Config  # noqa: F401  (tipo já importado)
from .markers import Marker, markers
from .preprocess import Preprocessed, preprocess
from .quality import EpochQuality, epoch_quality
from .spectral import welch_psd
from .stats import block_bootstrap_ci, cliffs_delta

#: Marcadores em que a variacao percentual nao significa nada: quantidades
#: COM SINAL e centradas perto de zero. Medido: o ALAY foi de -0,0304 para
#: -0,0476, o que sai como "+57 %" e se le como melhoria — quando na verdade
#: ficou mais negativo, ou seja na direcao oposta. Para estes reporta-se a
#: diferenca absoluta nas unidades do proprio marcador.
SIGNED_MARKERS = frozenset({"faa", "wpli_alpha1", "imcoh_alpha1"})

PHASE_BASELINE = "CALIBRATION"
PHASE_ACTIVE = "MANTRA"


@dataclass(frozen=True)
class MarkerSeries:
    """A série temporal de um marcador ao longo da sessão."""

    marker: Marker
    times_s: np.ndarray          # centro de cada época, em segundos de sessão
    values: np.ndarray           # nan onde a época foi rejeitada
    phases: np.ndarray           # rótulo de fase por época

    def marker_id_matches(self, ids: list[str]) -> bool:
        return self.marker.id in ids

    def of_phase(self, phase: str) -> np.ndarray:
        selected = self.values[self.phases == phase]
        return selected[np.isfinite(selected)]

    @property
    def coverage(self) -> float:
        if self.values.size == 0:
            return 0.0
        return float(np.isfinite(self.values).mean())


@dataclass(frozen=True)
class MarkerSummary:
    """O resumo de um marcador: quanto mudou, e se dá para acreditar."""

    marker: Marker
    baseline: float
    active: float
    pct_change: float
    cliffs_delta: float
    ci_low: float
    ci_high: float
    n_effective: float
    coverage: float
    n_baseline: int
    n_active: int
    r_emg: float = float("nan")
    #: Banda larga (20-45 Hz). O EMG mandibular contamina o beta (15-30) muito
    #: abaixo da banda estreita: medido nesta montagem, trincar multiplica a
    #: energia por 34x em 20-30 Hz. Um r_emg estreito baixo NAO exonera nada.
    r_emg_wide: float = float("nan")
    r_motion: float = float("nan")

    @property
    def crosses_zero(self) -> bool:
        return bool(
            not np.isfinite(self.ci_low) or self.ci_low <= 0.0 <= self.ci_high
        )

    @property
    def reliable(self) -> bool:
        """Há dados que cheguem e o IC não cruza zero."""
        return bool(
            np.isfinite(self.ci_low) and not self.crosses_zero and self.n_effective >= 5
        )

    @property
    def is_signed(self) -> bool:
        """A percentagem não se aplica: quantidade com sinal perto de zero."""
        return self.marker.id in SIGNED_MARKERS

    @property
    def delta(self) -> float:
        """Diferença absoluta, nas unidades do marcador."""
        return self.active - self.baseline

    def change_text(self) -> str:
        """Como se escreve a mudança. Percentagem só onde ela faz sentido.

        "sem dados" e "sem referência" não são a mesma coisa e não se dizem da
        mesma maneira: numa sessão de 3 min a coerência teve 11 janelas boas
        na meditação e **uma** na calibração, portanto havia valor mas não
        havia contra o que o comparar. Dizer "sem dados" mandava procurar um
        problema no sítio errado — o que falta é calibração mais longa.
        """
        if self.n_baseline < 3 <= self.n_active:
            return "sem referência"
        if self.is_signed:
            if not np.isfinite(self.delta):
                return "sem dados"
            # O delta prefixado, para nao se confundir com percentagem.
            return f"D{self.delta:+.3f}"
        if not np.isfinite(self.pct_change):
            return "sem dados"
        return f"{self.pct_change:+.0f}%"

    @property
    def direction_ok(self) -> bool | None:
        """``None`` para marcadores sem direção esperada — não é falso."""
        if self.marker.expected_direction == "unknown":
            return None
        change = self.delta if self.is_signed else self.pct_change
        if not np.isfinite(change):
            return None
        return (change > 0) == (self.marker.expected_direction == "up")

    def headline(self) -> str:
        """Frase curta e honesta. Nunca inventa melhoria (SPEC 5.1)."""
        if not np.isfinite(self.pct_change):
            return f"{self.marker.friendly or self.marker.label}: sem dados"
        name = self.marker.friendly or self.marker.label
        if not self.reliable:
            return f"{name}: estável"
        return f"{name}: {self.change_text()}"


@dataclass
class AnalysisResult:
    series: list[MarkerSeries] = field(default_factory=list)
    summaries: list[MarkerSummary] = field(default_factory=list)
    quality: EpochQuality | None = None
    preprocessed: Preprocessed | None = None
    notes: list[str] = field(default_factory=list)

    def summary_of(self, marker_id: str) -> MarkerSummary | None:
        return next((s for s in self.summaries if s.marker.id == marker_id), None)

    def series_of(self, marker_id: str) -> MarkerSeries | None:
        return next((s for s in self.series if s.marker.id == marker_id), None)


# --------------------------------------------------------------------------- #
def _phase_at(t: float, phases: Iterable[tuple[str, float, float]]) -> str:
    for name, start, end in phases:
        if start <= t < end:
            return name
    return ""


def _epoch_grid(
    n_samples: int, sfreq: float, win_s: float, hop_s: float
) -> list[tuple[int, int, float]]:
    win, hop = int(round(win_s * sfreq)), int(round(hop_s * sfreq))
    return [
        (s, s + win, (s + win / 2.0) / sfreq)
        for s in range(0, max(n_samples - win + 1, 0), hop)
    ]


def analyse(
    raw_eeg: np.ndarray,
    cfg: Config,
    phases: list[tuple[str, float, float]],
    accel: np.ndarray | None = None,
    gyro: np.ndarray | None = None,
    selected: Mapping[str, bool] | None = None,
) -> AnalysisResult:
    """Corre o registo completo sobre a gravação.

    ``phases`` são ``(nome, início_s, fim_s)`` na base de tempo da gravação,
    tal como ``events.json`` as regista.
    """
    result = AnalysisResult()
    pre = preprocess(raw_eeg, cfg)
    result.preprocessed = pre
    sfreq = pre.sfreq
    chans = cfg.montage.name_to_index

    quality = epoch_quality(pre, cfg, accel=accel, gyro=gyro)
    result.quality = quality
    if pre.eog_applied:
        result.notes.append("correção de EOG por regressão aplicada")

    wanted = [
        m
        for m in markers()
        if selected is None or selected.get(m.id, m.role == "marker")
    ]

    # -- grelha de potência ------------------------------------------------- #
    power_grid = _epoch_grid(
        pre.n_samples, sfreq, cfg.epochs.power.win_s, cfg.epochs.power.hop_s
    )
    psd_cache: list[tuple[np.ndarray, np.ndarray]] = []
    for start, stop, _ in power_grid:
        freqs, psd = welch_psd(
            pre.clean[:, start:stop],
            sfreq,
            cfg.epochs.power.welch_seg_s,
            cfg.epochs.power.welch_overlap,
        )
        psd_cache.append((freqs, psd))

    times_power = np.array([pre.t0_s + t for _, _, t in power_grid])
    phases_power = np.array([_phase_at(t, phases) for t in times_power])

    # -- grelha de conectividade, mais lenta -------------------------------- #
    conn = cfg.epochs.connectivity
    conn_grid = _epoch_grid(pre.n_samples, sfreq, conn.win_s, conn.hop_s)
    times_conn = np.array([pre.t0_s + t for _, _, t in conn_grid])
    phases_conn = np.array([_phase_at(t, phases) for t in times_conn])

    for marker in wanted:
        if marker.kind == "pair":
            values = np.full(len(conn_grid), np.nan)
            for i, (start, stop, _) in enumerate(conn_grid):
                if not quality.window_ok(start, stop, marker.channels):
                    continue
                values[i] = marker.fn(
                    pre.clean[:, start:stop],
                    sfreq,
                    chans,
                    cfg.bands,
                    min_segments=conn.min_segments,
                )
            series = MarkerSeries(marker, times_conn, values, phases_conn)
        elif marker.kind == "timeseries":
            values = np.full(len(power_grid), np.nan)
            for i, (start, stop, _) in enumerate(power_grid):
                if not quality.window_ok(start, stop, marker.channels):
                    continue
                values[i] = marker.fn(pre.clean[:, start:stop], chans, cfg.bands)
            series = MarkerSeries(marker, times_power, values, phases_power)
        else:
            values = np.full(len(power_grid), np.nan)
            for i, (start, stop, _) in enumerate(power_grid):
                if not quality.window_ok(start, stop, marker.channels):
                    continue
                freqs, psd = psd_cache[i]
                values[i] = marker.fn(psd, freqs, chans, cfg.bands, **marker.params)
            series = MarkerSeries(marker, times_power, values, phases_power)

        result.series.append(series)
        result.summaries.append(_summarise(series, cfg, quality))

    return result


def _summarise(
    series: MarkerSeries, cfg: Config, quality: EpochQuality
) -> MarkerSummary:
    baseline = series.of_phase(PHASE_BASELINE)
    active = series.of_phase(PHASE_ACTIVE)

    if baseline.size < 3 or active.size < 3:
        return MarkerSummary(
            marker=series.marker,
            baseline=float(np.median(baseline)) if baseline.size else float("nan"),
            active=float(np.median(active)) if active.size else float("nan"),
            pct_change=float("nan"),
            cliffs_delta=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n_effective=0.0,
            coverage=series.coverage,
            n_baseline=int(baseline.size),
            n_active=int(active.size),
        )

    med_b, med_a = float(np.median(baseline)), float(np.median(active))
    pct = (med_a / med_b - 1.0) * 100.0 if med_b != 0 else float("nan")

    bb = cfg.stats.block_bootstrap
    hop = (
        cfg.epochs.connectivity.hop_s
        if series.marker.is_connectivity
        else cfg.epochs.power.hop_s
    )
    estimate = block_bootstrap_ci(
        baseline,
        active,
        lambda x, y: float(np.median(y) - np.median(x)),
        hop_s=hop,
        min_length_s=bb.min_length_s,
        max_length_s=bb.max_length_s,
        n_boot=cfg.stats.bootstrap_n,
        ci_level=cfg.stats.ci_level,
    )

    return MarkerSummary(
        marker=series.marker,
        baseline=med_b,
        active=med_a,
        pct_change=pct,
        cliffs_delta=cliffs_delta(baseline, active),
        ci_low=estimate.ci_low,
        ci_high=estimate.ci_high,
        n_effective=estimate.n_effective,
        coverage=series.coverage,
        n_baseline=int(baseline.size),
        n_active=int(active.size),
        r_emg=quality.correlate(series.values, "emg", series.times_s),
        r_emg_wide=quality.correlate(series.values, "emg_wide", series.times_s),
        r_motion=quality.correlate(series.values, "motion", series.times_s),
    )


def baseline_spread(result: AnalysisResult, marker_id: str) -> float:
    """A oscilação de um marcador durante a calibração (MAD, escala de desvio).

    É a régua com que se decide se uma diferença é grande: grande **para esta
    pessoa**, e não em abstrato.
    """
    series = result.series_of(marker_id)
    if series is None:
        return float("nan")
    base = series.of_phase(PHASE_BASELINE)
    base = base[np.isfinite(base)]
    if base.size < 3:
        return float("nan")
    median = float(np.median(base))
    mad = float(np.median(np.abs(base - median))) * 1.4826
    return mad if np.isfinite(mad) and mad > 0 else float("nan")


def percent_change(result: AnalysisResult, summary: MarkerSummary) -> float:
    """A variação percentual, ou ``nan`` quando a percentagem não diz nada.

    Uma percentagem é a variação a dividir pela linha de base, e só significa
    alguma coisa se a linha de base for uma quantidade com zero verdadeiro e
    longe dele. Duas situações em que não é:

    - **Quantidades com sinal** (``SIGNED_MARKERS``). O ALAY é
      ``ln P[F4] − ln P[F3]``: já é uma razão, e o seu zero é arbitrário — é
      o ponto onde os dois hemisférios coincidem, não uma ausência de nada.
      Dividir por ele é dividir pela distância a uma origem convencionada.
      Numa sessão real deu **−46 %** para uma diferença de 0,002, enquanto o
      mesmo marcador media 0,2 desvios da oscilação normal. Os dois números
      descreviam a mesma coisa e contavam histórias opostas.
    - **Linhas de base indistinguíveis de zero**, a menos de uma dispersão:
      o denominador é ruído e o quociente explode.

    Esta é a régua única. O relatório e o ecrã final chamam-na os dois, senão
    voltam a discordar — foi assim que um mesmo marcador apareceu a −46 % num
    sítio e a ~0 no outro.
    """
    if summary.marker.id in SIGNED_MARKERS:
        return float("nan")
    baseline = summary.baseline
    spread = baseline_spread(result, summary.marker.id)
    if not np.isfinite(baseline) or not np.isfinite(summary.active):
        return float("nan")
    if not np.isfinite(spread) or abs(baseline) <= spread:
        return float("nan")
    return float((summary.active - baseline) / baseline * 100.0)


def rank_by_variation(
    result: AnalysisResult,
    cfg: Config,
    limit: int = 3,
    candidates: Iterable[str] | None = None,
) -> list[str]:
    """Os marcadores que mais mudaram nesta sessão, para destaque.

    Cada cérebro responde de forma diferente e o que se move numa pessoa não
    se move noutra, por isso o destaque é escolhido por sessão em vez de fixo.

    **O critério é a magnitude absoluta da variação face ao repouso**, em
    percentagem — a mesma escala que o gráfico de barras mostra, para o
    destaque e o gráfico não se contradizerem. Onde a percentagem não
    significa nada (quantidades com sinal que vivem perto de zero, como o
    ALAY: a razão tem um denominador minúsculo e sai um número enorme), o
    critério cai para a variação em unidades da oscilação do próprio repouso.

    Nada é **excluído** — o resultado tem sempre ``limit`` entradas quando há
    marcadores que cheguem. O que os avisos fazem é **despromover**: um
    marcador arrastado pela mandíbula, ou com cobertura baixa, só aparece em
    destaque se não houver melhor. O cartão continua a dizer, em texto, que a
    leitura pode não ser cerebral — a cor e a frase é que carregam a ressalva,
    não a ausência.

    A versão anterior filtrava a sério e ordenava pela magnitude dividida pela
    largura do IC. Numa sessão real isso deixou **um** cartão dos três: os
    filtros comiam-se uns aos outros, e um ecrã de destaque com um cartão não
    é um ecrã de destaque.
    """
    pool = set(candidates) if candidates is not None else {
        m.id for m in markers(role="marker")
    }
    scored: list[tuple[int, float, str]] = []
    for summary in result.summaries:
        marker = summary.marker
        if marker.id not in pool or marker.role != "marker":
            continue
        if not np.isfinite(summary.baseline) or not np.isfinite(summary.active):
            continue

        spread = baseline_spread(result, marker.id)
        change = summary.active - summary.baseline
        percent = percent_change(result, summary)
        if np.isfinite(percent):
            score = abs(percent)
        elif np.isfinite(spread):
            # Escala diferente da percentagem, por isso estes ficam atrás por
            # construção — o que é o que se quer: o destaque prefere o que se
            # sabe exprimir em percentagem, que é o que a pessoa vai ler.
            score = abs(change / spread)
        else:
            continue
        if not np.isfinite(score) or score == 0:
            continue

        # Despromoção, não exclusão. 0 = limpo, 1 = com ressalva.
        flagged = int(
            abs(summary.r_emg) > cfg.diagnostics.r_emg_red_threshold
            or abs(summary.r_motion) > cfg.diagnostics.r_motion_red_threshold
            or summary.coverage < cfg.quality.min_calibration_coverage
        )
        scored.append((flagged, score, marker.id))

    scored.sort(key=lambda item: (item[0], -item[1]))
    return [marker_id for _, _, marker_id in scored[:limit]]
