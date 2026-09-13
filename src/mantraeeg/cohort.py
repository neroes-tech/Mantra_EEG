"""Agregação entre sessões: o que acontece, em média, ao ouvir o mantra.

Uma sessão isolada não responde à pergunta — n=1 e sem grupo de comparação.
Juntar sessões também não a responde sozinho, mas responde a uma mais modesta
e honesta: **das sessões em que o sinal prestou, o que é que se moveu, em que
sentido, e com que consistência.**

Três decisões que governam este módulo, todas na direção conservadora:

- **A unidade é a sessão, não a época.** Juntar as épocas de toda a gente num
  só saco daria intervalos de confiança estreitíssimos e falsos: as épocas
  dentro de uma pessoa não são independentes umas das outras. Cada sessão
  contribui **um** número — a sua variação — e a estatística corre sobre
  essas.
- **A consistência de sinal conta mais do que a magnitude.** Com meia dúzia
  de sessões, "cinco das seis subiram" é uma afirmação mais forte e mais
  robusta do que uma média com um intervalo enorme. O teste de sinal não
  assume distribuição nenhuma e é o que se reporta em primeiro lugar.
- **Sessões com sinal mau ficam de fora, e diz-se quantas.** Incluir uma
  gravação de elétrodos soltos não acrescenta ruído neutro: acrescenta
  variação inventada que puxa a mediana para onde calhar.

O que isto **não** é: uma demonstração de que o mantra causa alguma coisa.
Sem um braço de controlo — as mesmas pessoas, os mesmos minutos, em silêncio
— qualquer variação encontrada pode ser o efeito de estar sentado, quieto e
de olhos fechados durante seis minutos. :func:`aggregate` devolve esse aviso
com os dados, e não como nota de rodapé.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .analysis import AnalysisResult, baseline_spread, percent_change
from .config import Config
from .markers import Marker, markers


@dataclass(frozen=True)
class SessionEntry:
    """Uma sessão já analisada, pronta a entrar na agregação."""

    name: str
    device: str
    mantra: str
    duration_s: float
    coverage: float
    result: AnalysisResult
    #: ``id`` -> variação percentual, onde a percentagem significa alguma coisa.
    percent: dict[str, float] = field(default_factory=dict)
    #: ``id`` -> variação em desvios da oscilação do próprio repouso. Existe
    #: sempre que houve leitura, e é a escala que permite comparar marcadores
    #: de unidades diferentes.
    standardised: dict[str, float] = field(default_factory=dict)
    #: Marcadores descartados nesta sessao por denominador colapsado. Ver
    #: :data:`MAX_STANDARDISED`.
    degenerate: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        return bool(self.standardised)


@dataclass(frozen=True)
class MarkerAggregate:
    """O que uma medida fez ao longo das sessões todas."""

    marker: Marker
    n_sessions: int
    #: Mediana das variações percentuais, entre sessões.
    median_percent: float
    #: Mediana das variações padronizadas. É o que se desenha, porque põe os
    #: oito marcadores no mesmo eixo.
    median_std: float
    ci_low: float
    ci_high: float
    #: Quantas sessões subiram, das que tiveram leitura.
    n_up: int
    #: Probabilidade de ver esta unanimidade (ou mais) por acaso, se subir e
    #: descer fossem igualmente prováveis. Teste de sinal bilateral, exato.
    sign_p: float
    values_std: tuple[float, ...]
    values_percent: tuple[float, ...]

    @property
    def n_down(self) -> int:
        return self.n_sessions - self.n_up

    @property
    def direction(self) -> str:
        return "subiu" if self.median_std > 0 else "desceu"

    @property
    def consistent(self) -> bool:
        """O sentido repete-se mais do que o acaso explica.

        Com seis sessões, cinco no mesmo sentido dá p = 0,22 — não chega. Seis
        em seis dá 0,03. É um critério exigente de propósito: é a única coisa
        que este conjunto de dados consegue sustentar.
        """
        return self.sign_p < 0.05 and self.n_sessions >= 5


@dataclass(frozen=True)
class CohortResult:
    sessions: tuple[SessionEntry, ...]
    included: tuple[SessionEntry, ...]
    aggregates: tuple[MarkerAggregate, ...]
    min_coverage: float

    @property
    def n_degenerate(self) -> int:
        """Valores descartados por denominador colapsado, entre as incluidas."""
        return sum(len(s.degenerate) for s in self.included)

    @property
    def n_excluded(self) -> int:
        return len(self.sessions) - len(self.included)

    @property
    def devices(self) -> tuple[str, ...]:
        return tuple(sorted({s.device for s in self.included}))

    @property
    def mantras(self) -> tuple[str, ...]:
        return tuple(sorted({s.mantra for s in self.included if s.mantra}))


def _sign_test_p(n_up: int, n: int) -> float:
    """Teste de sinal bilateral exato. Sem aproximações, que n é pequeno."""
    if n == 0:
        return 1.0
    extreme = min(n_up, n - n_up)
    tail = sum(math.comb(n, k) for k in range(extreme + 1))
    return min(1.0, 2.0 * tail / (2.0**n))


def _median_ci(values: np.ndarray, draws: int = 4000, seed: int = 11) -> tuple[float, float]:
    """IC de 95 % da mediana, por reamostragem das **sessões**.

    Reamostrar sessões e não épocas é o que impede o intervalo de encolher
    artificialmente: a unidade independente é a pessoa que se sentou na
    cadeira, não o pedaço de quatro segundos.
    """
    values = values[np.isfinite(values)]
    if values.size < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(draws, values.size))
    medians = np.median(values[idx], axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


#: Mínimo de janelas de calibração para uma sessão contribuir com um
#: marcador. Abaixo disto a dispersão da referência assenta em duas ou três
#: janelas e é ela própria ruído — e como a variação padronizada divide por
#: essa dispersão, uma referência instável produz valores enormes. Sem este
#: piso, uma sessão com três janelas de coerência dava −75 desvios e arrastava
#: sozinha a mediana e o intervalo de toda a coorte.
MIN_BASELINE_EPOCHS = 5

#: Tecto para a variacao padronizada de uma sessao. Acima disto o que se esta
#: a medir nao e uma mudanca — e um denominador colapsado.
#:
#: Medido: a sessao 15_18324811092026 deu +1 092 060 desvios na Quietude
#: corporal, porque o SMR relativo na calibracao foi 2,9e-09 quando o
#: plausivel anda entre 0,01 e 0,10. Nove ordens de grandeza abaixo do
#: possivel: naquela banda nao havia sinal nenhum durante a calibracao.
#: Dividir por essa dispersao produz um numero enorme que arrasta sozinho a
#: mediana e o intervalo de toda a coorte.
#:
#: 20 e generoso de proposito: um efeito de EEG real anda entre 0,2 e 2
#: desvios da oscilacao propria, portanto nada verdadeiro se perde aqui.
#: E uma regra, aplicada a todos os marcadores por igual, e o numero de
#: valores que ela retira e reportado na pagina — nao e escolher a dedo o
#: ponto que incomoda.
MAX_STANDARDISED = 20.0


def summarise_session(
    name: str,
    device: str,
    mantra: str,
    duration_s: float,
    result: AnalysisResult,
) -> SessionEntry:
    """Reduz uma sessão a um número por marcador."""
    percent: dict[str, float] = {}
    standardised: dict[str, float] = {}
    dropped: list[str] = []
    for summary in result.summaries:
        if summary.marker.role != "marker":
            continue
        if not np.isfinite(summary.baseline) or not np.isfinite(summary.active):
            continue
        if summary.n_baseline < MIN_BASELINE_EPOCHS:
            continue
        spread = baseline_spread(result, summary.marker.id)
        if not np.isfinite(spread) or spread <= 0:
            continue
        change = float((summary.active - summary.baseline) / spread)
        if abs(change) > MAX_STANDARDISED:
            dropped.append(summary.marker.id)
            continue
        standardised[summary.marker.id] = change
        value = percent_change(result, summary)
        if np.isfinite(value):
            percent[summary.marker.id] = float(value)

    coverage = float(np.nanmean([s.coverage for s in result.summaries]))
    return SessionEntry(
        name=name,
        device=device,
        mantra=mantra,
        duration_s=duration_s,
        coverage=coverage,
        result=result,
        percent=percent,
        standardised=standardised,
        degenerate=tuple(dropped),
    )


def aggregate(
    sessions: Iterable[SessionEntry], cfg: Config, min_coverage: float | None = None
) -> CohortResult:
    """Junta as sessões utilizáveis e resume cada marcador."""
    all_sessions = tuple(sessions)
    floor = cfg.report.tier_full if min_coverage is None else min_coverage
    included = tuple(
        s for s in all_sessions if s.usable and s.coverage >= floor
    )

    out: list[MarkerAggregate] = []
    for marker in markers(role="marker"):
        std = np.array(
            [s.standardised[marker.id] for s in included if marker.id in s.standardised],
            dtype=float,
        )
        pct = np.array(
            [s.percent[marker.id] for s in included if marker.id in s.percent],
            dtype=float,
        )
        if std.size == 0:
            continue
        n_up = int((std > 0).sum())
        low, high = _median_ci(std)
        out.append(
            MarkerAggregate(
                marker=marker,
                n_sessions=int(std.size),
                median_percent=float(np.median(pct)) if pct.size else float("nan"),
                median_std=float(np.median(std)),
                ci_low=low,
                ci_high=high,
                n_up=n_up,
                sign_p=_sign_test_p(n_up, int(std.size)),
                values_std=tuple(float(v) for v in std),
                values_percent=tuple(float(v) for v in pct),
            )
        )

    out.sort(key=lambda a: abs(a.median_std), reverse=True)
    return CohortResult(
        sessions=all_sessions,
        included=included,
        aggregates=tuple(out),
        min_coverage=floor,
    )
