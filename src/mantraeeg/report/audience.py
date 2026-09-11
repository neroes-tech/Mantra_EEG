"""Preparação das séries para o ecrã do participante.

**Duas vistas dos mesmos dados, deliberadamente diferentes.**

A vista de investigação (``report/figures.py``) mostra buracos onde as épocas
foram rejeitadas, porque interpolar sobre uma época má é inventar sinal.

Esta vista, a que o participante vê, **une os pontos**. Não é um descuido: uma
linha partida a meio é lida por quem não é da área como "o meu cérebro parou"
ou "isto não funciona", e nenhuma das duas coisas é verdade. O que é rejeitado
são janelas em que o **sensor** não deu leitura fiável, não janelas em que o
cérebro não fez nada.

Três decisões que tornam a vista honesta apesar de interpolada:

- **Eixo comum.** Todas as métricas partilham a mesma janela temporal, a maior
  em que **todas** têm leitura boa. Comparar linhas que acabam em sítios
  diferentes seria pior do que interpolar.
- **Eixo Y em unidades da variabilidade da própria calibração** (mediana e MAD,
  como a SPEC 3.4 prescreve). A percentagem não serve aqui: para o ALAY, que é
  uma quantidade com sinal centrada perto de zero, dividir pela linha de base
  produzia amplitudes de -794 % a +2076 % — o gráfico ficava ilegível e a
  escala não significava nada. Normalizar pela dispersão põe todas as métricas
  no mesmo eixo e responde à pergunta que interessa: "isto está acima ou abaixo
  do meu normal, e por quanto?".
- **Suavização ligeira.** A série época a época salta muito; uma mediana móvel
  de poucas épocas deixa a tendência legível sem deslocar nada. Está declarada
  no rodapé.
- **A cobertura viaja com os dados.** Quanto foi interpolado fica registado, e
  o relatório di-lo. Não se esconde, arredonda-se a apresentação.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..analysis import AnalysisResult, MarkerSeries

#: Marcadores em que a variação percentual não faz sentido: quantidades com
#: sinal e centradas perto de zero, onde "+57 %" significa "57 % mais negativo"
#: e lê-se ao contrário do que é.
SIGNED_MARKERS = frozenset({"faa", "wpli_alpha1", "imcoh_alpha1"})


#: Mediana móvel aplicada à série antes de desenhar, em épocas. Não desloca a
#: linha; só tira o salto época a época que torna o gráfico ilegível.
SMOOTH_EPOCHS = 5

#: Torna o MAD comparável a um desvio padrão (SPEC 3.4).
MAD_TO_SD = 1.4826


@dataclass(frozen=True)
class AudienceSeries:
    """Uma métrica pronta a desenhar: eixo comum, sem buracos."""

    marker_id: str
    label: str
    times_s: np.ndarray
    #: Desvio face à calibração, em unidades da variabilidade dela própria.
    #: É isto que permite pôr métricas de unidades diferentes no mesmo eixo.
    deviation: np.ndarray
    baseline: float
    #: Fração de pontos que foram interpolados por a leitura não ser fiável.
    interpolated_fraction: float


@dataclass(frozen=True)
class AudienceData:
    series: list[AudienceSeries]
    t_start: float
    t_end: float
    #: Instante em que a meditação começou, para a marca no gráfico.
    mantra_start_s: float | None
    coverage: float

    def by_id(self, marker_id: str) -> AudienceSeries | None:
        return next((s for s in self.series if s.marker_id == marker_id), None)


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    """Mediana móvel. Robusta, e não arrasta os saltos como a média faria."""
    if width <= 1 or values.size < width:
        return values
    half = width // 2
    padded = np.pad(values, half, mode="edge")
    return np.array(
        [np.median(padded[i : i + width]) for i in range(values.size)]
    )


def _resample(series: MarkerSeries, grid: np.ndarray) -> np.ndarray:
    """Põe uma série na grelha comum, sem inventar fora do seu domínio.

    Os marcadores de conectividade correm a 10 s com passo de 5 s, contra 4 s
    e 2 s dos de potência: sem isto não podiam partilhar eixo.
    """
    finite = np.isfinite(series.values)
    if finite.sum() < 2:
        return np.full(grid.shape, np.nan)
    return np.interp(
        grid,
        series.times_s[finite],
        series.values[finite],
        left=np.nan,
        right=np.nan,
    )


def prepare(
    result: AnalysisResult,
    marker_ids: list[str],
    phases: list[tuple[str, float, float]],
    baseline_phase: str = "CALIBRATION",
) -> AudienceData:
    """Constrói o conjunto de séries para o ecrã final.

    O eixo temporal é o maior intervalo em que **todas** as métricas pedidas
    têm leitura, para as linhas serem comparáveis entre si.
    """
    chosen = [s for s in result.series if s.marker_id_matches(marker_ids)]
    if not chosen:
        return AudienceData([], 0.0, 0.0, None, 0.0)

    # Grelha comum: a mais fina das séries escolhidas.
    finest = min(
        chosen,
        key=lambda s: (
            np.median(np.diff(s.times_s)) if s.times_s.size > 1 else np.inf
        ),
    )
    grid = finest.times_s

    resampled = {s.marker.id: _resample(s, grid) for s in chosen}

    # Janela comum: onde TODAS têm leitura. É isto que dá o mesmo eixo a todos
    # os gráficos, que é o que torna as linhas comparáveis.
    common = np.ones(grid.shape, dtype=bool)
    for values in resampled.values():
        common &= np.isfinite(values)
    if not common.any():
        # Nenhum instante com todas boas: cai para a união, e diz-se.
        common = np.zeros(grid.shape, dtype=bool)
        for values in resampled.values():
            common |= np.isfinite(values)
    if not common.any():
        return AudienceData([], 0.0, 0.0, None, 0.0)

    indices = np.nonzero(common)[0]
    span = slice(indices[0], indices[-1] + 1)
    times = grid[span]

    mantra_start = next(
        (start for name, start, _ in phases if name == "MANTRA"), None
    )

    out: list[AudienceSeries] = []
    total_interpolated = 0.0
    for series in chosen:
        values = resampled[series.marker.id][span]
        finite = np.isfinite(values)
        if finite.sum() < 2:
            continue
        # UNIR OS PONTOS: sem buracos na linha. O que se perdeu foi leitura do
        # sensor, não atividade cerebral, e uma linha partida diria a coisa
        # errada a quem não é da área.
        filled = np.interp(times, times[finite], values[finite])
        interpolated = 1.0 - float(finite.mean())
        total_interpolated += interpolated

        filled = _smooth(filled, SMOOTH_EPOCHS)

        # Referência: mediana e MAD das épocas de calibração desta métrica.
        # É a normalização da SPEC 3.4, e é a única que funciona para todas as
        # métricas ao mesmo tempo — incluindo as que têm sinal e vivem perto de
        # zero, onde a percentagem explode.
        base_values = series.of_phase(baseline_phase)
        baseline = float(np.median(base_values)) if base_values.size else np.nan
        if base_values.size >= 3:
            mad = float(np.median(np.abs(base_values - baseline))) * MAD_TO_SD
        else:
            mad = np.nan
        if not np.isfinite(mad) or mad <= 0:
            # MAD degenerado: cai para a dispersão de toda a série, e se nem
            # isso houver não se desenha nada em vez de inventar uma escala.
            spread = float(np.std(filled))
            mad = spread if spread > 0 else np.nan
        if not np.isfinite(baseline) or not np.isfinite(mad):
            continue

        deviation = (filled - baseline) / mad

        out.append(
            AudienceSeries(
                marker_id=series.marker.id,
                label=series.marker.friendly or series.marker.label,
                times_s=times,
                deviation=deviation,
                baseline=float(baseline),
                interpolated_fraction=interpolated,
            )
        )

    coverage = 1.0 - (total_interpolated / len(out)) if out else 0.0
    return AudienceData(
        series=out,
        t_start=float(times[0]),
        t_end=float(times[-1]),
        mantra_start_s=mantra_start,
        coverage=coverage,
    )
