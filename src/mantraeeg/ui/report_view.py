"""Página final: o que o participante vê.

Dois gráficos lado a lado.

**Esquerda, evolução no tempo.** As métricas seleccionadas em cima, cada uma
com a sua cor; clicar liga e desliga linhas e o eixo Y reajusta-se. O eixo Y
mede o desvio face à variabilidade da **própria calibração** da pessoa
(mediana e MAD, SPEC 3.4): é a única escala em que métricas de unidades
diferentes se podem sobrepor, e a única que funciona para o ALAY, que tem sinal
e vive perto de zero.

**Direita, o resumo.** Barras horizontais com todas as métricas seleccionadas,
comparadas com a linha de base.

As linhas **não têm buracos**, ao contrário da vista de investigação. Ver
``report/audience.py``: o que se perde são janelas em que o sensor não deu
leitura fiável, não janelas em que o cérebro parou, e uma linha partida diria a
coisa errada a quem não é da área. Quanto foi interpolado aparece no rodapé.

O que se escreve nunca inventa melhoria: quando o intervalo de confiança cruza
zero, diz "estável" (SPEC 5.1).
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from ..analysis import AnalysisResult, baseline_spread, percent_change
from ..config import Config
from ..report.audience import AudienceData, prepare

FG = "#eef1f7"
MUTED = "#8b95a7"
PANEL = "#11141b"
GRID = "#1e2431"
# Laranja, verde e azul-bebe primeiro: sao as tres do destaque, e tem de se
# distinguir num relance sem consultar a legenda. As restantes continuam a
# afastar-se no circulo cromatico, porque o botao "Adicionar" do painel pode
# por os catorze marcadores no mesmo grafico.
SERIES_COLOURS = (
    "#e8944d", "#5bbc99", "#7fcbe8", "#c9a0dc", "#e8c24d", "#e0796f",
    "#6fd8c4", "#9dbe5b", "#b08fe0", "#f0a6c8", "#5b8ad4", "#d4a455",
    "#8fd6a0", "#7fb2ff",
)


class MetricToggle(QtWidgets.QPushButton):
    """Pastilha clicável que liga e desliga uma linha do gráfico."""

    def __init__(self, marker_id: str, label: str, colour: str) -> None:
        super().__init__(label)
        self.marker_id = marker_id
        self.colour = colour
        self.setCheckable(True)
        self.setChecked(True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(38)
        self._restyle()
        self.toggled.connect(lambda _: self._restyle())

    def _restyle(self) -> None:
        if self.isChecked():
            style = (
                f"QPushButton{{background:{self.colour}; color:#07080b;"
                f"border:2px solid {self.colour}; border-radius:19px;"
                f"padding:0 20px; font-size:15px; font-weight:600;}}"
            )
        else:
            style = (
                f"QPushButton{{background:transparent; color:{MUTED};"
                f"border:2px solid #2a3040; border-radius:19px;"
                f"padding:0 20px; font-size:15px;}}"
                f"QPushButton:hover{{border-color:{self.colour};"
                f"color:{self.colour};}}"
            )
        self.setStyleSheet(style)


class ResultsView(QtWidgets.QWidget):
    """Os dois gráficos do ecrã final."""

    def __init__(self, cfg: Config, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._cfg = cfg
        self._data: AudienceData | None = None
        self._result: AnalysisResult | None = None
        self._toggles: dict[str, MetricToggle] = {}
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._build()

    # -- construção ------------------------------------------------------------ #
    def _build(self) -> None:
        pg.setConfigOptions(antialias=True, background=PANEL, foreground=MUTED)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        # Grelha e nao linha: com os 14 marcadores selecionados, uma linha
        # so transbordava do ecra.
        self._toggle_row = QtWidgets.QGridLayout()
        self._toggle_row.setSpacing(8)
        root.addLayout(self._toggle_row)

        charts = QtWidgets.QHBoxLayout()
        charts.setSpacing(14)
        root.addLayout(charts, stretch=1)

        self._plot = pg.PlotWidget()
        self._plot.setBackground(PANEL)
        self._plot.showGrid(x=True, y=True, alpha=0.15)
        self._plot.setLabel("left", "face ao seu normal")
        self._plot.setLabel("bottom", "tempo de sessão", units="s")
        self._plot.getPlotItem().getViewBox().setMouseEnabled(x=False, y=False)
        self._plot.addLine(y=0, pen=pg.mkPen("#3a4152", width=1))
        charts.addWidget(self._plot, stretch=3)

        self._bars = pg.PlotWidget()
        self._bars.setBackground(PANEL)
        self._bars.showGrid(x=True, alpha=0.15)
        self._bars.setLabel("bottom", "variação face à calibração", units="%")
        self._bars.getPlotItem().getViewBox().setMouseEnabled(x=False, y=False)
        charts.addWidget(self._bars, stretch=2)

        self._empty = QtWidgets.QLabel()
        self._empty.setWordWrap(True)
        self._empty.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._empty.setStyleSheet(
            "background:#11141b; border:1px solid #2a3142; border-radius:12px;"
            "color:#a7b0c0; font-size:15px; padding:34px;"
        )
        self._empty.setVisible(False)
        charts.addWidget(self._empty, stretch=3)

        self._footer = QtWidgets.QLabel()
        self._footer.setWordWrap(True)
        self._footer.setStyleSheet(f"color:#5d6779; font-size:12px;")
        root.addWidget(self._footer)

    # -- dados -------------------------------------------------------------------- #
    def show_results(
        self,
        result: AnalysisResult,
        phases: list[tuple[str, float, float]],
        marker_ids: list[str],
    ) -> None:
        self._result = result
        self._data = prepare(result, marker_ids, phases)
        self._rebuild_toggles(marker_ids)
        self._redraw()
        self._refresh_empty_state(result)

    def _refresh_empty_state(self, result: AnalysisResult) -> None:
        """Sem séries não se desenha nada — mas tem de se dizer porquê.

        Numa sessão real (headset 14) **nenhuma** das 46 épocas passou no
        controlo de qualidade, e o ecrã final ficou simplesmente em branco: o
        operador não tinha como distinguir "a análise falhou" de "o sinal não
        prestou". São coisas diferentes e levam a ações diferentes.
        """
        has_series = bool(self._data and self._data.series)
        self._plot.setVisible(has_series)
        self._bars.setVisible(has_series)
        if has_series:
            self._empty.setVisible(False)
            return

        import numpy as np_

        quality = result.quality
        total = int(quality.channel_ok.shape[1]) if quality.channel_ok.size else 0
        rows = [
            quality.name_to_row[n]
            for n in self._cfg.montage.metric_channels
            if n in quality.name_to_row
        ]
        good = (
            int(quality.channel_ok[rows].all(axis=0).sum())
            if rows and total
            else 0
        )
        sd = (
            float(np_.median(np_.std(result.preprocessed.clean, axis=1)))
            if result.preprocessed is not None
            else float("nan")
        )

        reason = (
            f"Nenhuma das {total} janelas de sinal passou no controlo de "
            f"qualidade ({good} boas)."
        )
        if np_.isfinite(sd):
            # O separador de milhares formatado a parte: um .replace(",", " ")
            # sobre a frase inteira comia tambem a virgula de "50 uV, portanto".
            amplitude = f"{sd:,.0f}".replace(",", " ")
            reason += (
                f" A amplitude mediana foi de {amplitude} µV — o EEG plausível "
                f"anda entre 5 e 50 µV, portanto o que entrou não é atividade "
                f"cerebral."
            )
        self._empty.setText(
            "Não há gráficos para mostrar.\n\n"
            + reason
            + "\n\nA gravação está guardada na mesma. Causas habituais: "
            "elétrodos sem contacto com o couro cabeludo, ou ruído elétrico "
            "do ambiente. Ver o contacto por elétrodo em Ctrl+I."
        )
        self._empty.setVisible(True)

    def _rebuild_toggles(self, marker_ids: list[str]) -> None:
        while self._toggle_row.count():
            item = self._toggle_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._toggles.clear()

        if self._data is None:
            return
        columns = 5 if len(self._data.series) > 8 else 4
        for i, series in enumerate(self._data.series):
            colour = SERIES_COLOURS[i % len(SERIES_COLOURS)]
            toggle = MetricToggle(series.marker_id, series.label, colour)
            toggle.toggled.connect(lambda _: self._redraw())
            self._toggles[series.marker_id] = toggle
            self._toggle_row.addWidget(toggle, i // columns, i % columns)

    def _redraw(self) -> None:
        self._plot.clear()
        self._plot.addLine(y=0, pen=pg.mkPen("#3a4152", width=1))
        self._curves.clear()
        if self._data is None or self._result is None:
            return

        # Marca do início da meditação, para se ver onde o mantra entrou.
        if self._data.mantra_start_s is not None:
            marker = pg.InfiniteLine(
                pos=self._data.mantra_start_s,
                angle=90,
                pen=pg.mkPen("#4b5a74", width=1, style=QtCore.Qt.PenStyle.DashLine),
                label="mantra",
                labelOpts={"color": MUTED, "position": 0.95, "movable": False},
            )
            self._plot.addItem(marker)

        active = []
        for i, series in enumerate(self._data.series):
            toggle = self._toggles.get(series.marker_id)
            if toggle is None or not toggle.isChecked():
                continue
            colour = SERIES_COLOURS[i % len(SERIES_COLOURS)]
            curve = self._plot.plot(
                series.times_s,
                series.deviation,
                pen=pg.mkPen(colour, width=2.5),
                name=series.label,
            )
            self._curves[series.marker_id] = curve
            active.append(series)

        if active:
            # Y automático sobre o que está seleccionado, com folga.
            values = np.concatenate([s.deviation for s in active])
            values = values[np.isfinite(values)]
            if values.size:
                span = max(float(np.ptp(values)), 1.0)
                self._plot.setYRange(
                    float(values.min()) - span * 0.15,
                    float(values.max()) + span * 0.15,
                )
            self._plot.setXRange(self._data.t_start, self._data.t_end, padding=0.02)

        self._redraw_bars()
        self._redraw_footer()

    def _redraw_bars(self) -> None:
        self._bars.clear()
        if self._result is None or self._data is None:
            return
        shown = [
            s
            for s in self._data.series
            if self._toggles.get(s.marker_id) and self._toggles[s.marker_id].isChecked()
        ]
        if not shown:
            return

        order = [s.marker_id for s in self._data.series]

        def colour_of(marker_id: str) -> str:
            return SERIES_COLOURS[order.index(marker_id) % len(SERIES_COLOURS)]

        def collect(use_percent: bool):
            """As barras numa das duas escalas. Ver o fallback abaixo."""
            labels, values, brushes = [], [], []
            for series in shown:
                summary = self._result.summary_of(series.marker_id)
                if summary is None:
                    continue
                if use_percent:
                    # A mesma regua do relatorio. Antes punha-se aqui o delta
                    # absoluto num eixo rotulado "%": o Controlo emocional
                    # aparecia a ~0 no ecra e a -46 % no relatorio.
                    value = percent_change(self._result, summary)
                    if not np.isfinite(value):
                        continue
                    # Uma barra desproporcionada achata as outras contra o eixo.
                    if abs(value) > self._cfg.report.large_pct:
                        continue
                else:
                    spread = baseline_spread(self._result, series.marker_id)
                    if not np.isfinite(spread) or spread <= 0:
                        continue
                    value = (summary.active - summary.baseline) / spread
                    if not np.isfinite(value):
                        continue
                labels.append(series.label)
                values.append(float(value))
                # Apagado quando o IC cruza zero: nunca anunciar o que nao se
                # sustenta.
                brushes.append(
                    colour_of(series.marker_id) if summary.reliable else "#39404f"
                )
            return labels, values, brushes

        labels, values, brushes = collect(use_percent=True)
        unit = "%"
        if not labels:
            # Nenhuma percentagem utilizavel — quantidades com sinal, linhas de
            # base perto de zero, ou variacoes desproporcionadas. Em vez de um
            # painel vazio a seguir a uma sessao de oito minutos, muda-se de
            # escala: a variacao em desvios da oscilacao do proprio repouso
            # existe sempre que houve leitura, e o eixo diz o que e.
            labels, values, brushes = collect(use_percent=False)
            unit = "desvios"
        self._bars.setLabel(
            "bottom", "variação face à calibração", units=unit
        )
        if not labels:
            return

        positions = np.arange(len(labels))
        for pos, value, brush in zip(positions, values, brushes):
            bar = pg.BarGraphItem(
                x0=0, y=[pos], height=0.55, width=[value], brush=brush, pen=None
            )
            self._bars.addItem(bar)
        self._bars.addLine(x=0, pen=pg.mkPen("#3a4152", width=1))
        axis = self._bars.getAxis("left")
        axis.setTicks([[(p, l) for p, l in zip(positions, labels)]])
        self._bars.setYRange(-0.7, len(labels) - 0.3)

    def _redraw_footer(self) -> None:
        if self._data is None:
            self._footer.setText("")
            return
        stable = []
        if self._result is not None:
            stable = [
                s.marker.friendly or s.marker.label
                for s in self._result.summaries
                if s.marker.id in {x.marker_id for x in self._data.series}
                and not s.reliable
            ]
        parts = [
            f"Qualidade de leitura: {self._data.coverage * 100:.0f} % das janelas.",
            "As linhas estão suavizadas e unidas nas janelas sem leitura fiável.",
        ]
        if stable:
            parts.append(
                "Sem variação fiável (barra apagada): " + ", ".join(stable) + "."
            )
        parts.append(self._cfg.report.participant_footer)
        self._footer.setText("  ".join(parts))

    def clear(self) -> None:
        self._plot.clear()
        self._bars.clear()
        self._footer.setText("")
        while self._toggle_row.count() > 1:
            item = self._toggle_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._toggles.clear()
        self._data = None
        self._result = None
