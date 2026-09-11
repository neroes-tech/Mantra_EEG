"""Inspetor de sinal ao vivo, com marcação de segmentos. O painel do CTRL+I.

Mostra os traçados dos oito canais, o semáforo por elétrodo com o motivo, e o
veredicto global de contacto. É a ferramenta para colocar o equipamento na
cabeça e iterar até o sinal estar bom, sem correr um script em lote de cada vez.

Permite **marcar segmentos** com o teclado durante a gravação. É isso que torna
possível validar o pré-processamento contra a realidade: gravam-se dois estados
conhecidos — tipicamente olhos abertos e olhos fechados — e pergunta-se ao
pipeline se consegue distingui-los (``scripts/compare_segments.py``). O efeito
de Berger é o árbitro final da qualidade do sinal, e vale mais do que qualquer
semáforo: um canal amarelo que mostra o alfa a subir ao fechar os olhos está a
medir cérebro; um canal verde que não mostra nada, provavelmente não.

Este widget é o mesmo que o M4 embebe na consola do operador.

**Nada aqui é um biomarcador.** É vigilância de elétrodos (SPEC 1.4), e nunca
aparece ao participante nada sobre o seu cérebro.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets
from scipy import signal as sp_signal

from ..acquisition import Acquisition, ContactReport
from ..config import Config

LEVEL_COLOUR = {"green": "#2e9b57", "yellow": "#c99a1e", "red": "#c0392b"}
LEVEL_LABEL = {"green": "OK", "yellow": "MARGINAL", "red": "MAU"}


class SignalInspector(QtWidgets.QWidget):
    """Traçados ao vivo, qualidade por elétrodo e marcação de segmentos."""

    def __init__(
        self,
        cfg: Config,
        acquisition: Acquisition,
        recorder: object | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._cfg = cfg
        self._acq = acquisition
        self._recorder = recorder
        self._sfreq = cfg.device.sfreq_nominal
        self._trace_s = cfg.ui.inspector_trace_s
        self._names = [cfg.montage.channels[i] for i in sorted(cfg.montage.channels)]

        self._current_mark: str | None = None
        self._mark_started_s = 0.0
        self._marks: list[tuple[float, str]] = []

        self.setWindowTitle("mantra-eeg — inspetor de sinal (CTRL+I)")
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self._build()

        self._trace_timer = QtCore.QTimer(self)
        self._trace_timer.timeout.connect(self._refresh_traces)
        self._trace_timer.start(int(cfg.ui.inspector_trace_period_s * 1000))

        self._quality_timer = QtCore.QTimer(self)
        self._quality_timer.timeout.connect(self._refresh_quality)
        self._quality_timer.start(int(cfg.contact_monitor.update_period_s * 1000))

    # -- construção ---------------------------------------------------------- #
    def _build(self) -> None:
        pg.setConfigOptions(antialias=False, background="#12141a", foreground="#c8ccd4")
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        self._plots = pg.GraphicsLayoutWidget()
        layout.addWidget(self._plots, stretch=3)

        self._curves: list[pg.PlotDataItem] = []
        self._axes: list[pg.PlotItem] = []
        for row, name in enumerate(self._names):
            ax = self._plots.addPlot(row=row, col=0)
            ax.setMenuEnabled(False)
            ax.hideButtons()
            ax.showAxis("bottom", row == len(self._names) - 1)
            ax.setLabel("left", name)
            ax.setMouseEnabled(x=False, y=False)
            if row:
                ax.setXLink(self._axes[0])
            self._axes.append(ax)
            self._curves.append(ax.plot(pen=pg.mkPen("#7fb2ff", width=1)))
        self._axes[-1].setLabel("bottom", "segundos")

        side = QtWidgets.QVBoxLayout()
        side.setSpacing(8)
        layout.addLayout(side, stretch=2)

        self._banner = QtWidgets.QLabel()
        self._banner.setWordWrap(True)
        self._banner.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._banner.setStyleSheet(
            "padding:10px; border-radius:6px; background:#1c1f27; color:#c8ccd4;"
        )
        side.addWidget(self._banner)

        self._mark_banner = QtWidgets.QLabel()
        self._mark_banner.setWordWrap(True)
        self._mark_banner.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._mark_banner.setStyleSheet(
            "padding:10px; border-radius:6px; background:#1a2233; color:#cfe0ff;"
        )
        side.addWidget(self._mark_banner)
        self._render_mark_banner()

        self._table = QtWidgets.QTableWidget(len(self._names), 6)
        self._table.setHorizontalHeaderLabels(
            ["canal", "sd µV", "p2p µV", "rede", "1/f", "estado"]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        self._table.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self._table.setStyleSheet(
            "QTableWidget{background:#1c1f27; color:#c8ccd4; gridline-color:#2b2f3a;}"
            "QHeaderView::section{background:#232733; color:#9aa3b2; border:0;"
            "padding:4px;}"
        )
        side.addWidget(self._table, stretch=1)

        self._reasons = QtWidgets.QTextEdit()
        self._reasons.setReadOnly(True)
        self._reasons.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._reasons.setStyleSheet(
            "background:#1c1f27; color:#9aa3b2; border:0; padding:6px;"
        )
        self._reasons.setMaximumHeight(120)
        side.addWidget(self._reasons)

        keys = "   ".join(f"[{k}] {v}" for k, v in self._cfg.ui.marker_keys.items())
        self._footer = QtWidgets.QLabel(
            f"{keys}\n\nVigilância de elétrodos — não são biomarcadores e nada "
            f"disto é mostrado ao participante."
        )
        self._footer.setWordWrap(True)
        self._footer.setStyleSheet("color:#6b7280; font-size:11px;")
        side.addWidget(self._footer)

    # -- marcação ------------------------------------------------------------ #
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        label = self._cfg.ui.marker_keys.get(event.text())
        if label is not None:
            self._mark(label)
            return
        if event.key() in (QtCore.Qt.Key.Key_Space, QtCore.Qt.Key.Key_Return):
            self._mark(None)
            return
        super().keyPressEvent(event)

    def _mark(self, label: str | None) -> None:
        """Fecha o segmento atual e, se houver rótulo, abre um novo."""
        t = self._acq.elapsed_s
        if self._recorder is not None:
            self._recorder.add_event(
                "mark_end" if label is None else "mark",
                t_s=t,
                detail=label or (self._current_mark or ""),
                sample=int(round(t * self._sfreq)),
            )
        if label is not None:
            self._marks.append((t, label))
            self._current_mark = label
            self._mark_started_s = t
        else:
            self._current_mark = None
        self._render_mark_banner()

    def _render_mark_banner(self) -> None:
        if self._recorder is None:
            self._mark_banner.setText(
                "<b>Sem gravação</b><br>"
                "<span style='color:#8fa0bf'>Use <code>--record</code> para as "
                "marcações ficarem guardadas.</span>"
            )
            return
        if self._current_mark is None:
            done = (
                f"<br><span style='color:#8fa0bf'>{len(self._marks)} segmentos "
                f"marcados</span>"
                if self._marks
                else ""
            )
            self._mark_banner.setText(
                f"<b>Sem segmento aberto</b> — prima uma tecla para marcar{done}"
            )
            return
        elapsed = self._acq.elapsed_s - self._mark_started_s
        self._mark_banner.setText(
            f"<b style='font-size:15px'>▶ {self._current_mark}</b>  "
            f"<span style='font-size:15px'>{elapsed:5.1f} s</span>"
            f"<br><span style='color:#8fa0bf'>{len(self._marks)} segmentos "
            f"marcados · Espaço para fechar</span>"
        )

    # -- atualização --------------------------------------------------------- #
    def _refresh_traces(self) -> None:
        window = self._acq.peek_last(self._trace_s)
        if window.shape[1] < 2:
            return
        eeg = np.asarray(window[0:8], dtype=np.float64)
        # Detrend por canal: os offsets de eletrodo são de centenas de mV e sem
        # isto os traçados seriam linhas rectas fora do ecrã.
        eeg = sp_signal.detrend(eeg, axis=-1, type="linear")
        t = np.arange(eeg.shape[1]) / self._sfreq

        for i, curve in enumerate(self._curves):
            curve.setData(t, eeg[i])
            span = float(np.percentile(np.abs(eeg[i]), 99.5)) or 1.0
            self._axes[i].setYRange(-span * 1.2, span * 1.2, padding=0)

    def _refresh_quality(self) -> None:
        self._render_mark_banner()
        report = self._acq.contact_report()
        if not report.channels:
            self._banner.setText(
                "<b>A acumular dados…</b><br>"
                f"<span style='color:#9aa3b2'>{report.global_warning or ''}</span>"
            )
            return
        self._render_banner(report)
        self._render_table(report)

    def _render_banner(self, report: ContactReport) -> None:
        needed = self._cfg.montage.min_good_channels_to_start
        if report.global_warning:
            self._banner.setStyleSheet(
                "padding:10px; border-radius:6px; background:#3b1d1b; color:#f2b8b5;"
            )
            self._banner.setText(f"<b>SEM CONTACTO</b><br>{report.global_warning}")
            return
        ok = report.n_good >= needed
        self._banner.setStyleSheet(
            "padding:10px; border-radius:6px; color:#dfe4ec; background:"
            + ("#17331f" if ok else "#33291a")
        )
        text = (
            f"<b>{'PRONTO' if ok else 'AINDA NÃO'}</b> — {report.n_good} de "
            f"{len(report.channels)} canais utilizáveis (mínimo {needed})"
        )
        if report.line_warning:
            text += f"<br><span style='color:#e0c07a'>{report.line_warning}</span>"
        self._banner.setText(text)

    def _render_table(self, report: ContactReport) -> None:
        reasons: list[str] = []
        for row, ch in enumerate(report.channels):
            values = [
                ch.name,
                f"{ch.std_uv:.1f}",
                f"{ch.p2p_uv:.0f}",
                f"{ch.line_rel:.1f}",
                f"{ch.slope:+.2f}" if np.isfinite(ch.slope) else "—",
                LEVEL_LABEL[ch.level],
            ]
            for col, text in enumerate(values):
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                if col == len(values) - 1:
                    item.setForeground(QtGui.QColor(LEVEL_COLOUR[ch.level]))
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self._table.setItem(row, col, item)
            if ch.reason != "ok":
                reasons.append(f"<b>{ch.name}</b>: {ch.reason}")
        self._reasons.setHtml("<br>".join(reasons) if reasons else "Todos os canais ok.")

    # -- ciclo de vida ------------------------------------------------------- #
    @property
    def marks(self) -> list[tuple[float, str]]:
        return list(self._marks)

    def stop(self) -> None:
        self._trace_timer.stop()
        self._quality_timer.stop()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self.stop()
        super().closeEvent(event)
