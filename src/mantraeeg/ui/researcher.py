"""Painel do investigador (Ctrl+I). Abre por cima de qualquer ecrã.

Não há janela de operador permanente: a banca corre num só monitor e o
participante não pode ver diagnósticos. Este painel é chamado, consultado e
fechado.

Mostra três coisas: **estado do dispositivo** (é o que responde a "o headset
está ligado?" depois de horas sem ninguém na banca), o **sinal ao vivo**, e as
**durações** de cada fase.

Nada aqui é um biomarcador. É vigilância de elétrodos (SPEC 1.4).
"""

from __future__ import annotations

from dataclasses import replace

from PyQt6 import QtCore, QtGui, QtWidgets

from ..config import Config
from ..device import DeviceManager, DeviceState
from ..markers import markers
from ..drive import DriveClient, DriveError
from ..storage import UploadQueue
from ..session import SessionPlan
from .inspector import SignalInspector
from .screens import DurationPicker

PANEL = "#141821"
TEXT = "#c8ccd4"
DIM = "#8a93a3"

STATE_COLOUR = {
    DeviceState.DISCONNECTED: "#c0392b",
    DeviceState.CONNECTING: "#c99a1e",
    DeviceState.READY: "#8fd6a0",
    DeviceState.STREAMING: "#2e9b57",
    DeviceState.STALE: "#c0392b",
    DeviceState.ERROR: "#c0392b",
}


class ResearcherPanel(QtWidgets.QDialog):
    """Diagnóstico e configuração, sobreposto ao ecrã da banca."""

    plan_changed = QtCore.pyqtSignal(object)
    selection_changed = QtCore.pyqtSignal(dict)
    #: Os marcadores marcados, para irem para o ecrã final agora. É um botão
    #: e não um efeito da própria marcação: marcar catorze caixas uma a uma
    #: redesenhava o gráfico catorze vezes à frente do participante.
    apply_requested = QtCore.pyqtSignal(list)
    reconnect_requested = QtCore.pyqtSignal()

    def __init__(
        self,
        cfg: Config,
        device: DeviceManager,
        plan: SessionPlan,
        parent: QtWidgets.QWidget | None = None,
        drive: DriveClient | None = None,
        uploads: UploadQueue | None = None,
    ) -> None:
        super().__init__(parent)
        self._cfg = cfg
        self._device = device
        self._plan = plan
        self._drive = drive
        self._uploads = uploads
        self.setWindowTitle("mantra-eeg — painel do investigador")
        self.setStyleSheet(f"background:#0b0d12; color:{TEXT};")
        self.setModal(False)
        self._inspector: SignalInspector | None = None
        self._build()

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh_status)
        self._timer.start(500)
        self._refresh_status()

    # -- construção ----------------------------------------------------------- #
    def _build(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        self._status = QtWidgets.QLabel()
        self._status.setWordWrap(True)
        self._status.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._status.setStyleSheet(
            f"background:{PANEL}; border-radius:8px; padding:14px; font-size:15px;"
        )
        layout.addWidget(self._status)

        buttons = QtWidgets.QHBoxLayout()
        self._reconnect = QtWidgets.QPushButton("Ligar / reiniciar dispositivo")
        self._reconnect_idle_style = (
            "QPushButton{background:#7fb2ff; color:#0b0d12; border:2px solid "
            "#7fb2ff; border-radius:8px; padding:10px 18px; font-weight:600;}"
            "QPushButton:hover{background:#a8caff;}"
        )
        # Enquanto procura: fundo apagado com contorno azul, para se ver que ha
        # trabalho a decorrer. Sem isto o clique nao dava sinal nenhum.
        self._reconnect_busy_style = (
            "QPushButton{background:#0b0d12; color:#7fb2ff; border:2px solid "
            "#7fb2ff; border-radius:8px; padding:10px 18px; font-weight:600;}"
        )
        self._reconnect.setStyleSheet(self._reconnect_idle_style)
        self._reconnect.clicked.connect(self._on_reconnect_clicked)
        buttons.addWidget(self._reconnect)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        layout.addWidget(self._durations_box())
        layout.addWidget(self._mantra_box())
        layout.addWidget(self._markers_box())
        layout.addWidget(self._drive_box())

        self._signal_holder = QtWidgets.QVBoxLayout()
        layout.addLayout(self._signal_holder, stretch=1)
        self._placeholder = QtWidgets.QLabel(
            "Sem dispositivo a adquirir — o sinal aparece quando houver aquisição."
        )
        self._placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(
            f"background:{PANEL}; border-radius:8px; color:{DIM}; padding:60px;"
        )
        self._signal_holder.addWidget(self._placeholder)

    def _durations_box(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Durações das fases")
        box.setStyleSheet(
            f"QGroupBox{{background:{PANEL}; border-radius:8px; padding:14px;"
            f"margin-top:8px; color:{DIM};}}"
            "QGroupBox::title{subcontrol-origin:margin; left:12px;}"
        )
        row = QtWidgets.QHBoxLayout(box)
        self._spins: dict[str, DurationPicker] = {}
        for key, text, value in (
            ("calibration", "calibração", self._plan.calibration_s),
            ("mantra", "mantra", self._plan.mantra_s),
            ("settle", "repouso", self._plan.settle_s),
        ):
            column = QtWidgets.QVBoxLayout()
            column.addWidget(QtWidgets.QLabel(text))
            # O mesmo seletor do ecrã de explicação, pela mesma razão: com
            # folha de estilos, a seta de cima do QSpinBox deixa de responder.
            spin = DurationPicker(int(value), compact=True)
            spin.valueChanged.connect(lambda _: self._emit_plan())
            self._spins[key] = spin
            column.addWidget(spin)
            row.addLayout(column)

        self._adequacy = QtWidgets.QLabel()
        self._adequacy.setWordWrap(True)
        self._adequacy.setStyleSheet(
            f"color:{DIM}; font-size:11px; font-family:'Consolas',monospace;"
        )
        row.addWidget(self._adequacy, stretch=1)
        self._refresh_adequacy()
        return box

    def _mantra_box(self) -> QtWidgets.QWidget:
        """Qual dos mantras toca nesta sessão.

        A escolha viaja no ``SessionPlan`` e acaba em ``raw_meta.json``: o
        relatório nomeia o que tocou, não o default da config.

        A reprodução é em ciclo infinito, portanto um mantra mais curto do que
        a fase repete em vez de deixar a pessoa em silêncio a meio — o aviso
        por baixo do seletor diz quando isso vai acontecer.
        """
        box = QtWidgets.QGroupBox("Mantra")
        box.setStyleSheet(
            f"QGroupBox{{background:{PANEL}; border-radius:8px; padding:14px;"
            f"margin-top:8px; color:{DIM};}}"
            "QGroupBox::title{subcontrol-origin:margin; left:12px;}"
        )
        column = QtWidgets.QVBoxLayout(box)

        self._mantra = QtWidgets.QComboBox()
        self._mantra.setStyleSheet(
            f"QComboBox{{background:#1d2230; color:{TEXT}; border:0;"
            f"padding:8px; font-size:14px;}}"
            f"QComboBox QAbstractItemView{{background:#1d2230; color:{TEXT};"
            f"selection-background-color:#2e3a52;}}"
        )
        for entry in self._cfg.paths.mantras:
            self._mantra.addItem(entry.label, entry.id)
        chosen = self._cfg.paths.mantra(self._plan.mantra_id)
        index = self._mantra.findData(chosen.id)
        if index >= 0:
            self._mantra.setCurrentIndex(index)
        self._mantra.currentIndexChanged.connect(self._emit_plan)
        column.addWidget(self._mantra)

        self._mantra_note = QtWidgets.QLabel()
        self._mantra_note.setWordWrap(True)
        self._mantra_note.setStyleSheet(f"color:{DIM}; font-size:11px;")
        column.addWidget(self._mantra_note)
        self._refresh_mantra_note()
        return box

    def _refresh_mantra_note(self) -> None:
        entry = self._cfg.paths.mantra(self._selected_mantra_id())
        minutes, seconds = divmod(int(self._plan.mantra_s), 60)
        self._mantra_note.setText(
            f"{entry.file.name} · fase de {minutes:02d}:{seconds:02d}. "
            f"Se o ficheiro for mais curto, repete em ciclo até a fase acabar."
        )

    def _selected_mantra_id(self) -> str:
        data = self._mantra.currentData()
        return str(data) if data else self._cfg.paths.default_mantra

    def _markers_box(self) -> QtWidgets.QWidget:
        """Que biomarcadores entram no relatorio final.

        Itera o registo de ``markers.py``, que e a fonte unica: acrescentar um
        marcador la faz com que apareca aqui sozinho.
        """
        box = QtWidgets.QGroupBox("Biomarcadores no relatório")
        box.setStyleSheet(
            f"QGroupBox{{background:{PANEL}; border-radius:8px; padding:14px;"
            f"margin-top:8px; color:{DIM};}}"
            "QGroupBox::title{subcontrol-origin:margin; left:12px;}"
        )
        outer = QtWidgets.QVBoxLayout(box)
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(24)
        outer.addLayout(grid)

        checkbox_style = (
            f"QCheckBox{{color:{TEXT}; font-size:13px; spacing:8px;}}"
            "QCheckBox::indicator{width:16px; height:16px; border-radius:4px;"
            "border:2px solid #55617a; background:#1d2230;}"
            "QCheckBox::indicator:checked{background:#8fd6a0;"
            "border-color:#8fd6a0;}"
        )
        self._marker_boxes: dict[str, QtWidgets.QCheckBox] = {}
        entries = list(markers())
        rows = (len(entries) + 2) // 3
        for i, marker in enumerate(entries):
            checkbox = QtWidgets.QCheckBox(marker.friendly or marker.label)
            checkbox.setToolTip(f"{marker.label}\n\n{marker.notes}")
            checkbox.setStyleSheet(checkbox_style)
            # Por defeito, os marcadores; os extras ficam desligados.
            checkbox.setChecked(marker.role == "marker")
            checkbox.stateChanged.connect(self._emit_selection)
            self._marker_boxes[marker.id] = checkbox
            grid.addWidget(checkbox, i % rows, i // rows)

        row = QtWidgets.QHBoxLayout()
        for text, value in (("Todos", True), ("Nenhum", False)):
            button = QtWidgets.QPushButton(text)
            button.setStyleSheet(
                f"QPushButton{{background:#1d2230; color:{DIM}; border:0;"
                f"border-radius:6px; padding:6px 14px; font-size:12px;}}"
            )
            button.clicked.connect(lambda _=False, v=value: self._set_all(v))
            row.addWidget(button)
        row.addStretch(1)

        self._apply = QtWidgets.QPushButton("Adicionar ao ecrã final")
        self._apply.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._apply.setStyleSheet(
            "QPushButton{background:#8fd6a0; color:#0b0d12; border:0;"
            "border-radius:6px; padding:8px 18px; font-size:13px;"
            "font-weight:600;}"
            "QPushButton:hover{background:#a9e6b8;}"
        )
        self._apply.setToolTip(
            "Acrescenta os marcadores marcados às linhas, às barras e aos "
            "botões do ecrã final. O destaque automático mantém-se."
        )
        self._apply.clicked.connect(self._emit_apply)
        row.addWidget(self._apply)

        self._apply_status = QtWidgets.QLabel("")
        self._apply_status.setStyleSheet(f"color:{DIM}; font-size:11px;")
        outer.addLayout(row)
        outer.addWidget(self._apply_status)
        return box

    def _emit_apply(self) -> None:
        chosen = [k for k, box in self._marker_boxes.items() if box.isChecked()]
        self.apply_requested.emit(chosen)
        self._apply_status.setText(
            f"{len(chosen)} marcadores enviados para o ecrã final."
            if chosen
            else "Nenhum marcador marcado — o ecrã final ficou com o destaque."
        )

    def _set_all(self, value: bool) -> None:
        for checkbox in self._marker_boxes.values():
            checkbox.blockSignals(True)
            checkbox.setChecked(value)
            checkbox.blockSignals(False)
        self._emit_selection()

    def _emit_selection(self) -> None:
        self.selection_changed.emit(self.selection)

    @property
    def selection(self) -> dict[str, bool]:
        return {k: c.isChecked() for k, c in self._marker_boxes.items()}

    def set_selection(self, selection: dict[str, bool]) -> None:
        for marker_id, checked in selection.items():
            checkbox = self._marker_boxes.get(marker_id)
            if checkbox is None:
                continue
            checkbox.blockSignals(True)
            checkbox.setChecked(checked)
            checkbox.blockSignals(False)

    def _drive_box(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Google Drive")
        box.setStyleSheet(
            f"QGroupBox{{background:{PANEL}; border-radius:8px; padding:14px;"
            f"margin-top:8px; color:{DIM};}}"
            "QGroupBox::title{subcontrol-origin:margin; left:12px;}"
        )
        layout = QtWidgets.QHBoxLayout(box)
        self._drive_status = QtWidgets.QLabel()
        self._drive_status.setWordWrap(True)
        self._drive_status.setTextFormat(QtCore.Qt.TextFormat.RichText)
        layout.addWidget(self._drive_status, stretch=1)

        self._sign_in = QtWidgets.QPushButton("Iniciar sessão Google")
        self._sign_in.setStyleSheet(
            "QPushButton{background:#8fd6a0; color:#0b0d12; border:0;"
            "border-radius:8px; padding:10px 18px; font-weight:600;}"
            "QPushButton:disabled{background:#2a2f3a; color:#5a6474;}"
        )
        self._sign_in.clicked.connect(self._do_sign_in)
        layout.addWidget(self._sign_in)

        # Sem isto nao ha forma de trocar de conta na mesma maquina: o token
        # fica guardado e o "Iniciar sessao" reutiliza-o. Um segundo membro
        # da equipa tinha de apagar um ficheiro a mao.
        self._sign_out = QtWidgets.QPushButton("Trocar de conta")
        self._sign_out.setStyleSheet(
            f"QPushButton{{background:#1d2230; color:{TEXT}; border:0;"
            f"border-radius:8px; padding:10px 14px;}}"
            "QPushButton:disabled{color:#5a6474;}"
        )
        self._sign_out.setToolTip(
            "Termina a sessão Google guardada nesta máquina. O envio e o "
            "email passam a usar a conta que iniciar sessão a seguir."
        )
        self._sign_out.clicked.connect(self._do_sign_out)
        layout.addWidget(self._sign_out)

        self._retry = QtWidgets.QPushButton("Repetir envios")
        self._retry.setStyleSheet(
            f"QPushButton{{background:#1d2230; color:{TEXT}; border:0;"
            f"border-radius:8px; padding:10px 14px;}}"
        )
        self._retry.clicked.connect(lambda: self._uploads and self._uploads.retry_failed())
        layout.addWidget(self._retry)
        return box

    def _on_reconnect_clicked(self) -> None:
        self._reconnect.setStyleSheet(self._reconnect_busy_style)
        self._reconnect.setText("a procurar o equipamento…")
        self._reconnect.setEnabled(False)
        QtWidgets.QApplication.processEvents()
        self.reconnect_requested.emit()

    def _do_sign_in(self) -> None:
        if self._drive is None:
            return
        self._sign_in.setEnabled(False)
        self._sign_in.setText("a abrir o browser…")
        QtWidgets.QApplication.processEvents()
        try:
            account = self._drive.sign_in()
            self._drive_status.setText(
                f"<b style='color:#8fd6a0'>sessão iniciada</b><br>"
                f"<span style='color:{DIM}'>{account}</span>"
            )
            if self._uploads is not None:
                self._uploads.kick()
        except DriveError as exc:
            self._drive_status.setText(f"<span style='color:#e08b84'>{exc}</span>")
        finally:
            self._sign_in.setText("Iniciar sessão Google")
            self._sign_in.setEnabled(True)

    def _do_sign_out(self) -> None:
        if self._drive is None:
            return
        account = self._drive.account
        self._drive.sign_out()
        self._drive_status.setText(
            f"<span style='color:{DIM}'>sessão de {account} terminada. "
            f"Inicie sessão com a conta que vai usar.</span>"
        )
        self._refresh_drive()

    def _refresh_drive(self) -> None:
        if self._drive is None:
            self._drive_status.setText(
                f"<span style='color:{DIM}'>envio desligado "
                f"(drive.enabled: false na config)</span>"
            )
            self._sign_in.setEnabled(False)
            self._sign_out.setEnabled(False)
            self._retry.setEnabled(False)
            return

        lines = []
        if self._drive.signed_in:
            lines.append(
                f"<b style='color:#8fd6a0'>sessão iniciada</b> "
                f"<span style='color:{DIM}'>{self._drive.account}</span>"
            )
        else:
            lines.append("<b style='color:#c99a1e'>sem sessão iniciada</b>")

        if self._uploads is not None:
            stats = self._uploads.stats()
            lines.append(
                f"<span style='color:{DIM}'>enviadas {stats.uploaded} · "
                f"por enviar {stats.pending} "
                f"({stats.bytes_pending / 1e6:.0f} MB)"
                + (f" · falhadas {stats.failed}" if stats.failed else "")
                + ("</span>")
            )
            if stats.last_error:
                lines.append(
                    f"<span style='color:#e08b84'>{stats.last_error[:140]}</span>"
                )
            self._retry.setEnabled(bool(stats.pending or stats.failed))
        self._drive_status.setText("<br>".join(lines))
        self._sign_in.setEnabled(not self._drive.signed_in)
        self._sign_out.setEnabled(self._drive.signed_in)

    # -- estado ---------------------------------------------------------------- #
    def _emit_plan(self) -> None:
        plan = self._plan.with_durations(
            calibration_s=float(self._spins["calibration"].value()),
            mantra_s=float(self._spins["mantra"].value()),
            settle_s=float(self._spins["settle"].value()),
        )
        self._plan = replace(plan, mantra_id=self._selected_mantra_id())
        self._refresh_adequacy()
        self._refresh_mantra_note()
        self.plan_changed.emit(self._plan)

    def set_plan(self, plan: SessionPlan) -> None:
        self._plan = plan
        for key, value in (
            ("calibration", plan.calibration_s),
            ("mantra", plan.mantra_s),
            ("settle", plan.settle_s),
        ):
            spin = self._spins[key]
            spin.blockSignals(True)
            spin.setValue(int(value))
            spin.blockSignals(False)
        index = self._mantra.findData(self._cfg.paths.mantra(plan.mantra_id).id)
        if index >= 0:
            self._mantra.blockSignals(True)
            self._mantra.setCurrentIndex(index)
            self._mantra.blockSignals(False)
        self._refresh_mantra_note()
        self._refresh_adequacy()

    def _refresh_adequacy(self) -> None:
        adequacy = self._cfg.reference_adequacy(
            self._plan.calibration_s, self._plan.calib_use_last_s
        )
        # A "Sintonia com o mantra" e a "Sintonia meditativa" sao o unico par
        # especifico de pratica com mantra, e sao as primeiras a cair com
        # calibracao curta: a grelha delas e de 10 s com passo de 5, portanto
        # 30 s uteis compram cinco janelas. Este aviso estava em cinzento de
        # 11 px e passou despercebido numa sessao real, onde a coerencia ficou
        # com UMA janela de calibracao.
        style = f"color:{DIM}; font-size:11px; font-family:'Consolas',monospace;"
        note = ""
        if adequacy.conn_quality == "insufficient":
            note = (
                "\n  -> SINTONIA COM O MANTRA e SINTONIA MEDITATIVA ficam"
                "\n     sem referencia. Precisam de ~180 s de calibracao."
            )
            style = (
                "color:#e0a04a; font-size:12px; font-weight:600;"
                " font-family:'Consolas',monospace;"
            )
        elif adequacy.conn_quality == "thin":
            note = "\n  -> sintonia com referencia fraca (~180 s seria folgado)"
            style = (
                "color:#c99a1e; font-size:11px;"
                " font-family:'Consolas',monospace;"
            )
        self._adequacy.setStyleSheet(style)
        self._adequacy.setText(adequacy.describe() + note)

    def _refresh_status(self) -> None:
        status = self._device.status()
        colour = STATE_COLOUR.get(status.state, DIM)
        lines = [
            f"<b style='color:{colour}; font-size:19px'>"
            f"{status.state.value.upper()}</b>"
        ]
        if status.device_id:
            lines.append(
                f"<span style='color:{DIM}'>{status.source_name} · "
                f"{status.device_id}</span>"
            )
        if status.detail:
            lines.append(f"<span style='color:{DIM}'>{status.detail}</span>")

        if status.state == DeviceState.STREAMING and not status.settled:
            lines.append(
                f"<span style='color:#c99a1e'>a assentar o amplificador — "
                f"{status.settle_remaining_s:.0f} s</span>"
            )
        if status.needs_restart:
            lines.append(
                "<b style='color:#e08b84'>Ligue o headset (auto-desliga por "
                "inatividade) e prima o botão abaixo.</b>"
            )
        self._status.setText("<br>".join(lines))
        searching = status.state == DeviceState.CONNECTING
        self._reconnect.setEnabled(not searching)
        self._reconnect.setStyleSheet(
            self._reconnect_busy_style if searching else self._reconnect_idle_style
        )
        self._reconnect.setText(
            "a procurar o equipamento…" if searching else "Ligar / reiniciar dispositivo"
        )
        self._refresh_drive()
        self._sync_inspector()

    def _sync_inspector(self) -> None:
        """Só monta o traçado quando há aquisição, e desmonta-o quando não há."""
        acquisition = self._device.acquisition
        streaming = acquisition is not None and acquisition.running
        if streaming and self._inspector is None:
            self._placeholder.hide()
            self._inspector = SignalInspector(self._cfg, acquisition)
            self._signal_holder.addWidget(self._inspector)
        elif not streaming and self._inspector is not None:
            self._inspector.stop()
            self._inspector.setParent(None)
            self._inspector = None
            self._placeholder.show()

    # -- ciclo de vida ---------------------------------------------------------- #
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self._timer.stop()
        if self._inspector is not None:
            self._inspector.stop()
        super().closeEvent(event)
