"""Ecrãs da banca. Uma aplicação, um monitor.

Cada computador do evento tem um único ecrã, e o ambiente é caótico: barulho,
palcos, luzes. Não há janela de operador separada — o painel de investigador
abre por cima com ``Ctrl+I`` quando é preciso.

Todos os ecrãs têm um **X** no canto superior direito. Durante a sessão aborta
e volta ao ecrã de boas-vindas; no ecrã de boas-vindas fecha a aplicação.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets

from ..config import Config

BG = "#07080b"
FG = "#eef1f7"
MUTED = "#8b95a7"
ACCENT = "#7fb2ff"
GOOD = "#8fd6a0"


def label(text: str, size: int, colour: str = FG, bold: bool = False) -> QtWidgets.QLabel:
    widget = QtWidgets.QLabel(text)
    widget.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    widget.setWordWrap(True)
    widget.setStyleSheet(
        f"color:{colour}; font-size:{size}px; font-weight:{'600' if bold else '300'};"
        f"font-family:'Segoe UI',sans-serif; background:transparent;"
    )
    return widget


def big_button(text: str, colour: str = ACCENT) -> QtWidgets.QPushButton:
    button = QtWidgets.QPushButton(text)
    button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
    button.setMinimumHeight(64)
    button.setStyleSheet(
        f"QPushButton{{background:{colour}; color:#07080b; border:0;"
        f"border-radius:32px; padding:0 48px; font-size:22px; font-weight:600;"
        f"font-family:'Segoe UI',sans-serif;}}"
        f"QPushButton:hover{{background:#a8caff;}}"
        f"QPushButton:disabled{{background:#2a2f3a; color:#5a6474;}}"
    )
    return button


def format_time(seconds: float) -> str:
    total = max(int(seconds + 0.999), 0)
    return f"{total // 60}:{total % 60:02d}"


class Screen(QtWidgets.QWidget):
    """Base: fundo escuro e conteúdo centrado."""

    def __init__(self) -> None:
        super().__init__()
        self.setStyleSheet(f"background:{BG};")
        self.body = QtWidgets.QVBoxLayout(self)
        self.body.setContentsMargins(90, 70, 90, 70)
        self.body.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)


# --------------------------------------------------------------------------- #
class WelcomeScreen(Screen):
    """Boas-vindas: logótipo, título do evento, e um só botão."""

    start = QtCore.pyqtSignal()

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        logo_path = Path(cfg.paths.logo)
        if logo_path.exists():
            pixmap = QtGui.QPixmap(str(logo_path)).scaledToHeight(
                120, QtCore.Qt.TransformationMode.SmoothTransformation
            )
            logo = QtWidgets.QLabel()
            logo.setPixmap(pixmap)
            logo.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.body.addWidget(logo)
        else:
            self.body.addWidget(label("NEROES", 46, FG, bold=True))
        self.body.addSpacing(50)
        self.body.addWidget(label(cfg.ui.title, 62, FG, bold=True))
        self.body.addSpacing(14)
        self.body.addWidget(label(cfg.ui.subtitle, 30, MUTED))
        self.body.addSpacing(70)

        button = big_button("Iniciar sessão")
        button.clicked.connect(self.start.emit)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(button)
        row.addStretch(1)
        self.body.addLayout(row)


# --------------------------------------------------------------------------- #
class ExplainScreen(Screen):
    """Explica o protocolo, mostra o diagrama e deixa editar as durações."""

    begin = QtCore.pyqtSignal()
    durations_changed = QtCore.pyqtSignal(float, float, float)

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self._cfg = cfg
        self.body.addWidget(label("Vamos ajustar os sensores", 44, FG, bold=True))
        self.body.addSpacing(10)
        self.body.addWidget(
            label(
                "Fique confortável e mantenha os olhos fechados durante toda a "
                "experiência.",
                24,
                MUTED,
            )
        )
        self.body.addSpacing(44)

        self._steps = QtWidgets.QHBoxLayout()
        self._steps.setSpacing(0)
        self.body.addLayout(self._steps)
        self._boxes: list[tuple[QtWidgets.QLabel, QtWidgets.QLabel]] = []
        for index, (title, note) in enumerate(
            (
                ("Calibração", "olhos fechados, em silêncio"),
                ("Mantra", "áudio, olhos fechados"),
                ("Repouso", "voltar ao normal"),
            )
        ):
            if index:
                arrow = label("→", 30, "#3a4152")
                arrow.setFixedWidth(60)
                self._steps.addWidget(arrow)
            self._steps.addWidget(self._step_box(title, note))

        self.body.addSpacing(40)
        self.body.addWidget(self._duration_row(), alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.body.addSpacing(40)

        button = big_button("Iniciar demo", GOOD)
        button.clicked.connect(self.begin.emit)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(button)
        row.addStretch(1)
        self.body.addLayout(row)

        self._status = label("", 17, MUTED)
        self.body.addSpacing(20)
        self.body.addWidget(self._status)

    def _step_box(self, title: str, note: str) -> QtWidgets.QWidget:
        box = QtWidgets.QFrame()
        box.setStyleSheet(
            "QFrame{background:#11141b; border:1px solid #1e2431; border-radius:14px;}"
        )
        box.setFixedWidth(280)
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(22, 24, 22, 24)
        layout.addWidget(label(title, 24, FG, bold=True))
        duration = label("", 40, ACCENT, bold=True)
        layout.addWidget(duration)
        note_label = label(note, 15, MUTED)
        layout.addWidget(note_label)
        self._boxes.append((duration, note_label))
        return box

    def _duration_row(self) -> QtWidgets.QWidget:
        holder = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(holder)
        row.setSpacing(26)
        self._spins: dict[str, QtWidgets.QSpinBox] = {}
        proto = self._cfg.protocol
        for key, text, value in (
            ("calibration", "calibração", proto.calibration_s),
            ("mantra", "mantra", proto.mantra_s),
            ("settle", "repouso", proto.settle_s),
        ):
            column = QtWidgets.QVBoxLayout()
            column.addWidget(label(text, 15, MUTED))
            spin = QtWidgets.QSpinBox()
            # Sem limites impostos: a duração é livre. Acima da duração do
            # vídeo, ele repete em ciclo.
            spin.setRange(1, 24 * 60 * 60)
            spin.setSuffix(" s")
            spin.setSingleStep(5)
            spin.setValue(int(value))
            spin.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            spin.setStyleSheet(
                "QSpinBox{background:#161a22; color:#eef1f7; border:1px solid "
                "#232a37; border-radius:8px; padding:8px; font-size:18px;}"
            )
            spin.valueChanged.connect(self._emit)
            self._spins[key] = spin
            column.addWidget(spin)
            row.addLayout(column)
        self._refresh_boxes()
        return holder

    def _emit(self) -> None:
        self._refresh_boxes()
        self.durations_changed.emit(
            float(self._spins["calibration"].value()),
            float(self._spins["mantra"].value()),
            float(self._spins["settle"].value()),
        )

    def _refresh_boxes(self) -> None:
        for (duration_label, _), key in zip(
            self._boxes, ("calibration", "mantra", "settle")
        ):
            duration_label.setText(format_time(self._spins[key].value()))

    def set_status(self, text: str, colour: str = MUTED) -> None:
        self._status.setText(text)
        self._status.setStyleSheet(
            f"color:{colour}; font-size:17px; background:transparent;"
        )


# --------------------------------------------------------------------------- #
class TimerScreen(Screen):
    """Calibração, mantra e repouso: uma instrução e um contador."""

    def __init__(self, instruction: str, hint: str = "") -> None:
        super().__init__()
        self._instruction, self._hint = instruction, hint
        self.instruction = label(instruction, 46, FG, bold=True)
        self.countdown = label("--", 170, ACCENT, bold=True)
        self.hint = label(hint, 19, MUTED)
        self.body.addStretch(1)
        self.body.addWidget(self.instruction)
        self.body.addSpacing(26)
        self.body.addWidget(self.countdown)
        self.body.addSpacing(16)
        self.body.addWidget(self.hint)
        self.body.addStretch(1)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.setStyleSheet(
            "QProgressBar{background:#12151c; border:0; border-radius:3px;}"
            f"QProgressBar::chunk{{background:{ACCENT}; border-radius:3px;}}"
        )
        self.body.addWidget(self.progress)

        # Barra do assentamento do amplificador. Verde, e distinta da azul: os
        # ~10 s iniciais nao sao calibracao — o relogio da sessao nem sequer
        # arrancou ainda, porque conta amostras e ainda nao ha nenhuma.
        self.warmup = QtWidgets.QProgressBar()
        self.warmup.setTextVisible(False)
        self.warmup.setFixedHeight(6)
        self.warmup.setStyleSheet(
            "QProgressBar{background:#12151c; border:0; border-radius:3px;}"
            "QProgressBar::chunk{background:" + GOOD + "; border-radius:3px;}"
        )
        self.body.addWidget(self.warmup)
        self.set_warmup(None)

    def set_warmup(self, fraction: float | None) -> None:
        """``None`` esconde a barra verde e devolve o ecra ao estado normal."""
        warming = fraction is not None
        self.warmup.setVisible(warming)
        self.progress.setVisible(not warming)
        self.countdown.setVisible(not warming)
        if warming:
            self.instruction.setText("A ligar o equipamento")
            self.hint.setText("um momento, por favor")
            self.warmup.setValue(int(max(0.0, min(fraction, 1.0)) * 100))
        else:
            self.instruction.setText(self._instruction)
            self.hint.setText(self._hint)

    def update_time(self, remaining_s: float | None, progress: float | None) -> None:
        self.countdown.setText("--" if remaining_s is None else format_time(remaining_s))
        self.progress.setValue(int((progress or 0.0) * 100))


# --------------------------------------------------------------------------- #
class ReportScreen(Screen):
    """Relatório e, opcionalmente, recolha de email para o enviar.

    O EEG é dado biométrico. Associá-lo a um identificador direto muda o
    enquadramento de RGPD (SPEC 11), por isso o consentimento é explícito, a
    finalidade está escrita no ecrã, o prazo de retenção é dito, e o endereço é
    guardado **separado** do sinal.
    """

    email_submitted = QtCore.pyqtSignal(str, str)   # nome, email
    new_session = QtCore.pyqtSignal()

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self._cfg = cfg
        self.body.addWidget(label("O seu registo", 46, FG, bold=True))
        self.body.addSpacing(8)
        self._summary = label("", 22, MUTED)
        self.body.addWidget(self._summary)
        self.body.addSpacing(24)

        from .report_view import ResultsView

        self.results = ResultsView(cfg)
        self.results.setMinimumHeight(360)
        self.body.addWidget(self.results, stretch=1)
        self.body.addSpacing(20)

        if cfg.ui.email.get("enabled", False):
            self.body.addWidget(self._email_block())

        self.body.addSpacing(16)
        self._footer = label(cfg.report.participant_footer, 15, "#5d6779")
        self.body.addWidget(self._footer)

    def _email_block(self) -> QtWidgets.QWidget:
        holder = QtWidgets.QFrame()
        holder.setStyleSheet(
            "QFrame{background:#11141b; border:1px solid #1e2431; border-radius:14px;}"
        )
        layout = QtWidgets.QVBoxLayout(holder)
        layout.setContentsMargins(28, 22, 28, 22)

        layout.addWidget(label("Receber o relatório por email", 22, FG, bold=True))
        purpose = str(self._cfg.ui.email.get("purpose", "")).strip()
        days = self._cfg.ui.email.get("retention_days", 30)
        layout.addWidget(label(purpose, 15, MUTED))
        layout.addSpacing(10)

        self._consent = QtWidgets.QCheckBox(
            f"Autorizo o envio e a guarda dos meus dados por {days} dias."
        )
        # A caixa preta sobre fundo preto nao se via. Contorno claro, fundo mais
        # claro que o painel, e preenchimento verde quando activa.
        self._consent.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._consent.setStyleSheet(
            "QCheckBox{color:" + FG + "; font-size:16px; spacing:12px;}"
            "QCheckBox::indicator{width:24px; height:24px; border-radius:6px;"
            "border:2px solid #55617a; background:#1d2230;}"
            "QCheckBox::indicator:hover{border-color:#7fb2ff;}"
            "QCheckBox::indicator:checked{background:" + GOOD + ";"
            "border-color:" + GOOD + ";}"
        )
        self._consent.stateChanged.connect(self._refresh_email_button)
        layout.addWidget(self._consent)
        layout.addSpacing(10)

        field_style = (
            "QLineEdit{background:#161a22; color:#eef1f7; border:1px solid "
            "#232a37; border-radius:10px; padding:0 14px; font-size:17px;}"
            "QLineEdit:focus{border-color:#7fb2ff;}"
        )
        row = QtWidgets.QHBoxLayout()
        self._name = QtWidgets.QLineEdit()
        self._name.setPlaceholderText("nome")
        self._name.setMinimumHeight(46)
        self._name.setStyleSheet(field_style)
        self._name.textChanged.connect(self._refresh_email_button)
        self._name.returnPressed.connect(self._submit_email)
        row.addWidget(self._name, stretch=1)

        self._email = QtWidgets.QLineEdit()
        self._email.setPlaceholderText("email@exemplo.pt")
        self._email.setMinimumHeight(46)
        self._email.setStyleSheet(field_style)
        self._email.textChanged.connect(self._refresh_email_button)
        self._email.returnPressed.connect(self._submit_email)
        row.addWidget(self._email, stretch=2)

        self._send = big_button("Enviar")
        self._send.setMinimumHeight(46)
        self._send.setEnabled(False)
        self._send.clicked.connect(self._submit_email)
        row.addWidget(self._send)
        layout.addLayout(row)

        self._email_status = label("", 15, MUTED)
        layout.addWidget(self._email_status)
        return holder

    def _refresh_email_button(self) -> None:
        text = self._email.text().strip()
        valid = "@" in text and "." in text.split("@")[-1] and len(text) > 5
        self._send.setEnabled(valid and self._consent.isChecked())

    def _submit_email(self) -> None:
        if not self._send.isEnabled():
            return
        self.email_submitted.emit(self._name.text().strip(), self._email.text().strip())
        self._name.setEnabled(False)
        self._email.setEnabled(False)
        self._send.setEnabled(False)
        self._consent.setEnabled(False)
        self._email_status.setText("A preparar o relatório…")

    def set_email_status(self, text: str, ok: bool | None = None) -> None:
        """O que aconteceu ao envio, no ecrã, à frente da pessoa.

        Um envio que falha em silêncio é pior do que não haver envio: a pessoa
        sai da banca a contar com um email que nunca chega.
        """
        if not hasattr(self, "_email_status"):
            return
        colour = MUTED if ok is None else (GOOD if ok else "#d98b7a")
        self._email_status.setStyleSheet(f"color:{colour};")
        self._email_status.setText(text)
        if ok is False:
            # Deixa tentar outra vez: o mais provável é ter faltado a rede.
            self._name.setEnabled(True)
            self._email.setEnabled(True)
            self._consent.setEnabled(True)
            self._refresh_email_button()

    # -- conteúdo --------------------------------------------------------------- #
    def set_summary(self, text: str) -> None:
        self._summary.setText(text)

    def show_results(self, result, phases, marker_ids) -> None:
        self.results.show_results(result, phases, marker_ids)

    def reset(self) -> None:
        self.results.clear()
        self._summary.setText("")
        if hasattr(self, "_email"):
            self._name.clear()
            self._name.setEnabled(True)
            self._email.clear()
            self._email.setEnabled(True)
            self._consent.setChecked(False)
            self._consent.setEnabled(True)
            self._email_status.setText("")
