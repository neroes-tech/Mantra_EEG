"""Aplicação da banca: uma janela, um monitor.

Cada computador do evento corre isto em ecrã inteiro. Não há janela de operador
separada — o participante não pode ver diagnósticos, e o painel de investigador
abre por cima com ``Ctrl+I`` quando é preciso.

Fluxo: boas-vindas → explicação (com os tempos editáveis) → calibração →
mantra → repouso → relatório.

**A aquisição só corre da calibração ao fim do repouso.** Fora disso o
dispositivo é libertado. É o que o protocolo pede — LED azul fixo só durante a
recolha — e é também o que impede a aplicação de bloquear depois de horas em
espera com o headset desligado, que foi o que aconteceu na versão anterior.

Atalhos: ``Ctrl+I`` painel do investigador, ``V`` mostrar/esconder o vídeo sem
parar o som, ``Esc`` e o **X** no canto abortam para o ecrã inicial — e fecham a
aplicação se já lá estiverem.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from ..audio import AudioPlayer
from ..config import Config
from ..device import DeviceManager, DeviceState
from ..drive import DriveClient, DriveError
from ..mail import GmailSender, MailError, compose, valid_address
from ..publish import (
    GitHubTarget,
    PublishError,
    publish_to_github,
    wait_until_live,
)
from ..report.html import render as render_report
from ..report.payload import ReportContext, build as build_report
from ..storage import UploadQueue
from ..montage import auto_checks, summarize
from ..recorder import Recorder
from ..analysis import analyse, rank_by_variation
from ..applog import log, log_exception
from ..markers import markers
from ..session import Phase, PhaseChange, SessionController, SessionPlan
from ..sources.base import ACCEL_SLICE, EEG_SLICE, GYRO_SLICE
from .researcher import ResearcherPanel
from .screens import (
    ExplainScreen,
    ReportScreen,
    Screen,
    TimerScreen,
    WelcomeScreen,
    format_time,
)

#: Fases em que o equipamento está a recolher (LED azul fixo).
RECORDING_PHASES = (Phase.CALIBRATION, Phase.MANTRA, Phase.SETTLE)


class MainWindow(QtWidgets.QMainWindow):
    """Janela única da banca."""

    #: O envio corre fora da thread da interface — sem isto a janela congela
    #: durante o upload, à frente do participante. O sinal traz o resultado de
    #: volta para o ecrã.
    #: (token, ok, detalhe). O token identifica o envio: se o participante
    #: carregar no X a meio, o email segue na mesma — o consentimento foi
    #: dado — mas o resultado não pode escrever no ecrã da pessoa seguinte.
    report_sent = QtCore.pyqtSignal(int, bool, str)
    #: (token, etapa). O que está a acontecer, à frente de quem espera.
    report_progress = QtCore.pyqtSignal(int, str)

    def __init__(
        self,
        cfg: Config,
        source_override: str | None = None,
        replay_file: str | None = None,
        fullscreen: bool = True,
    ) -> None:
        super().__init__()
        self._cfg = cfg
        self._fullscreen = fullscreen
        self._recorder: Recorder | None = None
        self._panel: ResearcherPanel | None = None
        self._last_session_dir: Path | None = None
        #: O resultado da última análise, para o relatório ser construído com
        #: o nome da pessoa quando ela o escrever — e não antes.
        self._last_analysis: tuple[object, list] | None = None
        #: O relatório, construído logo a seguir à análise. Quando a pessoa
        #: carrega em Enviar já só falta trocar o nome e publicar.
        self._report_data: dict | None = None
        self._contact_path: Path | None = None
        self._session_started = datetime.now()
        #: Identifica o envio em curso. Ver ``report_sent``.
        self._send_token = 0
        self._send_stage = ""
        self._send_deadline = QtCore.QTimer(self)
        self._send_deadline.setSingleShot(True)
        # Ligado UMA vez. Estava dentro de _send_report, portanto cada nova
        # tentativa acrescentava outra ligacao e o aviso de prazo esgotado
        # passava a disparar tantas vezes quantas as tentativas.
        self._send_deadline.timeout.connect(self._on_send_timeout)
        #: O destaque do ecrã final, para o botão "Adicionar" do painel poder
        #: acrescentar-lhe marcadores sem o perder.
        self._featured: list[str] = []
        #: Que biomarcadores entram no relatorio. None = os do registo.
        #: Que biomarcadores o painel pode acrescentar ao ecra final. Comeca
        #: com todos os marcadores ligados e os `extra` desligados. **Nao
        #: limita o destaque**: o top 3 e escolhido de entre todos, senao uma
        #: selecao curta deixava o ecra final com um cartao.
        self._marker_selection: dict[str, bool] = {
            m.id: (m.role == "marker") for m in markers()
        }
        self._video: QtWidgets.QWidget | None = None

        self._device = DeviceManager(
            cfg,
            on_chunk=self._on_chunk,
            source_override=source_override,
            replay_file=replay_file,
        )
        self._session = SessionController(
            SessionPlan.from_config(cfg), on_change=self._on_phase_change
        )
        self._audio = AudioPlayer(
            cfg.protocol.cues.start_sound, cfg.protocol.cues.end_sound, None
        )
        self._drive = self._build_drive()
        self._uploads = UploadQueue(
            root=cfg.paths.sessions_dir / cfg.paths.drive_folder_name,
            client=self._drive,
            folder_id=cfg.drive.folder_id,
            delete_after_upload=cfg.drive.delete_after_upload,
            max_attempts=cfg.drive.max_attempts,
            base_backoff_s=cfg.drive.base_backoff_s,
        )
        # Despacha o que tiver ficado por enviar da sessao anterior da app.
        self._uploads.kick()

        self.setWindowTitle(f"{cfg.ui.title} — {cfg.ui.subtitle}")
        self.setStyleSheet("background:#07080b;")
        self._build()
        self._bind_shortcuts()

        self._tick = QtCore.QTimer(self)
        self._tick.timeout.connect(self._on_tick)
        self._tick.start(100)

    # -- construção ------------------------------------------------------------ #
    def _build(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QStackedLayout(central)
        root.setStackingMode(QtWidgets.QStackedLayout.StackingMode.StackAll)
        root.setContentsMargins(0, 0, 0, 0)

        self._video = self._build_video()
        if self._video is not None:
            root.addWidget(self._video)
            self._video.hide()

        overlay = QtWidgets.QWidget()
        overlay.setStyleSheet("background:transparent;")
        overlay_layout = QtWidgets.QVBoxLayout(overlay)
        overlay_layout.setContentsMargins(0, 0, 0, 0)
        overlay_layout.setSpacing(0)

        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(0, 14, 22, 0)
        top.addStretch(1)
        self._close_button = QtWidgets.QPushButton("✕")
        self._close_button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._close_button.setFixedSize(46, 46)
        # O tooltip e definido por _show_screen, que sabe em que ecra estamos.
        # Aqui _screens ainda nao existe.
        self._close_button.setStyleSheet(
            "QPushButton{background:transparent; color:#5d6779; border:0;"
            "font-size:26px;}"
            "QPushButton:hover{color:#e08b84;}"
        )
        self._close_button.clicked.connect(self._on_close_clicked)
        top.addWidget(self._close_button)
        overlay_layout.addLayout(top)

        self._screens = QtWidgets.QStackedWidget()
        self._screens.setStyleSheet("background:transparent;")
        overlay_layout.addWidget(self._screens, stretch=1)
        root.addWidget(overlay)

        self.welcome = WelcomeScreen(self._cfg)
        self.explain = ExplainScreen(self._cfg)
        self.calibration = TimerScreen(
            "Feche os olhos e relaxe", "a preparar a sua linha de base"
        )
        self.mantra = TimerScreen("Sinta o mantra", "prima V para mostrar o vídeo")
        self.rest = TimerScreen("Repouso", "mantenha os olhos fechados")
        self.report = ReportScreen(self._cfg)

        for screen in (
            self.welcome,
            self.explain,
            self.calibration,
            self.mantra,
            self.rest,
            self.report,
        ):
            self._screens.addWidget(screen)

        self.welcome.start.connect(self._go_explain)
        self.explain.begin.connect(self._begin_session)
        self.explain.durations_changed.connect(self._set_durations)
        self.report.email_submitted.connect(self._store_email)
        self.report_sent.connect(self._on_report_sent)
        self.report_progress.connect(self._on_send_progress)
        self.report.new_session.connect(self._go_welcome)
        self._show_screen(self.welcome)

    def _build_drive(self) -> DriveClient | None:
        drive = self._cfg.drive
        if not drive.enabled:
            return None
        return DriveClient(
            client_id=drive.client_id,
            client_secret=drive.client_secret,
            token_path=drive.token_file,
            scope=drive.scope,
        )

    @property
    def drive(self) -> DriveClient | None:
        return self._drive

    @property
    def uploads(self) -> UploadQueue:
        return self._uploads

    def _build_video(self) -> QtWidgets.QWidget | None:
        entry = self._cfg.paths.mantra(self._session.plan.mantra_id)
        if not entry.file.exists():
            return None
        try:
            from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
            from PyQt6.QtMultimediaWidgets import QVideoWidget
        except Exception:
            return None
        widget = QVideoWidget()
        widget.setStyleSheet("background:#000000;")
        self._player = QMediaPlayer()
        self._player.setVideoOutput(widget)
        self._player_audio = QAudioOutput()
        self._player.setAudioOutput(self._player_audio)
        # Em ciclo: se a fase do mantra exceder a duração do ficheiro, ele
        # repete. O Om Namo Narayanaya dura 6 min, e uma fase de 10 min
        # deixaria a pessoa em silêncio a partir do minuto 6.
        self._player.setLoops(QMediaPlayer.Loops.Infinite)
        self._mantra_loaded = ""
        self._load_mantra()
        return widget

    def _load_mantra(self) -> None:
        """Põe no leitor o mantra que o painel escolheu.

        Chamado antes de cada arranque de fase e não só na construção: a
        escolha pode mudar entre participantes, e trocar a fonte já a tocar
        cortaria o som a meio.
        """
        if self._video is None and not hasattr(self, "_player"):
            return
        entry = self._cfg.paths.mantra(self._session.plan.mantra_id)
        if entry.id == self._mantra_loaded or not entry.file.exists():
            return
        self._player.setSource(
            QtCore.QUrl.fromLocalFile(str(entry.file.resolve()))
        )
        self._mantra_loaded = entry.id

    def _bind_shortcuts(self) -> None:
        for name, slot in (
            ("inspect_signal", self._open_panel),
            ("toggle_video", self._toggle_video),
            ("abort", self._on_close_clicked),
        ):
            sequence = self._cfg.ui.shortcuts.get(name)
            if not sequence:
                continue
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(sequence), self)
            shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
            shortcut.activated.connect(slot)

    # -- navegação -------------------------------------------------------------- #
    def _show_screen(self, screen: Screen) -> None:
        self._screens.setCurrentWidget(screen)
        self._set_video_visible(False)
        self._refresh_close_tooltip()

    def _refresh_close_tooltip(self) -> None:
        """O X faz coisas diferentes conforme o ecrã, e tem de o dizer."""
        at_welcome = self._screens.currentWidget() is self.welcome
        self._close_button.setToolTip(
            "Fechar" if at_welcome else "Abortar sessão. Retorna ao menu inicial"
        )

    def _go_welcome(self) -> None:
        """Volta ao início, fecha a gravação e liberta o dispositivo.

        Se houver um envio a decorrer, **não é cancelado**: a pessoa deu
        consentimento e escreveu o endereço, portanto o email segue. O que se
        perde é o ecrã onde o resultado apareceria — o resultado vai para
        ``data/app.log`` e para ``contacto.json``, e o ``_send_token`` impede
        que ele escreva no ecrã do participante seguinte.
        """
        if self._send_deadline.isActive():
            log(
                f"X premido durante o envio[{self._send_token}] "
                f"('{self._send_stage}'); o email continua"
            )
            self._send_deadline.stop()
        self._stop_media()
        self._finalize_recording()
        self._device.stop_streaming()
        self._device.disconnect()
        self._session.go_to(Phase.IDLE, "abort")
        self.report.reset()
        # Nada da pessoa anterior pode sobreviver a este ponto: sem isto, um
        # Enviar na sessão seguinte mandava o relatório de quem já saiu.
        self._report_data = None
        self._last_analysis = None
        self._featured = []
        self._contact_path = None
        self._session_started = datetime.now()
        self._show_screen(self.welcome)

    def _go_explain(self) -> None:
        self._show_screen(self.explain)
        self.explain.set_status("a ligar ao equipamento…")
        self._device.connect_async(self._on_connected)

    def _on_connected(self, status) -> None:
        if status.usable:
            # Arrancar a aquisicao JA, e nao so quando se carrega em iniciar.
            #
            # Os electrodos secos levam de 40 a 60 s a fazer contacto: medido
            # na sessao 16_16132913092026, a amplitude comecou em 47,6 uV e so
            # estabilizou em 20 uV ao fim de ~60 s. A calibracao apanhava
            # exactamente essa janela e ficou com 7 epocas boas em 27, contra
            # 78% no resto da sessao — uma gravacao excelente perdida por ter
            # comecado cedo demais.
            #
            # Esse tempo nao pode sair do participante: numa banca ninguem
            # espera um minuto a olhar para o ecra. Mas ele JA existe — e o
            # tempo em que o operador ajusta a touca e explica o protocolo.
            # Basta o sinal estar a correr durante essa conversa.
            self._device.start_streaming()
            self.explain.set_status(
                f"equipamento pronto · {status.device_id}", "#8fd6a0"
            )
        else:
            self.explain.set_status(
                "sem equipamento. Ligue o headset e prima Iniciar demo — a "
                "aplicação volta a tentar. Ctrl+I para o diagnóstico.",
                "#e08b84",
            )

    def _begin_session(self) -> None:
        if not self._device.start_streaming():
            self.explain.set_status(
                "não foi possível iniciar a aquisição. Ligue o headset e tente "
                "de novo. Ctrl+I para o diagnóstico.",
                "#e08b84",
            )
            return
        self._recorder = Recorder(self._cfg, device_short_id=self._device.short_id)
        # Limpar o historico ANTES de comecar. Sem isto, analysed_segments()
        # devolve as fases de todas as sessoes desde que a aplicacao abriu, e
        # como a base de tempo reinicia com o novo Recorder, saem janelas
        # sobrepostas a comecar as tres em zero. Numa sessao real (aparelho 14,
        # 18:21) o raw_meta trazia oito fases de tres sessoes diferentes, a
        # calibracao ficou com ZERO epocas, e todos os marcadores sairam "sem
        # referencia" apesar de haver 41% de cobertura.
        self._session.restart()
        self._session.go_to(Phase.CALIBRATION, "manual")

    def _set_durations(self, calibration: float, mantra: float, settle: float) -> None:
        plan = self._session.plan.with_durations(
            calibration_s=calibration, mantra_s=mantra, settle_s=settle
        )
        try:
            self._session.set_plan(plan)
        except RuntimeError:
            return
        if self._panel is not None:
            self._panel.set_plan(plan)

    def _on_close_clicked(self) -> None:
        """X e Esc: abortar para o início; no início, fechar a aplicação."""
        if self._screens.currentWidget() is self.welcome:
            self.close()
            return
        self._go_welcome()

    # -- fases -------------------------------------------------------------------- #
    def _on_phase_change(self, change: PhaseChange) -> None:
        if self._recorder is not None:
            self._recorder.add_event(
                "phase",
                t_s=change.at_s,
                detail=change.current.value,
                previous=change.previous.value,
                reason=change.reason,
                sample=int(round(change.at_s * self._cfg.device.sfreq_nominal)),
            )

        cues = self._cfg.protocol.cues.mode in ("audio", "both")
        if cues and change.current in (Phase.CALIBRATION, Phase.MANTRA):
            self._audio.cue_start()
        if cues and change.previous in (Phase.CALIBRATION, Phase.MANTRA):
            self._audio.cue_end()

        if change.current == Phase.CALIBRATION:
            self._show_screen(self.calibration)
        elif change.current == Phase.WAIT_START:
            # De olhos fechados não há nada a esperar: segue direto.
            self._session.go_to(Phase.MANTRA, "auto")
        elif change.current == Phase.MANTRA:
            self._show_screen(self.mantra)
            self._start_media()
        elif change.current == Phase.SETTLE:
            self._stop_media()
            self._show_screen(self.rest)
        elif change.current == Phase.ANALYSIS:
            self._end_of_session()

    def _end_of_session(self) -> None:
        """Fim do repouso: pára de recolher, fecha a gravação, mostra o relatório."""
        self._stop_media()
        self._device.stop_streaming()   # o LED volta a piscar
        path = self._finalize_recording()
        if path is not None:
            self._last_session_dir = path
            self._queue_upload(path)
        self._session.go_to(Phase.REPORT, "auto")

        plan = self._session.plan
        self.report.set_summary(
            f"calibração {format_time(plan.calibration_s)} · "
            f"mantra {format_time(plan.mantra_s)} · "
            f"repouso {format_time(plan.settle_s)}"
        )
        self._show_screen(self.report)
        self._run_analysis(path)

    def _queue_upload(self, session_dir: Path) -> None:
        """Arquiva e poe na fila. O local so e apagado depois de confirmado."""
        if self._drive is None:
            return
        try:
            self._uploads.add(session_dir)
        except Exception as exc:  # nunca deixar o envio partir a sessao
            print(f"falha ao arquivar {session_dir}: {exc}")

    def _run_analysis(self, session_dir: Path | None) -> None:
        """Análise post-hoc no fim da sessão, com a gravação já fechada.

        Corre aqui e não durante a sessão: nada de biomarcadores em tempo real
        (SPEC 1.4). Falhar a análise nunca pode partir a demo — se correr mal,
        o ecrã fica com o resumo e a gravação está a salvo na mesma.
        """
        if session_dir is None:
            return
        try:
            import numpy as np_

            raw = np_.load(session_dir / "raw.npy")
            phases = [
                (p.value, a, b) for p, a, b in self._session.analysed_segments()
            ]
            result = analyse(
                raw[EEG_SLICE],
                self._cfg,
                phases,
                accel=raw[ACCEL_SLICE],
                gyro=raw[GYRO_SLICE],
                selected=self._marker_selection,
            )
            self._last_analysis = (result, phases)

            # O destaque escolhe de entre TODOS os marcadores, e nao de entre
            # os que estao marcados no painel: a selecao do painel serve para
            # acrescentar linhas ao ecra final, nao para limitar o top 3.
            if self._cfg.report.featured_mode == "top_variation":
                chosen = rank_by_variation(
                    result, self._cfg, limit=self._cfg.report.featured_count
                )
            else:
                chosen = list(self._cfg.report.featured_markers)
            self._featured = list(chosen)
            self.report.show_results(result, phases, chosen)
            log(f"analise pronta; destaque {chosen}")

            # Preparado agora, com a pessoa ainda a olhar para o ecra: quando
            # ela carregar em Enviar so falta trocar o nome e publicar.
            self._prepare_report()
        except Exception as exc:
            log_exception("_run_analysis", exc)

    # -- multimédia ------------------------------------------------------------- #
    def _start_media(self) -> None:
        if self._video is None:
            return
        self._load_mantra()
        self._player.setPosition(int(self._cfg.paths.mantra_start_offset_s * 1000))
        self._player.play()

    def _stop_media(self) -> None:
        if self._video is not None:
            self._player.stop()
        self._set_video_visible(False)

    def _toggle_video(self) -> None:
        if self._session.phase not in (Phase.MANTRA, Phase.SETTLE):
            return
        if self._video is None:
            return
        self._set_video_visible(not self._video.isVisible())

    def _set_video_visible(self, visible: bool) -> None:
        if self._video is None or self._video.isVisible() == visible:
            return
        # Nunca pausar: esconder a imagem não pode cortar o som.
        self._video.setVisible(visible)
        if self._recorder is not None:
            # A luz do ecrã atravessa as pálpebras e mexe no alfa: cada
            # alternância é uma alteração de condição e fica auditável.
            t = self._device.elapsed_s
            self._recorder.add_event(
                "video_visible",
                t_s=t,
                detail="on" if visible else "off",
                sample=int(round(t * self._cfg.device.sfreq_nominal)),
            )

    # -- gravação ----------------------------------------------------------------- #
    def _on_chunk(self, chunk: np.ndarray) -> None:
        recorder = self._recorder
        if recorder is not None and self._session.phase in RECORDING_PHASES:
            recorder.write(chunk)

    def _finalize_recording(self) -> Path | None:
        recorder, self._recorder = self._recorder, None
        if recorder is None:
            return None
        info = self._device.info
        acquisition = self._device.acquisition
        if info is None or acquisition is None:
            recorder.abort()
            return None
        window = self._device.peek_last(
            self._cfg.montage.validation.auto_check_window_s
        )
        validation = (
            summarize(auto_checks(window[0:8], self._cfg)) if window.shape[1] > 2 else {}
        )
        return recorder.finalize(
            info,
            measured_sfreq=acquisition.stats.measured_sfreq,
            acquisition_stats=acquisition.stats,
            montage_validation=validation,
            extra_meta={
                "plan": self._session.plan.as_dict(),
                "phases": [
                    {"phase": p.value, "start_s": a, "end_s": b}
                    for p, a, b in self._session.analysed_segments()
                ],
                "aborted": self._session.aborted,
                "marker_selection": self._marker_selection,
            },
        )

    def _store_email(self, name: str, address: str) -> None:
        """Guarda o email **separado** do sinal (SPEC 11) e dispara o envio.

        O EEG é dado biométrico; mantê-lo desligado do identificador direto é
        o que mantém o enquadramento de RGPD simples. O ficheiro fica com
        ``sent: false`` até o envio confirmar, para nunca haver dúvida sobre o
        que saiu da máquina.

        Nada aqui pode levantar. Uma exceção num *slot* do Qt, num executável
        sem consola, deixava o ecrã parado em "a preparar o relatório" sem uma
        única pista — foi exatamente o que aconteceu numa sessão real.
        """
        try:
            self._contact_path = self._reports_dir() / "contacto.json"
            self._contact_path.write_text(
                json.dumps(
                    {
                        "nome": name,
                        "email": address,
                        "consent": True,
                        "purpose": str(self._cfg.ui.email.get("purpose", "")).strip(),
                        "retention_days": self._cfg.ui.email.get("retention_days"),
                        "collected_utc": datetime.now(timezone.utc).isoformat(),
                        "session": (
                            self._last_session_dir.name
                            if self._last_session_dir
                            else ""
                        ),
                        "sent": False,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            log_exception("_store_email", exc)

        try:
            self._send_report(name, address)
        except Exception as exc:
            log_exception("_send_report", exc)
            self.report.set_email_status(f"Não foi possível enviar: {exc}", False)

    # -- relatório ------------------------------------------------------------------ #
    def _reports_dir(self) -> Path:
        """Onde o relatório vive.

        **Não é a pasta da sessão.** Com ``delete_after_upload`` ligada, a
        fila do Drive apaga essa pasta assim que o arquivo sobe — e o envio,
        que acontece enquanto a pessoa escreve o email, ia escrever num sítio
        que já não existe. Esta pasta é só do relatório e ninguém lhe toca.
        """
        code = self._report_code()
        folder = self._cfg.paths.sessions_dir.parent / "reports" / code
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _report_code(self) -> str:
        started = self._report_started()
        suffix = (
            self._last_session_dir.name.split("_")[-1][:6]
            if self._last_session_dir
            else f"{started:%H%M%S}"
        )
        return f"FBE-{started:%Y}-{suffix}"

    def _report_started(self) -> datetime:
        if self._last_session_dir is not None:
            try:
                return datetime.fromtimestamp(self._last_session_dir.stat().st_mtime)
            except OSError:
                pass
        return self._session_started

    def _report_context(self, name: str) -> ReportContext:
        """Quem, quando, quanto tempo, e o que tocou."""
        started = self._report_started()
        mantra = self._cfg.paths.mantra(self._session.plan.mantra_id)
        return ReportContext(
            participant_name=name or "Participante",
            session_started=started,
            duration_s=self._session.plan.total_s,
            mantra_label=mantra.label,
            session_code=self._report_code(),
            festival_label=f"{self._cfg.ui.title} · neroes",
        )

    def _prepare_report(self) -> None:
        """Constrói o relatório **assim que a análise acaba**.

        Isto corre antes de alguém escrever o email, e é o que torna o envio
        uma operação de segundos: quando a pessoa carrega em Enviar já só
        falta trocar o nome, publicar e mandar. Antes o relatório inteiro era
        construído dentro do clique, no meio da rede, sem nada no ecrã a dizer
        em que passo estava.
        """
        if self._last_analysis is None:
            return
        started = time.monotonic()
        try:
            result, phases = self._last_analysis
            featured = rank_by_variation(
                result, self._cfg, limit=self._cfg.report.featured_count
            )
            data = build_report(
                result, self._cfg, self._report_context(""), featured, phases
            )
            folder = self._reports_dir()
            (folder / "report.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._report_data = data
            log(
                f"relatorio preparado em {time.monotonic() - started:.1f}s "
                f"-> {folder}"
            )
        except Exception as exc:
            self._report_data = None
            log_exception("_prepare_report", exc)

    def _render_report_for(self, name: str) -> Path:
        """A página com o nome da pessoa. Só troca o nome — é instantâneo."""
        if self._report_data is None:
            raise MailError("o relatório não foi gerado; ver data/app.log")
        data = dict(self._report_data)
        data["participant"] = {"name": name or "Participante"}
        path = self._reports_dir() / f"{self._report_code()}.html"
        path.write_text(render_report(data), encoding="utf-8")
        return path

    def _report_link(self, path: Path, code: str) -> str:
        """O link que vai no email, conforme o modo configurado.

        Levanta em vez de recorrer a um modo pior em silêncio: um email com
        um link que descarrega, quando o operador configurou o Pages, é mais
        difícil de detetar do que um envio que falha e diz porquê.
        """
        email_cfg = self._cfg.ui.email
        mode = str(email_cfg.get("link_mode", "drive"))

        if mode == "github_pages":
            target = GitHubTarget(
                owner=self._cfg.publish.owner,
                repo=self._cfg.publish.repo,
                branch=self._cfg.publish.branch,
                folder=self._cfg.publish.folder,
                token=self._cfg.publish.token,
                base_url=self._cfg.publish.base_url,
            )
            return publish_to_github(target, path, code)

        base = str(email_cfg.get("public_base_url", "")).strip().rstrip("/")
        if mode == "base_url":
            if not base:
                raise MailError(
                    "link_mode é 'base_url' mas public_base_url está vazio."
                )
            return f"{base}/{path.name}"

        if self._drive is None or not self._drive.signed_in:
            raise MailError(
                "sem sessão iniciada no Google: não há onde publicar o "
                "relatório. Abra o painel (Ctrl+I) e inicie sessão."
            )
        file_id = self._drive.upload(
            path, self._cfg.drive.folder_id, name=f"{code}.html", mime="text/html"
        )
        return self._drive.share_with_link(file_id)

    def _should_attach(self) -> bool:
        """O anexo só faz sentido quando o link não abre sozinho."""
        setting = self._cfg.ui.email.get("attach_report", "auto")
        if isinstance(setting, bool):
            return setting
        if str(setting).strip().lower() in ("true", "sim", "1"):
            return True
        if str(setting).strip().lower() in ("false", "nao", "não", "0"):
            return False
        return str(self._cfg.ui.email.get("link_mode", "drive")) == "drive"

    # -- envio -------------------------------------------------------------------- #
    def _send_report(self, name: str, address: str) -> None:
        """Publica e envia. Fora da thread da interface, com prazo."""
        if not valid_address(address):
            self.report.set_email_status("Endereço inválido.", False)
            return
        if self._report_data is None:
            # Última tentativa: a análise pode ter corrido e a preparação ter
            # falhado. Vale mais tentar agora do que desistir em silêncio.
            self._prepare_report()
        if self._report_data is None:
            self.report.set_email_status(
                "Sem relatório para enviar. A gravação está guardada.", False
            )
            return

        self._send_token += 1
        token = self._send_token
        timeout_s = float(self._cfg.ui.email.get("timeout_s", 120))

        def stage(text: str) -> None:
            log(f"envio[{token}] {text}")
            self.report_progress.emit(token, text)

        def work() -> None:
            try:
                stage("a preparar a página")
                path = self._render_report_for(name)
                stage("a publicar o relatório")
                link = self._report_link(path, self._report_code())
                if str(self._cfg.ui.email.get("link_mode", "")) == "github_pages":
                    stage("à espera que o site publique")
                    if not wait_until_live(link):
                        log(f"envio[{token}] o Pages ainda nao servia {link}")
                stage("a enviar o email")
                email_cfg = self._cfg.ui.email
                message = compose(
                    to=address,
                    name=name,
                    link=link,
                    template=str(email_cfg.get("body", "")),
                    subject=str(email_cfg.get("subject", "Relatório")),
                    logo=self._cfg.paths.logo_lockup,
                    attachments=(path,) if self._should_attach() else (),
                )
                GmailSender(
                    self._drive, str(email_cfg.get("sender_name", ""))
                ).send(message)
            except (MailError, DriveError, PublishError) as exc:
                log(f"envio[{token}] FALHOU: {exc}")
                self.report_sent.emit(token, False, str(exc))
            except Exception as exc:
                log_exception(f"envio[{token}]", exc)
                self.report_sent.emit(token, False, f"falha inesperada: {exc}")
            else:
                log(f"envio[{token}] enviado para {address}")
                self.report_sent.emit(token, True, address)

        self.report.set_email_status("A preparar…")
        self._send_stage = "a começar"
        # O prazo é o que impede uma espera infinita. Sem isto, uma chamada de
        # rede que nunca responde deixa o ecrã a dizer "a preparar" para
        # sempre, e foi o que aconteceu.
        self._send_deadline.start(int(timeout_s * 1000))
        threading.Thread(target=work, daemon=True, name=f"envio-{token}").start()

    def _on_send_progress(self, token: int, text: str) -> None:
        if token != self._send_token:
            return
        self._send_stage = text
        self.report.set_email_status(f"{text}…")

    def _on_send_timeout(self) -> None:
        """O envio passou do prazo. Diz onde ficou, em vez de ficar parado."""
        log(f"envio[{self._send_token}] excedeu o prazo em '{self._send_stage}'")
        self.report.set_email_status(
            f"Demorou demais ({self._send_stage}). Verifique a ligação e "
            f"tente outra vez; a gravação está guardada.",
            False,
        )

    def _on_report_sent(self, token: int, ok: bool, detail: str) -> None:
        self._send_deadline.stop()
        if ok and self._contact_path is not None:
            try:
                payload = json.loads(self._contact_path.read_text(encoding="utf-8"))
                payload["sent"] = True
                payload["sent_utc"] = datetime.now(timezone.utc).isoformat()
                self._contact_path.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as exc:
                log_exception("marcar contacto como enviado", exc)
        if token != self._send_token:
            # Resultado de um participante anterior: o email seguiu na mesma
            # (o consentimento foi dado), mas o ecrã já é de outra pessoa.
            return
        if ok:
            self.report.set_email_status(f"Enviado para {detail}.", True)
        else:
            self.report.set_email_status(f"Não foi enviado: {detail}", False)

    # -- painel do investigador ----------------------------------------------------- #
    def _open_panel(self) -> None:
        if self._panel is not None and self._panel.isVisible():
            self._panel.raise_()
            self._panel.activateWindow()
            return
        panel = ResearcherPanel(
            self._cfg, self._device, self._session.plan, self,
            drive=self._drive, uploads=self._uploads,
        )
        panel.plan_changed.connect(self._panel_plan_changed)
        panel.selection_changed.connect(self._on_selection_changed)
        panel.apply_requested.connect(self._apply_markers_to_report)
        panel.set_selection(self._marker_selection)
        panel.reconnect_requested.connect(self._reconnect)
        # 1300x860 nao cabe num portatil de 1366x768. Limita-se ao que o
        # ecra tem, e o scroll do painel trata do resto.
        available = self.screen().availableGeometry() if self.screen() else None
        if available is not None:
            panel.resize(
                min(1300, int(available.width() * 0.95)),
                min(860, int(available.height() * 0.90)),
            )
        else:
            panel.resize(1300, 860)
        panel.show()
        self._panel = panel

    def _on_selection_changed(self, selection: dict) -> None:
        self._marker_selection = dict(selection)

    def _apply_markers_to_report(self, marker_ids: list) -> None:
        """Acrescenta ao ecrã final os marcadores marcados no painel.

        O destaque continua lá: o botão **acrescenta**, não substitui. Um
        painel que apagasse o top 3 obrigaria o investigador a voltar a
        marcá-lo à mão a cada participante.
        """
        if self._last_analysis is None:
            log("Adicionar: ainda nao ha analise")
            return
        result, phases = self._last_analysis
        chosen = list(self._featured)
        chosen += [m for m in marker_ids if m not in chosen]
        try:
            self.report.show_results(result, phases, chosen)
            log(f"ecra final atualizado com {len(chosen)} marcadores: {chosen}")
        except Exception as exc:
            log_exception("_apply_markers_to_report", exc)

    def _panel_plan_changed(self, plan: SessionPlan) -> None:
        try:
            self._session.set_plan(plan)
        except RuntimeError:
            pass

    def _reconnect(self) -> None:
        """Ligar ou reiniciar o dispositivo a partir do painel."""
        was_streaming = self._device.status().state in (
            DeviceState.STREAMING,
            DeviceState.STALE,
        )

        def after(_status) -> None:
            if was_streaming:
                self._device.start_streaming()

        self._device.connect_async(after)

    # -- relógio --------------------------------------------------------------------- #
    def _refresh_explain_status(self) -> None:
        """Diz ao operador quando os sensores estao assentes.

        Em linguagem de banca, nao de laboratorio: a pessoa esta a olhar para
        este ecra. "A preparar os sensores" e "Sensores prontos" chegam.
        """
        if self._screens.currentWidget() is not self.explain:
            return
        status = self._device.status()
        if not status.usable:
            return
        remaining = self._device.settle_remaining_s()
        if remaining > 0:
            self.explain.set_status(
                f"a preparar os sensores… {remaining:.0f} s", "#c99a1e"
            )
        else:
            self.explain.set_status("sensores prontos", "#8fd6a0")

    def _on_tick(self) -> None:
        self._refresh_explain_status()
        self._refresh_explain_status()
        self._session.tick(self._device.elapsed_s)
        phase = self._session.phase
        screen = {
            Phase.CALIBRATION: self.calibration,
            Phase.MANTRA: self.mantra,
            Phase.SETTLE: self.rest,
        }.get(phase)
        if screen is None:
            return

        # Os ~10 s de assentamento do amplificador nao sao calibracao: o relogio
        # da sessao so arranca depois, porque conta amostras. Mostra-se com a
        # barra verde, e so quando ela acaba e que o contador e a barra azul
        # aparecem.
        status = self._device.status()
        if phase == Phase.CALIBRATION and not status.settled:
            total = max(self._cfg.device.settle_discard_s, 1e-6)
            self.calibration.set_warmup(1.0 - status.settle_remaining_s / total)
            return
        if phase == Phase.CALIBRATION:
            self.calibration.set_warmup(None)
        screen.update_time(self._session.remaining_s, self._session.progress)

    def _refresh_explain_status(self) -> None:
        """Mantem a mensagem do ecra de instrucoes a par do estado real.

        Sem isto, ligar o headset pelo Ctrl+I deixava na ecra a mensagem
        negativa do arranque, que ja nao era verdade.
        """
        if self._screens.currentWidget() is not self.explain:
            return
        status = self._device.status()
        if status.state == DeviceState.CONNECTING:
            self.explain.set_status("a procurar o equipamento…", "#c99a1e")
        elif status.usable:
            self.explain.set_status(
                f"equipamento pronto · {status.device_id or status.source_name}",
                "#8fd6a0",
            )
        else:
            self.explain.set_status(
                "sem equipamento. Ligue o headset e prima Iniciar demo, ou use "
                "Ctrl+I para ligar.",
                "#e08b84",
            )

    # -- fecho ------------------------------------------------------------------------ #
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self._tick.stop()
        self._stop_media()
        self._finalize_recording()
        if self._panel is not None:
            self._panel.close()
        self._uploads.stop()
        self._device.disconnect()
        super().closeEvent(event)

    def show_window(self) -> None:
        if self._fullscreen:
            self.showFullScreen()
        else:
            self.resize(1400, 900)
            self.show()
