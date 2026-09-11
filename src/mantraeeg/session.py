"""Máquina de estados do protocolo (SPEC 2).

Deliberadamente **sem Qt**: a sequência de fases, as durações e as transições são
lógica pura e testável sem abrir uma janela. A camada de interface limita-se a
chamar :meth:`SessionController.tick` num temporizador e a desenhar o que o
controlador diz.

[DESVIO] A SPEC 6 põe isto em ``ui/session.py``. Vive aqui porque não depende da
interface, e é o que permite testá-lo sem ecrã.

Protocolo de **olhos fechados** (ver ``config.protocol``): a pessoa fecha os
olhos ao início da calibração e mantém-nos fechados até ao fim da meditação. A
invariante ocular é validada na carga da config — comparar uma calibração de
olhos abertos com uma meditação de olhos fechados produziria uma "melhoria" que
é inteiramente efeito de Berger.

Como a pessoa não vê o ecrã, as deixas de início e fim são **áudio**. O
temporizador no ecrã é para o operador e para quem assiste à banca.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Iterator

from .config import Config, EyeCondition


class Phase(str, Enum):
    """Fases da sessão. Os valores vão para ``events.json``."""

    IDLE = "IDLE"
    CONNECT = "CONNECT"
    CONTACT = "CONTACT"
    MENU = "MENU"
    CALIBRATION = "CALIBRATION"
    WAIT_START = "WAIT_START"
    MANTRA = "MANTRA"
    SETTLE = "SETTLE"
    ANALYSIS = "ANALYSIS"
    REPORT = "REPORT"


#: Fases cronometradas, que avançam sozinhas quando o tempo acaba. As restantes
#: esperam por uma ação do operador.
TIMED_PHASES = (Phase.CALIBRATION, Phase.MANTRA, Phase.SETTLE)

#: Fases cujo sinal entra na análise post-hoc.
ANALYSED_PHASES = (Phase.CALIBRATION, Phase.MANTRA, Phase.SETTLE)

ORDER: tuple[Phase, ...] = (
    Phase.IDLE,
    Phase.CONNECT,
    Phase.CONTACT,
    Phase.MENU,
    Phase.CALIBRATION,
    Phase.WAIT_START,
    Phase.MANTRA,
    Phase.SETTLE,
    Phase.ANALYSIS,
    Phase.REPORT,
)


@dataclass(frozen=True)
class SessionPlan:
    """As durações efetivamente usadas nesta sessão.

    Editáveis na interface (SPEC 12); os valores usados vão para
    ``raw_meta.json``, não os defaults da config.
    """

    calibration_s: float
    calib_use_last_s: float
    mantra_s: float
    settle_s: float
    eye_calibration: EyeCondition
    eye_mantra: EyeCondition
    eye_settle: EyeCondition
    #: O ``id`` do mantra escolhido no painel. Viaja com o plano para acabar
    #: em ``raw_meta.json``: o relatório não pode nomear um mantra a partir do
    #: default da config quando o operador escolheu outro.
    mantra_id: str = ""

    @classmethod
    def from_config(cls, cfg: Config) -> "SessionPlan":
        p = cfg.protocol
        return cls(
            calibration_s=p.calibration_s,
            calib_use_last_s=p.calib_use_last_s,
            mantra_s=p.mantra_s,
            settle_s=p.settle_s,
            eye_calibration=p.eye_calibration,
            eye_mantra=p.eye_mantra,
            eye_settle=p.eye_settle,
            mantra_id=cfg.paths.default_mantra,
        )

    def with_durations(
        self,
        calibration_s: float | None = None,
        mantra_s: float | None = None,
        settle_s: float | None = None,
    ) -> "SessionPlan":
        """Novo plano com as durações alteradas no menu.

        ``calib_use_last_s`` acompanha a calibração: nunca pode excedê-la, e
        mantém a mesma proporção de cauda útil.
        """
        calibration = self.calibration_s if calibration_s is None else calibration_s
        fraction = (
            self.calib_use_last_s / self.calibration_s if self.calibration_s > 0 else 0.5
        )
        return replace(
            self,
            calibration_s=calibration,
            calib_use_last_s=min(calibration, calibration * fraction),
            mantra_s=self.mantra_s if mantra_s is None else mantra_s,
            settle_s=self.settle_s if settle_s is None else settle_s,
        )

    def duration_of(self, phase: Phase) -> float | None:
        """Duração de uma fase cronometrada, ou ``None`` se avançar à mão."""
        return {
            Phase.CALIBRATION: self.calibration_s,
            Phase.MANTRA: self.mantra_s,
            Phase.SETTLE: self.settle_s,
        }.get(phase)

    def eye_of(self, phase: Phase) -> EyeCondition | None:
        return {
            Phase.CALIBRATION: self.eye_calibration,
            Phase.MANTRA: self.eye_mantra,
            Phase.SETTLE: self.eye_settle,
        }.get(phase)

    @property
    def settle_is_baseline(self) -> bool:
        """O pós-mantra é longo o suficiente para servir de 2ª linha de base."""
        return self.settle_s >= 45.0

    @property
    def total_s(self) -> float:
        return self.calibration_s + self.mantra_s + self.settle_s

    def as_dict(self) -> dict[str, float | str]:
        return {
            "calibration_s": self.calibration_s,
            "calib_use_last_s": self.calib_use_last_s,
            "mantra_s": self.mantra_s,
            "settle_s": self.settle_s,
            "eye_calibration": self.eye_calibration,
            "eye_mantra": self.eye_mantra,
            "eye_settle": self.eye_settle,
            # O que tocou, e não o default da config: o relatório nomeia o
            # mantra a partir daqui.
            "mantra_id": self.mantra_id,
        }


@dataclass(frozen=True)
class PhaseChange:
    """Uma transição, para o registo e para a interface."""

    previous: Phase
    current: Phase
    at_s: float
    reason: str  # "manual" | "timeout" | "abort"


class SessionController:
    """Conduz a sessão pelas fases, cronometrando as que têm duração.

    Não sabe nada de Qt nem de aquisição. O relógio é injetado: ``tick(now_s)``
    recebe o tempo decorrido de sinal gravado, o que faz as fases alinharem-se
    com as amostras e não com o relógio de parede da interface.
    """

    def __init__(
        self,
        plan: SessionPlan,
        on_change: Callable[[PhaseChange], None] | None = None,
    ) -> None:
        self._plan = plan
        self._on_change = on_change
        self._phase = Phase.IDLE
        self._phase_started_s = 0.0
        self._now_s = 0.0
        self._history: list[PhaseChange] = []
        self._aborted = False

    # -- estado -------------------------------------------------------------- #
    @property
    def plan(self) -> SessionPlan:
        return self._plan

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def aborted(self) -> bool:
        return self._aborted

    @property
    def history(self) -> list[PhaseChange]:
        return list(self._history)

    @property
    def elapsed_in_phase_s(self) -> float:
        return max(self._now_s - self._phase_started_s, 0.0)

    @property
    def remaining_s(self) -> float | None:
        """Segundos até a fase acabar, ou ``None`` se ela esperar pelo operador."""
        duration = self._plan.duration_of(self._phase)
        if duration is None:
            return None
        return max(duration - self.elapsed_in_phase_s, 0.0)

    @property
    def progress(self) -> float | None:
        duration = self._plan.duration_of(self._phase)
        if not duration:
            return None
        return min(self.elapsed_in_phase_s / duration, 1.0)

    @property
    def is_recording_phase(self) -> bool:
        """As fases cujo sinal entra na análise."""
        return self._phase in ANALYSED_PHASES

    def set_plan(self, plan: SessionPlan) -> None:
        """Só permitido antes de a calibração começar (o menu é que edita)."""
        if self._phase not in (Phase.IDLE, Phase.CONNECT, Phase.CONTACT, Phase.MENU):
            raise RuntimeError(
                f"as durações não podem mudar durante {self._phase.value}"
            )
        self._plan = plan

    # -- transições ----------------------------------------------------------- #
    def tick(self, now_s: float) -> PhaseChange | None:
        """Avança o relógio. Devolve a transição, se o tempo da fase acabou."""
        self._now_s = now_s
        if self._aborted or self._phase not in TIMED_PHASES:
            return None
        remaining = self.remaining_s
        if remaining is not None and remaining <= 0.0:
            return self._go(self.next_phase(), "timeout")
        return None

    def advance(self) -> PhaseChange | None:
        """Passa à fase seguinte por ação do operador."""
        if self._aborted:
            return None
        return self._go(self.next_phase(), "manual")

    def go_to(self, phase: Phase, reason: str = "manual") -> PhaseChange | None:
        return self._go(phase, reason)

    def abort(self) -> PhaseChange | None:
        """Interrompe a sessão. O que já foi gravado mantém-se."""
        self._aborted = True
        return self._go(Phase.IDLE, "abort")

    def restart(self) -> None:
        """Nova sessão sem reiniciar a aplicação nem reconectar (SPEC 8.1)."""
        self._aborted = False
        self._history.clear()
        self._phase = Phase.CONTACT
        self._phase_started_s = self._now_s

    def next_phase(self) -> Phase:
        index = ORDER.index(self._phase)
        return ORDER[min(index + 1, len(ORDER) - 1)]

    def _go(self, phase: Phase, reason: str) -> PhaseChange | None:
        if phase == self._phase:
            return None
        change = PhaseChange(self._phase, phase, self._now_s, reason)
        self._phase = phase
        self._phase_started_s = self._now_s
        self._history.append(change)
        if self._on_change is not None:
            self._on_change(change)
        return change

    # -- consulta -------------------------------------------------------------- #
    def analysed_segments(self) -> Iterator[tuple[Phase, float, float]]:
        """``(fase, início, fim)`` das fases analisadas, em segundos de gravação.

        Reconstruído do histórico: é isto que a análise post-hoc usa para saber
        onde começa a calibração e onde acaba o mantra.
        """
        starts: dict[Phase, float] = {}
        for change in self._history:
            if change.previous in ANALYSED_PHASES and change.previous in starts:
                yield change.previous, starts[change.previous], change.at_s
            if change.current in ANALYSED_PHASES:
                starts[change.current] = change.at_s
        if self._phase in ANALYSED_PHASES and self._phase in starts:
            yield self._phase, starts[self._phase], self._now_s
