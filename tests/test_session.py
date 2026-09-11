"""Máquina de estados do protocolo. Lógica pura, testada sem abrir uma janela."""

from __future__ import annotations

import pytest

from mantraeeg.session import (
    ANALYSED_PHASES,
    ORDER,
    Phase,
    SessionController,
    SessionPlan,
)


@pytest.fixture
def plan(cfg) -> SessionPlan:
    return SessionPlan.from_config(cfg)


@pytest.fixture
def controller(plan) -> SessionController:
    return SessionController(plan)


def run_to(controller: SessionController, phase: Phase, t: float = 0.0) -> None:
    controller.go_to(phase)
    controller.tick(t)


# --------------------------------------------------------------------------- #
# Plano
# --------------------------------------------------------------------------- #
def test_plan_comes_from_config(plan, cfg):
    assert plan.calibration_s == cfg.protocol.calibration_s
    assert plan.mantra_s == cfg.protocol.mantra_s
    assert plan.eye_calibration == plan.eye_mantra == "closed"


def test_editing_durations_keeps_the_calibration_tail_proportional(plan):
    """SPEC 12: alterar durações na interface altera o protocolo."""
    doubled = plan.with_durations(calibration_s=plan.calibration_s * 2)
    assert doubled.calibration_s == plan.calibration_s * 2
    assert doubled.calib_use_last_s == pytest.approx(plan.calib_use_last_s * 2)
    assert doubled.calib_use_last_s <= doubled.calibration_s


def test_calib_tail_never_exceeds_the_calibration(plan):
    shrunk = plan.with_durations(calibration_s=10.0)
    assert shrunk.calib_use_last_s <= shrunk.calibration_s


def test_settle_becomes_a_baseline_only_when_long_enough(plan):
    assert not plan.with_durations(settle_s=15.0).settle_is_baseline
    assert plan.with_durations(settle_s=60.0).settle_is_baseline


def test_plan_serialises_the_eye_condition_with_the_durations(plan):
    """A condição ocular viaja com os dados (SPEC 2)."""
    payload = plan.as_dict()
    assert payload["eye_calibration"] == "closed"
    assert payload["eye_mantra"] == "closed"
    assert payload["calibration_s"] == plan.calibration_s


# --------------------------------------------------------------------------- #
# Transições
# --------------------------------------------------------------------------- #
def test_starts_idle(controller):
    assert controller.phase == Phase.IDLE
    assert controller.remaining_s is None


def test_advance_follows_the_protocol_order(controller):
    seen = [controller.phase]
    for _ in range(len(ORDER) - 1):
        controller.advance()
        seen.append(controller.phase)
    assert seen == list(ORDER)


def test_timed_phase_advances_on_its_own(controller, plan):
    run_to(controller, Phase.CALIBRATION)
    assert controller.tick(plan.calibration_s - 1) is None
    change = controller.tick(plan.calibration_s + 0.1)
    assert change is not None
    assert change.reason == "timeout"
    assert controller.phase == Phase.WAIT_START


def test_untimed_phase_waits_for_the_operator(controller):
    run_to(controller, Phase.WAIT_START)
    assert controller.remaining_s is None
    assert controller.tick(10_000) is None
    assert controller.phase == Phase.WAIT_START


def test_remaining_and_progress_track_the_clock(controller, plan):
    run_to(controller, Phase.MANTRA)
    controller.tick(plan.mantra_s / 2)
    assert controller.remaining_s == pytest.approx(plan.mantra_s / 2)
    assert controller.progress == pytest.approx(0.5)


def test_abort_stops_everything(controller):
    run_to(controller, Phase.MANTRA)
    controller.abort()
    assert controller.aborted
    assert controller.phase == Phase.IDLE
    assert controller.tick(10_000) is None
    assert controller.advance() is None


def test_restart_returns_to_contact_without_reconnecting(controller):
    """Nova sessão sem reiniciar a aplicação (SPEC 8.1)."""
    run_to(controller, Phase.REPORT)
    controller.restart()
    assert controller.phase == Phase.CONTACT
    assert not controller.aborted
    assert controller.history == []


def test_durations_cannot_change_once_the_session_is_running(controller, plan):
    controller.go_to(Phase.MENU)
    controller.set_plan(plan.with_durations(mantra_s=120.0))  # permitido
    controller.go_to(Phase.CALIBRATION)
    with pytest.raises(RuntimeError, match="não podem mudar"):
        controller.set_plan(plan.with_durations(mantra_s=60.0))


# --------------------------------------------------------------------------- #
# Segmentos para a análise
# --------------------------------------------------------------------------- #
def test_analysed_segments_reconstruct_the_phase_boundaries(controller, plan):
    """É daqui que a análise sabe onde começa a calibração e acaba o mantra."""
    controller.tick(0.0)
    controller.go_to(Phase.CALIBRATION)
    controller.tick(plan.calibration_s + 0.1)          # -> WAIT_START
    controller.go_to(Phase.MANTRA)
    controller.tick(plan.calibration_s + plan.mantra_s + 0.2)  # -> SETTLE
    controller.tick(plan.calibration_s + plan.mantra_s + plan.settle_s + 0.3)

    segments = {p: (a, b) for p, a, b in controller.analysed_segments()}
    assert Phase.CALIBRATION in segments
    assert Phase.MANTRA in segments
    start, end = segments[Phase.CALIBRATION]
    assert end - start == pytest.approx(plan.calibration_s, abs=0.5)
    start, end = segments[Phase.MANTRA]
    assert end - start == pytest.approx(plan.mantra_s, abs=0.5)


def test_only_analysed_phases_are_reported(controller, plan):
    controller.tick(0.0)
    for phase in (Phase.CONTACT, Phase.MENU, Phase.CALIBRATION):
        controller.go_to(phase)
    controller.tick(plan.calibration_s + 1)
    phases = {p for p, _, _ in controller.analysed_segments()}
    assert phases <= set(ANALYSED_PHASES)
    assert Phase.MENU not in phases


def test_phase_changes_are_recorded_with_a_reason(controller, plan):
    changes: list = []
    controller_with_hook = SessionController(plan, on_change=changes.append)
    controller_with_hook.go_to(Phase.CALIBRATION)
    controller_with_hook.tick(plan.calibration_s + 1)
    assert [c.reason for c in changes] == ["manual", "timeout"]
    assert changes[-1].current == Phase.WAIT_START
