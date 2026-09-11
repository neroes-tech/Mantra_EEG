"""Uma sessão nova não pode herdar as fases da anterior.

Numa sessão real (aparelho 14, 18:21) o ``raw_meta.json`` trouxe **oito**
fases de três sessões diferentes, todas a começar em zero porque a base de
tempo reinicia com o novo ficheiro. As janelas ficaram sobrepostas, a
calibração ficou com zero épocas, e os oito marcadores saíram "sem
referência" apesar de a gravação ter 41 % de cobertura.
"""

from __future__ import annotations

from mantraeeg.session import Phase, SessionController, SessionPlan


def _run(controller: SessionController, calibration: float, mantra: float) -> None:
    controller.go_to(Phase.CALIBRATION, "manual")
    controller.tick(calibration)
    controller.go_to(Phase.MANTRA, "auto")
    controller.tick(mantra)
    controller.go_to(Phase.SETTLE, "timeout")
    controller.tick(15.0)
    controller.go_to(Phase.ANALYSIS, "timeout")


def test_restart_drops_the_previous_session(cfg):
    controller = SessionController(SessionPlan.from_config(cfg), on_change=lambda c: None)

    _run(controller, 30.0, 60.0)
    first = list(controller.analysed_segments())
    assert first, "a primeira sessão não produziu fases"

    # Nova sessão, como _begin_session faz: restart() antes de arrancar.
    controller.restart()
    _run(controller, 60.0, 120.0)
    second = list(controller.analysed_segments())

    phases = [p for p, _, _ in second]
    assert phases.count(Phase.CALIBRATION) == 1, second
    assert phases.count(Phase.MANTRA) == 1, second

    windows = sorted((a, b) for _, a, b in second)
    for (_, end), (start, _) in zip(windows, windows[1:]):
        assert start >= end, f"janelas sobrepostas: {second}"


def test_restart_clears_the_abort_flag(cfg):
    """Uma sessão abortada não pode marcar a seguinte como abortada."""
    controller = SessionController(SessionPlan.from_config(cfg), on_change=lambda c: None)
    controller.go_to(Phase.CALIBRATION, "manual")
    controller.abort()
    assert controller.aborted

    controller.restart()
    _run(controller, 60.0, 120.0)
    assert not controller.aborted
