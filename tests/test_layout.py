"""Nada fica inalcançável em nenhuma resolução plausível de portátil.

Num portátil de 1366x768 o botão de iniciar ficava fora do ecrã, o bloco do
consentimento e os campos do email ficavam cortados — o participante não
conseguia sequer pedir o relatório — e a parte de baixo do painel do
investigador não tinha forma de ser alcançada. Margens fixas de 90/70 px mais
250 px de espaçadores rígidos não cabem em 768 px de altura.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6 import QtWidgets  # noqa: E402

from mantraeeg.ui.screens import ExplainScreen, ReportScreen, WelcomeScreen  # noqa: E402

#: Resoluções a cobrir. A mais pequena é deliberadamente menor do que
#: qualquer portátil real: se passar aqui, passa na banca.
SIZES = [(1920, 1080), (1366, 768), (1280, 720), (1024, 640)]


@pytest.fixture(scope="module")
def qt_app():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.mark.parametrize("width,height", SIZES)
@pytest.mark.parametrize("name", ["welcome", "explain", "report"])
def test_every_screen_is_reachable(qt_app, cfg, name, width, height):
    screen = {
        "welcome": WelcomeScreen,
        "explain": ExplainScreen,
        "report": ReportScreen,
    }[name](cfg)
    screen.resize(width, height)
    screen.show()
    qt_app.processEvents()

    content = screen._scroll.widget()
    needed = content.sizeHint().height()
    viewport = screen._scroll.viewport().height()
    reach = screen._scroll.verticalScrollBar().maximum() + viewport

    assert reach >= needed, (
        f"{name} a {width}x{height}: precisa de {needed} px e so chega a {reach}"
    )
    screen.close()


def test_explain_fits_without_scrolling_on_a_small_laptop(qt_app, cfg):
    """O ecrã das durações tem de caber inteiro: é onde se carrega em iniciar.

    Ter scroll é a rede de segurança, não o plano. Se este ecrã precisar de
    deslizar para se ver o botão, alguém vai ficar à espera sem perceber
    porquê.
    """
    screen = ExplainScreen(cfg)
    screen.resize(1280, 720)
    screen.show()
    qt_app.processEvents()
    needed = screen._scroll.widget().sizeHint().height()
    assert needed <= screen._scroll.viewport().height(), (
        f"o ecra de explicacao precisa de {needed} px em 720"
    )
    screen.close()
