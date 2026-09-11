"""O relatório do participante não pode dizer o que não mediu.

Três invariantes, todas de honestidade e não de formatação. São as que, se
partirem em silêncio, mandam para casa uma pessoa com uma conclusão errada
sobre si própria.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from mantraeeg.analysis import analyse
from mantraeeg.markers import REGISTRY
from mantraeeg.report.html import render
from mantraeeg.report.payload import ReportContext, build


@pytest.fixture()
def session(cfg):
    """Uma sessão sintética: ruído estacionário, sem efeito nenhum.

    A resposta correta a este sinal é "não mudou nada", e é isso que os testes
    exigem que o relatório diga.
    """
    rng = np.random.default_rng(20260910)
    sfreq = cfg.device.sfreq_nominal
    total_s = 200.0
    n = int(total_s * sfreq)
    raw = rng.normal(0.0, 12.0, size=(8, n))
    phases = [("CALIBRATION", 0.0, 100.0), ("MANTRA", 100.0, total_s)]
    result = analyse(raw, cfg, phases)
    return result, phases


def _context(mantra: str | None) -> ReportContext:
    return ReportContext(
        participant_name="Teste",
        session_started=datetime(2026, 9, 10, 11, 12),
        duration_s=200.0,
        mantra_label=mantra,
        session_code="FBE-2026-000000",
        festival_label="Festival",
    )


def test_silent_session_never_names_a_mantra(cfg, session):
    """Uma sessão de controlo não pode sair com o nome de um mantra."""
    result, phases = session
    data = build(result, cfg, _context(None), [], phases)

    assert data["session"]["silent"] is True
    assert data["session"]["mantra"] == "Nenhum — sessão em silêncio"

    page = render(data)
    for entry in cfg.paths.mantras:
        assert entry.label not in page, f"o relatório nomeia {entry.label}"


def test_stable_markers_are_never_painted_as_effects(cfg, session):
    """Sem efeito real, nenhuma pastilha fica verde e nada diz que mudou."""
    result, phases = session
    data = build(result, cfg, _context("Sri Vitthala"), [], phases)

    for metric in data["metrics"]:
        if metric["reliable"]:
            continue
        assert metric["deltaColor"] == "#6E8B93", metric["name"]
        # Sem efeito medido, o cartão descreve a medida e cala-se. A ressalva
        # de "manteve-se dentro do normal" vive uma vez na leitura da sessão,
        # e não repetida em cada cartão.
        text = metric["interpretation"]
        for claim in ("Nesta sessão subiu", "Nesta sessão desceu", "maior do que"):
            assert claim not in text, f"{metric['name']}: {text}"


def test_every_marker_carries_participant_copy():
    """O registo é fonte única: um marcador novo traz o texto com ele."""
    missing = [m.id for m in REGISTRY if not m.explanation.strip()]
    assert missing == [], f"sem explicação para o participante: {missing}"


def test_page_is_self_contained(cfg, session):
    """Nada de ficheiros externos: o relatório viaja por email e imprime."""
    result, phases = session
    data = build(result, cfg, _context(None), [], phases)
    page = render(data)

    assert "fonts.googleapis.com" in page
    # Qualquer outro src/href tem de ser data: — um caminho relativo saía
    # partido assim que o ficheiro mudasse de pasta.
    import re

    for url in re.findall(r'(?:src|href)="([^"]+)"', page):
        assert url.startswith(("data:", "https://fonts.googleapis.com")), url
