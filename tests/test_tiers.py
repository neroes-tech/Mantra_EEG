"""O relatório muda de forma conforme o sinal que houve.

Num festival, com um palco ao lado, sessões com sinal mau não são exceção —
são o caso normal a planear. O que não pode acontecer é o relatório continuar
a falar de variações quando não há sinal para as sustentar.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest

from mantraeeg.analysis import analyse
from mantraeeg.report.html import render
from mantraeeg.report.payload import ReportContext, build

CONTEXT = ReportContext(
    participant_name="Teste",
    session_started=datetime(2026, 9, 11, 12, 0),
    duration_s=200.0,
    mantra_label="Sri Vitthala",
    session_code="FBE-2026-000000",
    festival_label="Festival",
)


@pytest.fixture()
def result(cfg):
    rng = np.random.default_rng(7)
    n = int(200 * cfg.device.sfreq_nominal)
    raw = rng.normal(0.0, 12.0, size=(8, n))
    return analyse(raw, cfg, [("CALIBRATION", 0.0, 100.0), ("MANTRA", 100.0, 200.0)])


def _with_coverage(result, coverage: float):
    """O mesmo resultado, com a cobertura forçada, para testar os limiares."""
    return replace(
        result,
        summaries=[replace(s, coverage=coverage) for s in result.summaries],
    )


@pytest.mark.parametrize(
    "coverage,expected",
    [(0.95, "full"), (0.11, "full"), (0.09, "descriptive"),
     (0.03, "descriptive"), (0.01, "insufficient"), (0.0, "insufficient")],
)
def test_tier_follows_coverage(cfg, result, coverage, expected):
    data = build(_with_coverage(result, coverage), cfg, CONTEXT, [])
    assert data["tier"] == expected


def test_low_signal_never_claims_a_change(cfg, result):
    """Sem sinal, nenhuma palavra de variação — nem no destaque nem na prosa."""
    for coverage in (0.05, 0.005):
        data = build(_with_coverage(result, coverage), cfg, CONTEXT, [])
        assert data["headline"] == []
        page = render(data)
        for claim in ("Em destaque", "vs repouso", "Biomarcador a biomarcador"):
            assert claim not in page, f"cobertura {coverage}: {claim}"
        prosa = " ".join(data["reading"])
        assert "mudaram mais do que" not in prosa
        assert "biomarcadores mudou" not in prosa


def test_insufficient_shows_no_numbers_but_still_says_something(cfg, result):
    data = build(_with_coverage(result, 0.005), cfg, CONTEXT, [])
    page = render(data)
    assert "O que conseguimos gravar" in page
    assert "repetimos a sessão" in " ".join(data["reading"])
    # O espectro sai: com o sensor sem contacto, nem ele e da pessoa.
    assert not data["spectrum"]


def test_descriptive_keeps_the_spectrum(cfg, result):
    data = build(_with_coverage(result, 0.05), cfg, CONTEXT, [])
    assert data["spectrum"].get("freqs"), "o espectro é o que sobra para mostrar"
    assert "O teu espectro" in render(data)
