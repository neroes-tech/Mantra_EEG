"""O ecrã final tem sempre o número de destaques pedido.

Numa sessão real o destaque apareceu com **um** cartão em vez de três: os
filtros de qualidade excluíam marcadores, e a escolha era feita só de entre os
que estavam marcados no painel — que por defeito eram três. Estes testes
fecham as duas causas.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mantraeeg.analysis import analyse, baseline_spread, rank_by_variation
from mantraeeg.markers import markers


@pytest.fixture()
def result(cfg):
    rng = np.random.default_rng(11)
    sfreq = cfg.device.sfreq_nominal
    n = int(240 * sfreq)
    # Sinal com um pouco de estrutura, para as medianas por fase diferirem.
    t = np.arange(n) / sfreq
    base = rng.normal(0.0, 14.0, size=(8, n))
    base[:4] += 8.0 * np.sin(2 * np.pi * 10.0 * t)
    return analyse(base, cfg, [("CALIBRATION", 0.0, 120.0), ("MANTRA", 120.0, 240.0)])


def test_returns_exactly_the_limit_asked(cfg, result):
    for limit in (1, 3, 5):
        chosen = rank_by_variation(result, cfg, limit=limit)
        assert len(chosen) == limit, f"limite {limit} devolveu {chosen}"


def test_never_picks_an_extra(cfg, result):
    roles = {m.id: m.role for m in markers()}
    for marker_id in rank_by_variation(result, cfg, limit=5):
        assert roles[marker_id] == "marker", marker_id


def test_flagged_markers_are_demoted_not_dropped(cfg, result):
    """Um marcador arrastado pela mandíbula desce, mas ainda aparece.

    Excluí-los era o que deixava o ecrã com um cartão. A ressalva passa a
    viajar no texto do cartão, não na ausência dele.
    """
    dirty = replace(result.summaries[0], r_emg=0.99)
    clean_ids = [s.marker.id for s in result.summaries[1:] if s.marker.role == "marker"]
    polluted = replace(result, summaries=[dirty, *result.summaries[1:]])

    ranked = rank_by_variation(polluted, cfg, limit=len(clean_ids) + 1)
    assert dirty.marker.id in ranked
    assert ranked[-1] == dirty.marker.id, "o marcador sinalizado devia ficar em último"


def test_ranking_follows_absolute_change(cfg, result):
    """A ordem é a da magnitude da variação, e não a da fiabilidade."""
    ranked = rank_by_variation(result, cfg, limit=4)
    scores = []
    for marker_id in ranked:
        summary = result.summary_of(marker_id)
        spread = baseline_spread(result, marker_id)
        change = summary.active - summary.baseline
        if np.isfinite(spread) and abs(summary.baseline) > spread:
            scores.append(abs(change / summary.baseline) * 100.0)
    assert scores == sorted(scores, reverse=True), scores
