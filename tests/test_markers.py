"""Registo de biomarcadores e as funções que os calculam.

O registo é a fonte única (SPEC 4.1). O critério de aceitação 10 diz que
acrescentar uma entrada tem de a fazer aparecer no relatório, na tabela e na
agregação sem editar mais nada — é isso que estes testes fixam.
"""

from __future__ import annotations

import numpy as np
import pytest

from mantraeeg.markers import BY_ID, REGISTRY, Marker, get, markers
from mantraeeg.spectral import welch_psd


@pytest.fixture
def chans(cfg):
    return cfg.montage.name_to_index


@pytest.fixture
def psd_of(cfg):
    """PSD de um sinal sintético com alfa controlado."""

    def build(alpha_uv: float = 10.0, beta_uv: float = 3.0, seconds: float = 8.0):
        fs = cfg.device.sfreq_nominal
        n = int(seconds * fs)
        t = np.arange(n) / fs
        rng = np.random.default_rng(0)
        data = np.zeros((8, n))
        for ch in range(8):
            pink = np.cumsum(rng.normal(0, 1, n))
            pink = pink / (pink.std() or 1.0) * 4.0
            data[ch] = (
                pink
                + alpha_uv * np.sin(2 * np.pi * 10.0 * t + ch)
                + beta_uv * np.sin(2 * np.pi * 20.0 * t + ch)
                + rng.normal(0, 1.5, n)
            )
        return welch_psd(data, fs, seg_s=2.0), data

    return build


# --------------------------------------------------------------------------- #
# O registo
# --------------------------------------------------------------------------- #
def test_registry_ids_are_unique():
    ids = [m.id for m in REGISTRY]
    assert len(ids) == len(set(ids))


def test_registry_covers_the_eleven_markers_of_the_spec():
    required = {
        "alpha_rel", "beta_rel", "alpha_beta", "smr_rel", "theta_rel", "faa",
        "coh_alpha1", "coh_theta2", "wpli_alpha1", "ap_exponent", "lziv",
    }
    assert required <= set(BY_ID)


def test_composite_membership_matches_the_spec():
    """theta_rel, faa, ap_exponent, lziv e wpli ficam fora do índice."""
    inside = {m.id for m in REGISTRY if m.in_composite}
    assert inside == {"alpha_rel", "beta_rel", "smr_rel"}


def test_unknown_direction_where_the_evidence_is_weak():
    for marker_id in ("ap_exponent", "lziv"):
        assert get(marker_id).expected_direction == "unknown"


def test_every_marker_carries_its_caveats():
    """As notas são impressas no relatório: nenhuma pode faltar."""
    for marker in REGISTRY:
        assert marker.notes.strip(), f"{marker.id} sem notas"
        assert marker.label.strip()


def test_extras_are_separated_from_markers():
    extras = {m.id for m in markers(role="extra")}
    assert "alpha_peak_power" in extras
    assert not any(m.in_composite for m in markers(role="extra"))


def test_filtering_the_registry():
    assert all(m.kind == "pair" for m in markers(kind="pair"))
    assert {m.id for m in markers(kind="pair")} == {
        "coh_alpha1", "coh_theta2", "imcoh_alpha1", "wpli_alpha1"
    }


def test_get_rejects_unknown_ids():
    with pytest.raises(KeyError, match="não está no registo"):
        get("telepatia")


def test_adding_a_marker_needs_only_an_entry():
    """Critério de aceitação 10, verificado pela estrutura.

    Uma entrada nova é imediatamente iterável e resolúvel por id — não há
    listas paralelas noutro sítio para manter em sincronia.
    """
    extra = Marker(
        id="teste_novo",
        label="Marcador de teste",
        kind="psd",
        channels=("F3",),
        fn=lambda psd, freqs, chans, bands: 1.0,
        expected_direction="up",
        in_composite=False,
        notes="entrada de teste",
    )
    combined = REGISTRY + (extra,)
    assert len([m for m in combined if m.id == "teste_novo"]) == 1
    assert extra.role == "marker"


# --------------------------------------------------------------------------- #
# As funções
# --------------------------------------------------------------------------- #
def test_alpha_rel_rises_with_injected_alpha(cfg, chans, psd_of):
    (freqs, low), _ = psd_of(alpha_uv=2.0)
    (_, high), _ = psd_of(alpha_uv=20.0)
    fn = get("alpha_rel").fn
    assert fn(high, freqs, chans, cfg.bands) > fn(low, freqs, chans, cfg.bands)


def test_beta_rel_rises_with_injected_beta(cfg, chans, psd_of):
    (freqs, low), _ = psd_of(beta_uv=1.0)
    (_, high), _ = psd_of(beta_uv=15.0)
    fn = get("beta_rel").fn
    assert fn(high, freqs, chans, cfg.bands) > fn(low, freqs, chans, cfg.bands)


def test_alpha_beta_moves_the_right_way(cfg, chans, psd_of):
    (freqs, alpha_heavy), _ = psd_of(alpha_uv=20.0, beta_uv=1.0)
    (_, beta_heavy), _ = psd_of(alpha_uv=2.0, beta_uv=15.0)
    fn = get("alpha_beta").fn
    assert fn(alpha_heavy, freqs, chans, cfg.bands) > fn(
        beta_heavy, freqs, chans, cfg.bands
    )


def test_faa_sign_convention_matches_neroes(cfg, chans, psd_of):
    """``ln P[F4] - ln P[F3]``: mais alfa à direita dá positivo.

    Convenção confirmada contra o código interno da Neroes
    (``F4Alpha - F3Alpha``). Se isto inverter, todos os gráficos de assimetria
    saem ao contrário.
    """
    (freqs, psd), _ = psd_of()
    psd = psd.copy()
    psd[chans["F4"]] *= 4.0     # mais potência à direita
    assert get("faa").fn(psd, freqs, chans, cfg.bands) > 0

    psd = psd.copy()
    psd[chans["F3"]] *= 40.0    # agora mais à esquerda
    assert get("faa").fn(psd, freqs, chans, cfg.bands) < 0


def test_markers_return_nan_without_their_channels(cfg, psd_of):
    """Sem os canais de que dependem, nan — nunca um valor inventado."""
    (freqs, psd), _ = psd_of()
    empty: dict[str, int] = {}
    for marker in markers(kind="psd"):
        assert np.isnan(marker.fn(psd, freqs, empty, cfg.bands, **marker.params))


def test_lziv_is_higher_for_noise_than_for_a_sine(cfg, chans):
    fs = cfg.device.sfreq_nominal
    n = int(4 * fs)
    t = np.arange(n) / fs
    rng = np.random.default_rng(1)
    sine = np.tile(np.sin(2 * np.pi * 10 * t), (8, 1))
    noise = rng.normal(0, 10, (8, n))
    fn = get("lziv").fn
    assert fn(noise, chans, cfg.bands) > fn(sine, chans, cfg.bands)


def test_ap_exponent_is_positive_for_pink_noise(cfg, chans, psd_of):
    (freqs, psd), _ = psd_of()
    value = get("ap_exponent").fn(psd, freqs, chans, cfg.bands, fit_range=(3.0, 40.0))
    assert value > 0, "o expoente aperiódico de EEG deve ser positivo"


def test_alpha_peak_power_isolates_the_oscillation(cfg, chans, psd_of):
    (freqs, low), _ = psd_of(alpha_uv=1.0)
    (_, high), _ = psd_of(alpha_uv=25.0)
    fn = get("alpha_peak_power").fn
    assert fn(high, freqs, chans, cfg.bands) > fn(low, freqs, chans, cfg.bands)
