"""Validação de montagem: o risco de F3/F4 trocados.

Se o mapeamento índice→elétrodo estiver invertido, o ALAY sai com o sinal ao
contrário e a coerência fica errada, sem que nada no gráfico o denuncie. Estas
verificações são a única coisa entre isso e um relatório errado.
"""

from __future__ import annotations

import numpy as np
import pytest

from mantraeeg.montage import (
    auto_checks,
    check_channel_independence,
    check_frontopolar_dominance,
    check_lateral_gaze,
    identify_tapped_channel,
    summarize,
)
from mantraeeg.sources.base import EEG_SLICE


def blinky(cfg, fp_uv: float, frontal_uv: float, anti_phase: bool = False, seed: int = 0):
    """8 canais com pestanejos, mais fortes em Fp1/Fp2.

    Os pestanejos são transientes gaussianos de ~150 ms, não sinusoides lentas:
    a energia tem de cair dentro da banda de deteção (1–5 Hz), senão o filtro
    elimina-a e a verificação não tem nada para medir.
    """
    sfreq = cfg.device.sfreq_nominal
    n = int(cfg.montage.validation.auto_check_window_s * sfreq)
    t = np.arange(n) / sfreq
    rng = np.random.default_rng(seed)

    sigma_s = 0.06
    blink = np.zeros(n)
    for onset in np.arange(1.0, t[-1] - 1.0, 3.0):  # ~20 pestanejos por minuto
        blink += np.exp(-0.5 * ((t - onset) / sigma_s) ** 2)
    blink /= blink.max() or 1.0

    eeg = rng.normal(0, 3.0, (8, n))
    m = cfg.montage
    for name in m.eog_channels:
        sign = -1.0 if (anti_phase and name == "Fp2") else 1.0
        eeg[m.index_of(name)] += sign * fp_uv * blink
    for name in ("F3", "F4"):
        eeg[m.index_of(name)] += frontal_uv * blink
    return eeg


# --------------------------------------------------------------------------- #
# Independência entre canais
# --------------------------------------------------------------------------- #
def test_independent_channels_pass():
    rng = np.random.default_rng(0)
    result = check_channel_independence(rng.normal(0, 10, (8, 2500)), 0.98, 0.02)
    assert result.passed is True


def test_common_mode_only_is_caught():
    """Referência/terra soltos: a assinatura medida no headset real."""
    rng = np.random.default_rng(0)
    common = np.cumsum(rng.normal(0, 40, 2500))
    eeg = np.tile(common, (8, 1)) + rng.normal(0, 0.3, (8, 2500))

    result = check_channel_independence(eeg, 0.98, 0.02)
    assert result.passed is False
    assert "referência" in result.detail
    assert result.values["min_interchannel_r"] > 0.98
    assert result.values["max_pair_diff_ratio"] < 0.02


@pytest.mark.parametrize("amplitude_uv", [1_490.0, 10_500.0, 50_000.0])
def test_no_contact_is_caught_at_any_amplitude(amplitude_uv):
    """Regressão: o limiar tem de ser relativo, não em µV absolutos.

    Com um limiar absoluto de 5 µV, este caso passava despercebido a 10 500 µV
    de amplitude — a diferença residual entre canais escala com o sinal comum,
    por isso um limiar fixo falha precisamente nas amplitudes maiores, que são
    as piores. Medido em hardware real com os clips de referência soltos.
    """
    rng = np.random.default_rng(0)
    n = 2500
    common = np.cumsum(rng.normal(0, 1, n))
    common = common / common.std() * amplitude_uv
    eeg = np.tile(common, (8, 1)) + rng.normal(0, amplitude_uv * 0.0005, (8, n))

    result = check_channel_independence(eeg, 0.98, 0.02)
    assert result.passed is False, (
        f"referência solta não detetada a {amplitude_uv:.0f} µV: "
        f"{result.values}"
    )


def test_independence_needs_enough_data():
    assert check_channel_independence(np.zeros((8, 1)), 0.98, 0.02).passed is None
    assert check_channel_independence(np.zeros((1, 500)), 0.98, 0.02).passed is None


# --------------------------------------------------------------------------- #
# Dominância frontopolar: valida o eixo ântero-posterior
# --------------------------------------------------------------------------- #
def test_frontopolar_dominance_passes_when_fp_leads(cfg):
    eeg = blinky(cfg, fp_uv=80.0, frontal_uv=20.0)
    result = check_frontopolar_dominance(eeg, cfg)
    assert result.passed is True
    assert result.values["ratio"] > cfg.montage.validation.frontopolar_dominance_min_ratio


def test_frontopolar_dominance_fails_when_indices_are_swapped(cfg):
    """Se 4/5 não forem o par frontopolar, o pestanejo aparece no sítio errado."""
    eeg = blinky(cfg, fp_uv=20.0, frontal_uv=80.0)
    result = check_frontopolar_dominance(eeg, cfg)
    assert result.passed is False
    assert "frontopolar" in result.detail


# --------------------------------------------------------------------------- #
# Olhar lateral: a única verificação automática que resolve esquerda/direita
# --------------------------------------------------------------------------- #
def test_lateral_gaze_detected_when_fp_channels_are_antiphase(cfg):
    result = check_lateral_gaze(blinky(cfg, 80.0, 10.0, anti_phase=True), cfg)
    assert result.passed is True
    assert result.values["r_fp1_fp2"] < 0


def test_lateral_gaze_inconclusive_with_only_vertical_blinks(cfg):
    """Pestanejo vertical põe Fp1/Fp2 em fase: a lateralidade fica por confirmar."""
    result = check_lateral_gaze(blinky(cfg, 80.0, 10.0, anti_phase=False), cfg)
    assert result.passed is None
    assert "lateralidade" in result.detail


# --------------------------------------------------------------------------- #
# Teste de toque: a verificação definitiva
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tapped", [0, 1, 3, 7])
def test_tap_test_identifies_the_right_index(cfg, synthetic_eeg, tapped):
    eeg = synthetic_eeg(seconds=5.0, noise_uv=4.0, dc_offset_uv=0.0)
    rng = np.random.default_rng(2)
    eeg[tapped] += rng.normal(0, 120.0, eeg.shape[1])

    found, ratio = identify_tapped_channel(eeg, cfg)
    assert found == tapped
    assert ratio >= cfg.montage.validation.tap_test_min_ratio


def test_tap_test_reports_nothing_when_no_channel_stands_out(cfg, synthetic_eeg):
    found, ratio = identify_tapped_channel(
        synthetic_eeg(seconds=5.0, dc_offset_uv=0.0), cfg
    )
    assert found is None
    assert ratio < cfg.montage.validation.tap_test_min_ratio


def test_tap_test_would_catch_swapped_f3_f4(cfg, synthetic_eeg):
    """O cenário concreto que preocupa: tocar em F3 e reagir o índice do F4."""
    eeg = synthetic_eeg(seconds=5.0, noise_uv=4.0, dc_offset_uv=0.0)
    rng = np.random.default_rng(3)
    f4_index = cfg.montage.index_of("F4")
    eeg[f4_index] += rng.normal(0, 120.0, eeg.shape[1])

    found, _ = identify_tapped_channel(eeg, cfg)
    assert found == f4_index
    assert found != cfg.montage.index_of("F3")


# --------------------------------------------------------------------------- #
# Conjunto automático
# --------------------------------------------------------------------------- #
def test_auto_checks_short_circuit_when_channels_are_not_independent(cfg):
    rng = np.random.default_rng(0)
    n = int(cfg.montage.validation.auto_check_window_s * cfg.device.sfreq_nominal)
    common = np.cumsum(rng.normal(0, 40, n))
    eeg = np.tile(common, (8, 1)) + rng.normal(0, 0.3, (8, n))

    results = auto_checks(eeg, cfg)
    assert results[0].passed is False
    assert all(r.passed is None for r in results[1:]), (
        "sem canais independentes, as outras verificações não significam nada"
    )


def test_auto_checks_on_plausible_data(cfg):
    results = auto_checks(blinky(cfg, 80.0, 20.0, anti_phase=True), cfg)
    assert [r.name for r in results] == [
        "independencia_canais",
        "dominancia_frontopolar",
        "olhar_lateral",
    ]
    assert summarize(results)["any_failed"] is False


def test_auto_checks_on_the_real_recording(cfg, real_recording):
    """Corre sobre dados genuínos sem rebentar e produz algo serializável."""
    import json

    results = auto_checks(real_recording[EEG_SLICE], cfg)
    payload = summarize(results)
    json.dumps(payload)  # tem de caber no raw_meta.json
    assert len(payload["checks"]) == 3
