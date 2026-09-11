"""A config valida na carga, não em uso."""

from __future__ import annotations

import dataclasses

import pytest

from mantraeeg.bands import Bands
from mantraeeg.config import ConfigError, n_epochs


def test_default_config_loads(cfg):
    assert cfg.version >= 1
    assert cfg.device.sfreq_nominal == 250.0
    assert cfg.montage.n_eeg == 8


def test_dataclasses_are_frozen(cfg):
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.protocol.mantra_s = 1.0  # type: ignore[misc]


def test_montage_is_the_custom_one_not_the_unicorn_default(cfg):
    """A montagem tem de ser a nossa. O default do Unicorn começa em Fz."""
    assert cfg.montage.channels[0] == "F3"
    assert cfg.montage.channels[1] == "F4"
    assert cfg.montage.index_of("F3") != cfg.montage.index_of("F4")
    assert set(cfg.montage.metric_channels) == {"F3", "F4", "C3", "C4"}
    assert "Fz" not in cfg.montage.name_to_index


def test_car_must_stay_off(cfg_factory):
    """SPEC 1.2 proíbe CAR: distorceria a assimetria e a coerência F3-F4."""
    with pytest.raises(ConfigError, match="apply_car"):
        cfg_factory({"montage": {"apply_car": True}})


def test_duplicate_electrode_names_rejected(cfg_factory):
    with pytest.raises(ConfigError, match="repetidas"):
        cfg_factory({"montage": {"channels": {0: "F3", 1: "F3"}}})


def test_metric_channel_not_in_montage_rejected(cfg_factory):
    with pytest.raises(ConfigError, match="metric_channels"):
        cfg_factory({"montage": {"metric_channels": ["F3", "T7"]}})


def test_calib_use_last_cannot_exceed_calibration(cfg_factory):
    with pytest.raises(ConfigError, match="calib_use_last_s"):
        cfg_factory({"protocol": {"calibration_s": 30.0, "calib_use_last_s": 60.0}})


def test_beta_must_not_overlap_smr(cfg_factory):
    """BIOMARKERS.md: beta começa em 15 Hz por causa do SMR (12-15)."""
    with pytest.raises(ValueError, match="sobrepõe-se ao SMR"):
        cfg_factory({"bands": {"beta": [13.0, 30.0]}})


def test_alpha_subbands_must_fit_inside_alpha(cfg_factory):
    with pytest.raises(ValueError, match="contida em alpha"):
        cfg_factory({"bands": {"alpha1": [6.0, 10.0]}})


def test_inverted_band_rejected():
    with pytest.raises(ValueError, match="tem de ser <"):
        Bands.from_config({"alpha": [12.0, 8.0]})


def test_connectivity_window_must_yield_enough_segments(cfg_factory):
    """B10: uma janela de 4 s com sub-segmentos de 2 s dá 3 segmentos — insuficiente."""
    with pytest.raises(ConfigError, match="segmentos"):
        cfg_factory(
            {
                "epochs": {
                    "connectivity": {"win_s": 4.0, "hop_s": 2.0, "welch_seg_s": 2.0}
                }
            }
        )


def test_redundant_notch_rejected_but_50hz_kept(cfg_factory):
    """O notch de 50 Hz é preciso; o de 100 Hz o passa-banda já o mata."""
    cfg_factory({"preprocess": {"notch_freqs_hz": [50.0]}})  # não levanta
    with pytest.raises(ConfigError, match="redundante"):
        cfg_factory({"preprocess": {"notch_freqs_hz": [50.0, 100.0]}})


def test_contact_monitor_detrend_cannot_be_disabled(cfg_factory):
    """Sem detrend, os offsets DC do Unicorn põem tudo a vermelho."""
    with pytest.raises(ConfigError, match="detrend"):
        cfg_factory({"contact_monitor": {"detrend_window": False}})


def test_participant_profile_requires_featured_markers(cfg_factory):
    """Os três marcadores só se escolhem depois da agregação (SPEC 5.3)."""
    with pytest.raises(ConfigError, match="featured_markers"):
        cfg_factory({"report": {"profile": "participant", "featured_markers": []}})
    cfg_factory(
        {"report": {"profile": "participant", "featured_markers": ["alpha_rel"]}}
    )


def test_unknown_source_rejected(cfg_factory):
    with pytest.raises(ConfigError, match="device.source"):
        cfg_factory({"device": {"source": "telepatia"}})


# --------------------------------------------------------------------------- #
# Aritmética de grelha e adequação da referência
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "duration, win, hop, expected",
    [
        (30.0, 4.0, 2.0, 14),
        (90.0, 4.0, 2.0, 44),
        (30.0, 10.0, 5.0, 5),
        (90.0, 10.0, 5.0, 17),
        (3.0, 4.0, 2.0, 0),
        (4.0, 4.0, 2.0, 1),
    ],
)
def test_n_epochs(duration, win, hop, expected):
    assert n_epochs(duration, win, hop) == expected


def test_reference_adequacy_flags_short_calibration(cfg):
    """60 s de calibração dão 5 janelas de coerência: referência insuficiente."""
    short = cfg.reference_adequacy(60.0, 30.0)
    assert short.n_power_epochs == 14
    assert short.n_conn_windows == 5
    assert short.conn_quality == "insufficient"

    longer = cfg.reference_adequacy(120.0, 90.0)
    assert longer.n_power_epochs == 44
    assert longer.n_conn_windows == 17
    assert longer.power_quality == "ok"
    assert longer.conn_quality == "ok"


def test_reference_adequacy_is_monotone_in_duration(cfg):
    order = {"insufficient": 0, "thin": 1, "ok": 2}
    previous = -1
    for total in (40.0, 60.0, 90.0, 120.0, 180.0, 300.0):
        adequacy = cfg.reference_adequacy(total, total)
        assert order[adequacy.conn_quality] >= previous
        previous = order[adequacy.conn_quality]


def test_reference_adequacy_uses_only_the_tail(cfg):
    """Só os últimos calib_use_last_s contam — o resto é a explicação em voz alta."""
    assert cfg.reference_adequacy(300.0, 30.0).n_power_epochs == 14


# --------------------------------------------------------------------------- #
# Invariante ocular (SPEC 2) — verificada, não suposta
# --------------------------------------------------------------------------- #
def test_eye_condition_is_closed_for_the_mantra_protocol(cfg):
    assert cfg.protocol.eye_calibration == "closed"
    assert cfg.protocol.eye_mantra == "closed"
    assert cfg.protocol.eye_condition == "closed"


def test_mismatched_eye_conditions_are_rejected(cfg_factory):
    """Calibração de olhos abertos com meditação de olhos fechados é inválida.

    O efeito de Berger multiplica o alfa por 2 a 5x só por fechar as pálpebras —
    uma ordem de grandeza acima de qualquer efeito de meditação. Todos os
    marcadores de potência mostrariam uma melhoria que é inteiramente pálpebras.
    """
    with pytest.raises(ConfigError, match="Berger"):
        cfg_factory({"protocol": {"eye_calibration": "open", "eye_mantra": "closed"}})
    with pytest.raises(ConfigError, match="invariante ocular"):
        cfg_factory({"protocol": {"eye_calibration": "closed", "eye_mantra": "open"}})


def test_matching_eye_conditions_are_accepted(cfg_factory):
    for eye in ("open", "closed"):
        built = cfg_factory(
            {"protocol": {"eye_calibration": eye, "eye_mantra": eye, "eye_settle": eye}}
        )
        assert built.protocol.eye_condition == eye


def test_composite_weights_follow_the_eye_condition(cfg_factory):
    """De olhos fechados o alfa deixa de ser fraco e o peso desloca-se para ele."""
    closed = cfg_factory({})
    weights = closed.composite.weights_for(closed.protocol.eye_condition)
    assert weights["alpha_rel"] == 0.50
    assert weights["beta_rel"] == -0.30

    opened = cfg_factory(
        {"protocol": {"eye_calibration": "open", "eye_mantra": "open", "eye_settle": "open"}}
    )
    open_weights = opened.composite.weights_for(opened.protocol.eye_condition)
    assert open_weights["alpha_rel"] == 0.35
    assert open_weights["beta_rel"] == -0.40


def test_visual_cues_rejected_when_eyes_are_closed(cfg_factory):
    """De olhos fechados a pessoa não vê a deixa; tem de ser áudio."""
    with pytest.raises(ConfigError, match="olhos fechados"):
        cfg_factory({"protocol": {"cues": {"mode": "visual"}}})
    cfg_factory({"protocol": {"cues": {"mode": "both"}}})


def test_eye_for_unknown_phase_raises(cfg):
    with pytest.raises(KeyError, match="fase desconhecida"):
        cfg.protocol.eye_for("intervalo")


def test_ui_shortcuts_are_configured(cfg):
    """Os atalhos vêm da config, não hardcoded na UI."""
    assert cfg.ui.shortcuts["inspect_signal"] == "Ctrl+I"
    assert cfg.ui.shortcuts["toggle_video"] == "V"
    assert cfg.ui.inspector_trace_s > 0


def test_settle_becomes_baseline_only_when_long_enough(cfg_factory):
    assert not cfg_factory({"protocol": {"settle_s": 15.0}}).protocol.settle_is_baseline
    assert cfg_factory({"protocol": {"settle_s": 60.0}}).protocol.settle_is_baseline
