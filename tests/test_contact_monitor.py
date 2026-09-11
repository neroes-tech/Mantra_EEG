"""Monitor de contacto, testado contra uma gravação Unicorn real.

Os canais do Unicorn trazem offsets DC de eletrodo de 185 a 711 mV e, quando o
contacto é mau, uma deriva que chega a 5 660 µV/s. O offset por si só não afeta o
desvio padrão (que é centrado na média), mas a deriva **dentro da janela** afeta,
e afeta o pico-a-pico muito mais: sem detrend, os oito canais ficariam vermelhos
e o operador nunca sairia do estado ``CONTACT``.
"""

from __future__ import annotations

import numpy as np
import pytest

from mantraeeg.acquisition import ContactMonitor
from mantraeeg.sources.base import EEG_SLICE


@pytest.fixture
def monitor(cfg) -> ContactMonitor:
    return ContactMonitor(cfg)


def window_from(recording: np.ndarray, cfg, seconds: float | None = None) -> np.ndarray:
    n = int((seconds or cfg.contact_monitor.window_s) * cfg.device.sfreq_nominal)
    return recording[:, -n:]


# --------------------------------------------------------------------------- #
# O achado dos offsets DC
# --------------------------------------------------------------------------- #
def test_real_recording_has_huge_dc_offsets(real_recording):
    """Documenta o porquê do detrend: a premissa do teste seguinte."""
    means = np.abs(real_recording[EEG_SLICE].mean(axis=1))
    assert means.min() > 100_000, "esperado offset de eletrodo de centenas de mV"


def test_real_recording_is_usable_after_filtering(monitor, real_recording, cfg):
    """A gravação de 2022 é utilizável — o problema dela era ambiente, não contacto.

    Avaliada sobre o sinal filtrado, dá sd de 5 a 121 uV e declives log-log de
    -0,6 a -2,7, todos com decaimento 1/f plausível. Julgada sobre o sinal bruto
    ficavam os oito canais vermelhos, porque o que dominava era a rede de 50 Hz
    que o pipeline remove logo a seguir.
    """
    report = monitor.evaluate(window_from(real_recording, cfg))
    assert report.channels, "o monitor nao avaliou nenhum canal"
    # Quatro dos oito: metade da touca. Nao e uma condicao de arranque — a
    # aplicacao nao bloqueia nada — e so a afirmacao de que esta gravacao
    # tinha contacto a serio, ao contrario das que vieram da banca.
    assert report.n_good >= 4, (
        f"canais utilizaveis a menos: "
        f"{[(c.name, c.level, c.reason) for c in report.channels]}"
    )
    assert all(
        c.slope < 0 for c in report.channels if np.isfinite(c.slope)
    ), "EEG real tem de decair com a frequencia"


def test_real_recording_raises_a_line_environment_warning(monitor, real_recording, cfg):
    """A rede e sinalizada ao nivel do relatorio, nao a reprovar canais."""
    report = monitor.evaluate(window_from(real_recording, cfg))
    assert report.line_warning is not None
    assert "50 Hz" in report.line_warning
    assert max(c.line_rel for c in report.channels) > cfg.contact_monitor.line_warn_ratio


def test_real_recording_has_one_near_dead_channel(monitor, real_recording, cfg):
    """O índice 3 tem sd ~23 µV contra 100-600 dos outros: elétrodo quase solto."""
    report = monitor.evaluate(window_from(real_recording, cfg))
    stds = {c.index: c.std_uv for c in report.channels}
    others = [v for i, v in stds.items() if i != 3]
    assert stds[3] < min(others), "o índice 3 devia ser o mais silencioso"


def test_real_recording_does_not_trigger_a_false_no_contact_warning(
    monitor, real_recording, cfg
):
    """Os canais são maus mas independentes: não é o caso da referência solta."""
    report = monitor.evaluate(window_from(real_recording, cfg))
    assert report.global_warning is None


def test_detrend_removes_within_window_drift_not_the_offset(cfg, real_recording):
    """Precisão sobre o que o detrend faz.

    O desvio padrão já é centrado na média, portanto o offset de eletrodo (185 a
    711 mV) não o afeta. O que o detrend remove é a **deriva dentro da janela**.
    No ficheiro de 2022 essa deriva em 2 s é modesta (razão ~1,03); numa aquisição
    ao vivo com os elétrodos soltos mediu-se o DC a subir ~5 660 µV/s, e aí a
    deriva domina por completo.
    """
    from scipy import signal

    n = int(cfg.contact_monitor.window_s * cfg.device.sfreq_nominal)
    raw = real_recording[EEG_SLICE][:, -n:].astype(np.float64)
    detrended = signal.detrend(raw, axis=-1, type="linear")
    assert (detrended.std(axis=1) <= raw.std(axis=1) + 1e-9).all()


def test_moderate_drift_is_rescued_by_detrend(cfg, synthetic_eeg):
    """Deriva moderada nao pode reprovar um canal: o detrend trata dela."""
    from scipy import signal

    sfreq = cfg.device.sfreq_nominal
    window_s = cfg.contact_monitor.window_s
    eeg = synthetic_eeg(seconds=window_s, noise_uv=4.0, dc_offset_uv=0.0)
    t = np.arange(eeg.shape[1]) / sfreq
    drift = 0.5 * cfg.contact_monitor.max_drift_uv_per_s
    with_drift = eeg + drift * t

    limit = cfg.contact_monitor.yellow.max_std_uv
    assert (with_drift.std(axis=1) > limit).all(), "a rampa domina o sd bruto"
    detrended = signal.detrend(with_drift, axis=-1, type="linear")
    assert (detrended.std(axis=1) < limit).all()

    window = np.zeros((17, with_drift.shape[1]), dtype=np.float32)
    window[EEG_SLICE] = with_drift
    assert ContactMonitor(cfg).evaluate(window).n_good == 8


def test_extreme_drift_is_red(cfg, synthetic_eeg):
    """Ao vivo, com os eletrodos soltos, mediram-se 5 660 uV/s. Isso e vermelho."""
    sfreq = cfg.device.sfreq_nominal
    eeg = synthetic_eeg(
        seconds=cfg.contact_monitor.window_s, noise_uv=4.0, dc_offset_uv=0.0
    )
    t = np.arange(eeg.shape[1]) / sfreq
    eeg = eeg + 5_660.0 * t

    window = np.zeros((17, eeg.shape[1]), dtype=np.float32)
    window[EEG_SLICE] = eeg
    report = ContactMonitor(cfg).evaluate(window)
    assert all(c.level == "red" for c in report.channels)
    assert all("flutuar" in c.reason for c in report.channels)


def test_dead_electrode_is_flagged(monitor, cfg, synthetic_eeg):
    eeg = synthetic_eeg(seconds=cfg.contact_monitor.window_s)
    eeg[3] = 200_000.0  # completamente plano
    window = np.zeros((17, eeg.shape[1]), dtype=np.float32)
    window[EEG_SLICE] = eeg

    report = monitor.evaluate(window)
    dead = report.by_name(cfg.montage.channels[3])
    assert dead.level == "red"
    assert "morto" in dead.reason


def test_clean_signal_is_green(monitor, cfg, synthetic_eeg):
    eeg = synthetic_eeg(seconds=cfg.contact_monitor.window_s, alpha_uv=8.0, noise_uv=4.0)
    window = np.zeros((17, eeg.shape[1]), dtype=np.float32)
    window[EEG_SLICE] = eeg
    report = monitor.evaluate(window)
    assert report.n_good == 8
    assert report.global_warning is None


def test_line_noise_is_measured_despite_being_outside_the_passband(
    monitor, cfg, synthetic_eeg
):
    """O indice de rede mede 48-52 Hz numa janela so detrended.

    Se fosse medido depois do passa-banda de 1-45 Hz estaria atenuado ~40 dB e
    nunca dispararia — o bug que esta verificacao previne.
    """
    seconds = cfg.contact_monitor.window_s
    eeg = synthetic_eeg(seconds=seconds, noise_uv=3.0)
    t = np.arange(eeg.shape[1]) / cfg.device.sfreq_nominal
    eeg[2] += 60.0 * np.sin(2 * np.pi * cfg.device.line_freq_hz * t)

    window = np.zeros((17, eeg.shape[1]), dtype=np.float32)
    window[EEG_SLICE] = eeg
    report = monitor.evaluate(window)

    contaminated = report.by_name(cfg.montage.channels[2])
    clean = report.by_name(cfg.montage.channels[0])
    assert contaminated.line_rel > clean.line_rel * 5


def test_line_noise_alone_does_not_disqualify_a_channel(monitor, cfg, synthetic_eeg):
    """A rede e informativa: o notch remove-a antes de qualquer marcador.

    Reprovar um canal por causa da rede era rejeitar sinal aproveitavel pelo
    ruido que o proprio pipeline elimina a seguir.
    """
    seconds = cfg.contact_monitor.window_s
    eeg = synthetic_eeg(seconds=seconds, noise_uv=3.0)
    t = np.arange(eeg.shape[1]) / cfg.device.sfreq_nominal
    eeg += 80.0 * np.sin(2 * np.pi * cfg.device.line_freq_hz * t)

    window = np.zeros((17, eeg.shape[1]), dtype=np.float32)
    window[EEG_SLICE] = eeg
    report = monitor.evaluate(window)

    assert report.n_good == 8, "a rede sozinha nao pode reprovar canais"
    assert report.line_warning is not None, "mas tem de avisar sobre o ambiente"


def test_floating_electrode_is_caught_by_raw_amplitude(monitor, cfg, synthetic_eeg):
    """Um eletrodo a flutuar da amplitude bruta absurda, mesmo que filtre bem."""
    seconds = cfg.contact_monitor.window_s
    eeg = synthetic_eeg(seconds=seconds, noise_uv=4.0, dc_offset_uv=0.0)
    t = np.arange(eeg.shape[1]) / cfg.device.sfreq_nominal
    eeg[5] += 60_000.0 * t  # rampa de dezenas de mV em 2 s

    window = np.zeros((17, eeg.shape[1]), dtype=np.float32)
    window[EEG_SLICE] = eeg
    report = monitor.evaluate(window)

    floating = report.by_name(cfg.montage.channels[5])
    assert floating.level == "red"
    assert "flutuar" in floating.reason


def test_slope_is_reported_but_does_not_grade(monitor, cfg):
    """O declive e informativo. Numa janela de 2 s e ruidoso demais para decidir.

    Com ruido branco puro os declives medidos espalham-se de -0,5 a +0,7: usa-lo
    como criterio reprovaria canais bons e aprovaria ruido, ao acaso. Quem
    arbitra se um canal mede cerebro e o teste de Berger, com minutos de dados.
    """
    n = int(cfg.contact_monitor.window_s * cfg.device.sfreq_nominal)
    rng = np.random.default_rng(0)
    window = np.zeros((17, n), dtype=np.float32)
    window[EEG_SLICE] = rng.normal(0, 15.0, (8, n))

    report = monitor.evaluate(window)
    assert all(np.isfinite(c.slope) for c in report.channels), "declive tem de ser reportado"
    assert not any("1/f" in c.reason for c in report.channels), (
        "o declive nao pode aparecer como motivo de reprovacao"
    )


def test_slope_separates_eeg_from_white_noise_with_enough_data(cfg, synthetic_eeg):
    """Com 30 s em vez de 2 s, o declive ja discrimina — e por isso serve na analise."""
    from mantraeeg.spectral import spectral_slope, welch_psd

    sfreq = cfg.device.sfreq_nominal
    rng = np.random.default_rng(1)
    eeg = synthetic_eeg(seconds=30.0, noise_uv=8.0, dc_offset_uv=0.0)
    white = rng.normal(0, 15.0, eeg.shape)

    freqs, psd_eeg = welch_psd(eeg, sfreq, seg_s=4.0)
    _, psd_white = welch_psd(white, sfreq, seg_s=4.0)
    fit = cfg.contact_monitor.slope_fit_range

    slope_eeg = spectral_slope(psd_eeg, freqs, fit_range=fit)
    slope_white = spectral_slope(psd_white, freqs, fit_range=fit)

    assert abs(np.median(slope_white)) < 0.2, "ruido branco tem de ser plano"
    assert np.median(slope_eeg) < -0.25, "sinal com componente 1/f tem de decair"
    # A separacao e o que interessa: com 2 s as duas distribuicoes sobrepoem-se,
    # com 30 s deixam de se sobrepor.
    assert max(slope_eeg) < min(slope_white), (
        f"distribuicoes ainda sobrepostas: eeg {np.round(slope_eeg, 2)} vs "
        f"branco {np.round(slope_white, 2)}"
    )


# --------------------------------------------------------------------------- #
# Referência/terra soltos
# --------------------------------------------------------------------------- #
def test_common_mode_only_signal_is_caught(monitor, cfg):
    """A assinatura observada com os clips soltos: 8 canais, um só sinal.

    Correlação 1,000000 entre todos os pares, diferenças sub-µV, e uma rampa DC
    monotónica. O indicador de validação do dispositivo leu 1 durante todo esse
    registo, por isso a deteção tem de ser nossa.
    """
    sfreq = cfg.device.sfreq_nominal
    n = int(cfg.contact_monitor.window_s * sfreq)
    rng = np.random.default_rng(1)
    common = np.cumsum(rng.normal(0, 40, n)) + np.linspace(45_000, 243_000, n)

    eeg = np.tile(common, (8, 1))
    eeg += rng.normal(0, 0.3, eeg.shape)  # diferença ínfima face ao sinal comum
    window = np.zeros((17, n), dtype=np.float32)
    window[EEG_SLICE] = eeg

    report = monitor.evaluate(window)
    assert report.global_warning is not None
    assert "mesmo sinal" in report.global_warning
    assert "referência" in report.global_warning


def test_independent_channels_raise_no_warning(monitor, cfg, synthetic_eeg):
    eeg = synthetic_eeg(seconds=cfg.contact_monitor.window_s, seed=7)
    window = np.zeros((17, eeg.shape[1]), dtype=np.float32)
    window[EEG_SLICE] = eeg
    assert monitor.evaluate(window).global_warning is None


def test_short_window_is_handled(monitor):
    assert monitor.evaluate(np.zeros((17, 1), dtype=np.float32)).channels == ()
