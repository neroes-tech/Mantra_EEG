"""Conectividade entre pares de elétrodos (``BIOMARKERS.md`` B10).

Cadência própria, mais lenta que a das métricas de potência: janelas de 10 s
com sub-segmentos de 1 s dão 19 segmentos, que é o mínimo para estimar
coerência. Uma janela de 4 s com sub-segmentos de 2 s daria 3, e não chega.

Três medidas, e é preciso as três:

``coherence`` (MSC)
    A medida da literatura de Meditação Transcendental. Mas a referência comum
    e a condução de volume inflacionam-na entre elétrodos próximos.

``imag_coherence``
    Parte imaginária da mesma cross-spectrum — custo zero, e por construção
    insensível à fuga instantânea da condução de volume e da referência comum.

``debiased_wpli``
    wPLI ao quadrado desenviesado (Vinck). Cego a acoplamento a lag zero por
    construção, o que o torna verificação e não substituto: existe trabalho que
    reporta sincronia alfa **a lag zero** durante MT, e o wPLI não a veria.

Ler as três em conjunto: MSC↑ com ImCoh↑ e dwPLI↑ é acoplamento com atraso,
real. MSC↑ com as outras duas paradas é artefacto de referência **ou**
sincronia genuína a lag zero — e as duas não se distinguem. Dizer isso é mais
honesto do que anunciar "a coerência subiu".
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal


def _cross_spectra(
    x: np.ndarray, y: np.ndarray, sfreq: float, seg_s: float, overlap: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Cross-spectra por segmento, sem os promediar.

    Devolve ``(freqs, Sxy, Sxx, Syy, n_segmentos)`` com ``Sxy`` complexo por
    segmento — as três medidas precisam da distribuição, não só da média.
    """
    nperseg = min(int(round(seg_s * sfreq)), len(x))
    if nperseg < 8:
        raise ValueError(f"sub-segmento demasiado curto: {nperseg} amostras")
    step = max(int(round(nperseg * (1.0 - overlap))), 1)

    window = sp_signal.get_window("hann", nperseg)
    scale = 1.0 / (sfreq * (window**2).sum())
    freqs = np.fft.rfftfreq(nperseg, 1.0 / sfreq)

    starts = range(0, len(x) - nperseg + 1, step)
    fx, fy = [], []
    for start in starts:
        fx.append(np.fft.rfft(sp_signal.detrend(x[start : start + nperseg]) * window))
        fy.append(np.fft.rfft(sp_signal.detrend(y[start : start + nperseg]) * window))
    if not fx:
        raise ValueError("janela demasiado curta para um único sub-segmento")

    fx_arr, fy_arr = np.array(fx), np.array(fy)
    sxy = fx_arr * np.conj(fy_arr) * scale
    sxx = (np.abs(fx_arr) ** 2) * scale
    syy = (np.abs(fy_arr) ** 2) * scale
    return freqs, sxy, sxx, syy, len(fx)


def _band(freqs: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    return (freqs >= band[0]) & (freqs <= band[1])


def coherence_band(
    x: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
    seg_s: float = 1.0,
    min_segments: int = 19,
) -> float:
    """Coerência de magnitude quadrada (MSC) média na banda.

    A MSC estimada a partir de ``L`` segmentos tem viés positivo de cerca de
    ``1/L`` mesmo para sinais independentes. Por isso o número de segmentos tem
    de ser constante entre condições — daqui sai ``nan`` se não houver
    ``min_segments``, em vez de um valor com viés diferente dos outros.
    """
    freqs, sxy, sxx, syy, n = _cross_spectra(x, y, sfreq, seg_s)
    if n < min_segments:
        return float("nan")
    mask = _band(freqs, band)
    num = np.abs(sxy[:, mask].mean(axis=0)) ** 2
    den = sxx[:, mask].mean(axis=0) * syy[:, mask].mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        msc = np.where(den > 0, num / den, np.nan)
    return float(np.nanmean(msc))


def imaginary_coherence_band(
    x: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
    seg_s: float = 1.0,
    min_segments: int = 19,
) -> float:
    """Parte imaginária da coerência, em módulo.

    Insensível por construção à contribuição instantânea da referência comum e
    da condução de volume, que é o que inflaciona a MSC entre elétrodos
    próximos.
    """
    freqs, sxy, sxx, syy, n = _cross_spectra(x, y, sfreq, seg_s)
    if n < min_segments:
        return float("nan")
    mask = _band(freqs, band)
    mean_sxy = sxy[:, mask].mean(axis=0)
    den = np.sqrt(sxx[:, mask].mean(axis=0) * syy[:, mask].mean(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.where(den > 0, np.abs(np.imag(mean_sxy)) / den, np.nan)
    return float(np.nanmean(value))


def debiased_wpli_band(
    x: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
    seg_s: float = 1.0,
    min_segments: int = 19,
) -> float:
    """wPLI ao quadrado **desenviesado** (Vinck et al.).

    O wPLI simples é positivamente enviesado e de alta variância com poucos
    segmentos; com os 19 desta grelha o viés é material. O estimador
    desenviesado remove o termo de auto-produto e é o que se deve usar quando o
    wPLI arbitra alguma coisa.
    """
    freqs, sxy, _, _, n = _cross_spectra(x, y, sfreq, seg_s)
    if n < min_segments:
        return float("nan")
    mask = _band(freqs, band)
    imag = np.imag(sxy[:, mask])

    sum_imag = imag.sum(axis=0)
    sum_abs = np.abs(imag).sum(axis=0)
    sum_sq = (imag**2).sum(axis=0)

    numerator = sum_imag**2 - sum_sq
    denominator = sum_abs**2 - sum_sq
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.where(denominator > 0, numerator / denominator, np.nan)
    return float(np.nanmean(value))
