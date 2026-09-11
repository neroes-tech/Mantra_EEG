"""Estimativa espectral. Assinaturas conforme ``BIOMARKERS.md``.

Funções puras: recebem dados, devolvem números. Sem estado, sem I/O, sem leitura
de config global.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal


def welch_psd(
    x: np.ndarray,
    sfreq: float,
    seg_s: float = 2.0,
    overlap: float = 0.5,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """PSD de Welch.

    ``x``: ``(n_ch, n_samples)`` de uma época, ou ``(n_samples,)``.
    Devolve ``(freqs, psd)`` com ``psd`` em ``(n_ch, n_freqs)`` e unidades de
    µV²/Hz.

    Com ``seg_s`` de 2,0 s a 250 Hz a resolução é 0,5 Hz, conforme SPEC 3.2.
    """
    x = np.asarray(x, dtype=np.float64)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[None, :]

    nperseg = min(int(round(seg_s * sfreq)), x.shape[-1])
    if nperseg < 2:
        raise ValueError(
            f"segmento demasiado curto: {nperseg} amostras para seg_s={seg_s}"
        )
    noverlap = int(round(nperseg * overlap))

    freqs, psd = sp_signal.welch(
        x, fs=sfreq, nperseg=nperseg, noverlap=noverlap, window=window, axis=-1
    )
    return freqs, psd[0] if squeeze else psd


def _band_mask(freqs: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    return (freqs >= band[0]) & (freqs <= band[1])


def band_power(
    psd: np.ndarray, freqs: np.ndarray, band: tuple[float, float]
) -> np.ndarray:
    """Integra a PSD na banda pela regra do trapézio. Devolve ``(n_ch,)`` em µV²."""
    mask = _band_mask(freqs, band)
    if not mask.any():
        raise ValueError(
            f"banda {band} não intersecta as frequências disponíveis "
            f"({freqs[0]:.2f}–{freqs[-1]:.2f} Hz)"
        )
    return np.trapezoid(psd[..., mask], freqs[mask], axis=-1)


def relative_power(
    psd: np.ndarray,
    freqs: np.ndarray,
    band: tuple[float, float],
    total: tuple[float, float] = (1.0, 45.0),
) -> np.ndarray:
    """``band_power(band) / band_power(total)``, por canal. Adimensional, 0–1.

    Devolve ``nan`` onde o denominador não for positivo, em vez de ``inf``.
    """
    num = band_power(psd, freqs, band)
    den = band_power(psd, freqs, total)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def spectral_slope(
    psd: np.ndarray,
    freqs: np.ndarray,
    fit_range: tuple[float, float] = (2.0, 40.0),
    peak_mask_bands: tuple[tuple[float, float], ...] = ((7.0, 14.0),),
) -> np.ndarray:
    """Declive log-log robusto, com as bandas oscilatórias mascaradas.

    Estimador rápido do componente aperiódico, para uso por época — o specparam
    completo custa dezenas de ms por ajuste e não cabe no orçamento de 10 s
    (ver SPEC 3, e ``markers.ap_exponent.method_per_epoch`` na config).

    Devolve o **declive** (negativo para EEG). O expoente aperiódico é o simétrico.
    """
    psd = np.atleast_2d(np.asarray(psd, dtype=np.float64))
    mask = _band_mask(freqs, fit_range) & (freqs > 0)
    for band in peak_mask_bands:
        mask &= ~_band_mask(freqs, band)
    if mask.sum() < 3:
        return np.full(psd.shape[0], np.nan)

    log_f = np.log10(freqs[mask])
    out = np.empty(psd.shape[0])
    for i, row in enumerate(psd):
        values = row[mask]
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            out[i] = np.nan
            continue
        out[i] = np.polyfit(log_f, np.log10(values), 1)[0]
    return out


def peak_frequency(
    psd: np.ndarray, freqs: np.ndarray, band: tuple[float, float]
) -> np.ndarray:
    """Frequência do máximo da PSD dentro da banda, por canal.

    Estimativa crua (o valor do relatório vem do specparam). Serve para
    verificar se um canal tem um pico alfa onde deve ter.
    """
    psd = np.atleast_2d(np.asarray(psd, dtype=np.float64))
    mask = _band_mask(freqs, band)
    if not mask.any():
        return np.full(psd.shape[0], np.nan)
    sub, sub_f = psd[:, mask], freqs[mask]
    return sub_f[np.argmax(sub, axis=-1)]


# --------------------------------------------------------------------------- #
# Componente aperiódico e pico alfa individual
# --------------------------------------------------------------------------- #
def aperiodic_baseline(
    psd: np.ndarray,
    freqs: np.ndarray,
    fit_range: tuple[float, float] = (2.0, 40.0),
    exclude: tuple[tuple[float, float], ...] = ((7.0, 14.0),),
) -> np.ndarray:
    """Ajuste 1/f em log-log, com as bandas oscilatórias excluídas do ajuste.

    Devolve a linha de base avaliada em **todas** as ``freqs``, para se poder
    subtrair. Estimador leve; o specparam completo fica para os ajustes por fase
    (``markers.ap_exponent.method_per_phase``).
    """
    psd = np.asarray(psd, dtype=np.float64)
    mask = (freqs >= fit_range[0]) & (freqs <= fit_range[1]) & (freqs > 0)
    for lo, hi in exclude:
        mask &= ~((freqs >= lo) & (freqs <= hi))
    if mask.sum() < 3 or np.any(psd[mask] <= 0):
        return np.full_like(np.asarray(freqs, dtype=np.float64), np.nan)
    coeffs = np.polyfit(np.log10(freqs[mask]), np.log10(psd[mask]), 1)
    safe = np.maximum(np.asarray(freqs, dtype=np.float64), 1e-9)
    return 10.0 ** np.polyval(coeffs, np.log10(safe))


def periodic_power(
    psd: np.ndarray,
    freqs: np.ndarray,
    band: tuple[float, float],
    fit_range: tuple[float, float] = (2.0, 40.0),
) -> float:
    """Potência da banda **acima** do 1/f. Isola oscilação de banda larga.

    Imune tanto ao denominador da potência relativa como a deslocamentos
    globais do espectro — é a medida que distingue "o alfa subiu" de "o resto
    desceu" (``BIOMARKERS.md`` B7, ``corrected_band_power``).
    """
    baseline = aperiodic_baseline(psd, freqs, fit_range)
    if not np.all(np.isfinite(baseline)):
        return float("nan")
    mask = _band_mask(freqs, band)
    excess = np.maximum(np.asarray(psd)[mask] - baseline[mask], 0.0)
    return float(np.trapezoid(excess, freqs[mask]))


def individual_alpha_frequency(
    psd: np.ndarray,
    freqs: np.ndarray,
    search: tuple[float, float] = (7.0, 14.0),
    fit_range: tuple[float, float] = (2.0, 40.0),
    smooth_hz: float = 1.0,
) -> float:
    """Frequência do pico alfa: um **máximo local** acima do 1/f.

    Duas armadilhas, ambas medidas neste projeto:

    Procurar o máximo da PSD crua devolve a aresta inferior da janela, porque o
    1/f decai monotonamente e domina qualquer pico.

    Subtrair uma reta em log-log e procurar o maior excesso **também** falha: o
    espectro tem joelho, a reta subestima o extremo inferior, e o excesso volta
    a ser máximo aos 7 Hz. Deu 7,1 Hz em sete de oito canais.

    O que identifica um pico é ser um **máximo local** — a curva tem de subir e
    depois descer. É isso que se procura aqui, sobre o excesso suavizado, e
    ignorando as arestas da janela de procura.

    Devolve ``nan`` quando não há pico, que é uma resposta legítima: nem toda a
    gente tem um pico alfa visível, sobretudo de olhos abertos.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    baseline = aperiodic_baseline(psd, freqs, fit_range, exclude=(search,))
    if not np.all(np.isfinite(baseline)):
        return float("nan")

    excess = np.asarray(psd, dtype=np.float64) - baseline
    resolution = float(np.median(np.diff(freqs))) if freqs.size > 1 else 1.0
    width = max(int(round(smooth_hz / resolution)), 1)
    if width > 1:
        kernel = np.ones(width) / width
        excess = np.convolve(excess, kernel, mode="same")

    mask = _band_mask(freqs, search)
    indices = np.nonzero(mask)[0]
    if indices.size < 3:
        return float("nan")

    # Máximos locais estritos, excluindo as duas arestas da janela de procura.
    interior = indices[1:-1]
    peaks = [
        i
        for i in interior
        if excess[i] > excess[i - 1] and excess[i] > excess[i + 1] and excess[i] > 0
    ]
    if not peaks:
        return float("nan")
    return float(freqs[max(peaks, key=lambda i: excess[i])])


def band_reactivity(
    psd_a: np.ndarray,
    psd_b: np.ndarray,
    freqs: np.ndarray,
    band: tuple[float, float],
    flanks: tuple[tuple[float, float], ...],
) -> float:
    """Quanto a banda mudou **acima do que todo o espectro mudou**, em log2.

    Diferença de diferenças: ``log2(banda_b/banda_a) - log2(flancos_b/flancos_a)``.
    Se o espectro inteiro subir ou descer — deriva, artefacto, mudança do
    denominador — os dois termos cancelam-se e o resultado é ~0. Só uma
    alteração **específica da banda** produz valor diferente de zero.

    É a assinatura do efeito de Berger: a razão fechados/abertos fica abaixo de
    1 nos flancos e acima de 1 no alfa. Mais específico do que a magnitude do
    alfa sozinha, e por isso é o critério certo para um efeito modesto.
    """
    def ratio(band_: tuple[float, float]) -> float:
        pa = band_power(np.asarray(psd_a), freqs, band_)
        pb = band_power(np.asarray(psd_b), freqs, band_)
        return float(pb) / float(pa) if float(pa) > 0 else float("nan")

    target = ratio(band)
    flank_ratios = [ratio(f) for f in flanks]
    flank_ratios = [r for r in flank_ratios if np.isfinite(r) and r > 0]
    if not np.isfinite(target) or target <= 0 or not flank_ratios:
        return float("nan")
    return float(np.log2(target) - np.mean(np.log2(flank_ratios)))


def reactivity_peak_frequency(
    psd_a: np.ndarray,
    psd_b: np.ndarray,
    freqs: np.ndarray,
    search: tuple[float, float] = (6.0, 14.0),
    smooth_hz: float = 1.0,
) -> float:
    """Frequência onde a razão ``psd_b / psd_a`` é máxima.

    Para posicionar a banda alfa, o que interessa é a frequência que **reage**,
    e não a que tem mais potência em repouso. Neste projeto os dois divergiram:
    o pico estático de olhos fechados ficou nos 7,5–8,2 Hz nos canais frontais,
    enquanto a razão fechados/abertos só atinge o máximo perto dos 10 Hz. O
    pico estático está contaminado pelo ombro teta/alfa; o pico de reatividade
    é o componente que responde ao fecho das pálpebras.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    pa = np.asarray(psd_a, dtype=np.float64)
    pb = np.asarray(psd_b, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(pa > 0, pb / pa, np.nan)

    resolution = float(np.median(np.diff(freqs))) if freqs.size > 1 else 1.0
    width = max(int(round(smooth_hz / resolution)), 1)
    if width > 1:
        kernel = np.ones(width) / width
        ratio = np.convolve(np.nan_to_num(ratio, nan=0.0), kernel, mode="same")

    mask = _band_mask(freqs, search) & np.isfinite(ratio)
    if not mask.any():
        return float("nan")
    return float(freqs[mask][int(np.argmax(ratio[mask]))])
