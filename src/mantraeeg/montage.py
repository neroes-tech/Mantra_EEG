"""Validação da montagem.

A montagem é personalizada (SPEC 1.2) e o mapeamento índice→elétrodo pode estar
trocado. Se F3 e F4 estiverem invertidos, o ALAY sai com o sinal ao contrário e
a coerência fica errada, e ninguém repara — o gráfico continua bonito.

Quatro verificações, três delas gratuitas:

``teste de toque`` (primária, definitiva)
    O operador toca num elétrodo de cada vez, por nome; o código diz qual o
    **índice** que reagiu. Valida o mapa inteiro, não só o par F3/F4.

``dominância frontopolar`` (automática)
    Fp1/Fp2 têm de dominar F3/F4 na banda de pestanejo. Confirma o eixo
    ântero-posterior e que os índices 4 e 5 são mesmo o par frontopolar.

``olhar lateral`` (automática)
    Olhar à esquerda e depois à direita produz deflexões de polaridade **oposta**
    em Fp1 e Fp2. É esta que resolve esquerda/direita.

``independência entre canais`` (automática, corre sempre)
    Oito elétrodos em contacto com um escalpe nunca correlacionam a 1,0 entre si.
    Quando correlacionam, o que está a ser gravado é deriva de modo comum.

As automáticas correm sobre os primeiros segundos de **cada** sessão e avisam se
a ordenação esperada for violada. É uma sentinela permanente, não uma
verificação única de arranque.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import signal

from .config import Config


@dataclass(frozen=True)
class CheckResult:
    """Resultado de uma verificação. ``passed=None`` = não foi possível avaliar."""

    name: str
    passed: bool | None
    detail: str
    values: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "values": self.values,
        }


def _detrended(eeg: np.ndarray) -> np.ndarray:
    return signal.detrend(np.asarray(eeg, dtype=np.float64), axis=-1, type="linear")


def _bandpassed(eeg: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    nyq = sfreq / 2.0
    lo, hi = band[0] / nyq, min(band[1] / nyq, 0.99)
    sos = signal.butter(4, [lo, hi], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, _detrended(eeg), axis=-1)


# --------------------------------------------------------------------------- #
# Independência entre canais
# --------------------------------------------------------------------------- #
def check_channel_independence(
    eeg: np.ndarray, min_r: float, max_pair_diff_ratio: float
) -> CheckResult:
    """Deteta referência/terra soltos: todos os canais a ver o mesmo sinal.

    Assinatura observada num headset com os clips soltos: correlação entre todos
    os pares igual a 1,000000, e a diferença entre quaisquer dois canais com uma
    amplitude ínfima face à de cada canal — 1 µV de diferença para 1490 µV de
    sinal numa aquisição, 7 µV para 10500 µV noutra. O indicador de validação do
    próprio dispositivo leu ``1`` durante todo esse registo, pelo que não serve.

    O critério da diferença é **relativo** à amplitude do canal, e não em µV
    absolutos: a diferença residual escala com o sinal comum, e um limiar fixo
    deixa passar exatamente os casos de maior amplitude, que são os piores.
    """
    eeg = _detrended(eeg)
    n_ch = eeg.shape[0]
    if n_ch < 2 or eeg.shape[1] < 2:
        return CheckResult("independencia_canais", None, "dados insuficientes")

    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(eeg)
    off = corr[~np.eye(n_ch, dtype=bool)]
    if not np.all(np.isfinite(off)):
        return CheckResult("independencia_canais", None, "correlação indefinida")

    min_corr = float(np.min(off))
    diffs = [
        float((eeg[i] - eeg[j]).std())
        for i in range(n_ch)
        for j in range(i + 1, n_ch)
    ]
    max_diff = float(max(diffs)) if diffs else 0.0
    median_sd = float(np.median(eeg.std(axis=-1)))
    diff_ratio = max_diff / median_sd if median_sd > 0 else float("inf")
    values = {
        "min_interchannel_r": min_corr,
        "max_pair_diff_uv": max_diff,
        "median_channel_sd_uv": median_sd,
        "max_pair_diff_ratio": diff_ratio,
    }

    if min_corr >= min_r and diff_ratio <= max_pair_diff_ratio:
        return CheckResult(
            "independencia_canais",
            False,
            (
                f"os {n_ch} canais transportam o mesmo sinal (correlação mínima "
                f"{min_corr:.4f}; a maior diferença entre dois canais tem "
                f"{max_diff:.2f} µV contra {median_sd:.0f} µV de sinal por canal, "
                f"ou seja {diff_ratio * 100:.2f} %). Não é EEG: verificar primeiro "
                f"os clips de referência e terra, depois o contacto dos elétrodos "
                f"do escalpe."
            ),
            values,
        )
    return CheckResult(
        "independencia_canais",
        True,
        (
            f"canais independentes (correlação mínima {min_corr:.3f}, maior "
            f"diferença entre pares {diff_ratio * 100:.1f} % do sinal)"
        ),
        values,
    )


# --------------------------------------------------------------------------- #
# Dominância frontopolar
# --------------------------------------------------------------------------- #
def check_frontopolar_dominance(eeg: np.ndarray, cfg: Config) -> CheckResult:
    """Fp1/Fp2 devem dominar F3/F4 na banda de pestanejo (1–5 Hz)."""
    v = cfg.montage.validation
    m = cfg.montage
    band = _bandpassed(eeg, cfg.device.sfreq_nominal, v.blink_band_hz)
    amp = band.std(axis=-1)

    try:
        fp = [amp[m.index_of(n)] for n in m.eog_channels]
        frontal = [amp[m.index_of(n)] for n in ("F3", "F4") if n in m.name_to_index]
    except KeyError as exc:
        return CheckResult("dominancia_frontopolar", None, str(exc))
    if not fp or not frontal:
        return CheckResult("dominancia_frontopolar", None, "canais em falta")

    fp_amp, f_amp = float(np.mean(fp)), float(np.mean(frontal))
    ratio = fp_amp / f_amp if f_amp > 0 else float("inf")
    values = {
        "frontopolar_uv": fp_amp,
        "frontal_uv": f_amp,
        "ratio": ratio,
        "min_ratio": v.frontopolar_dominance_min_ratio,
    }
    if ratio >= v.frontopolar_dominance_min_ratio:
        return CheckResult(
            "dominancia_frontopolar",
            True,
            f"Fp1/Fp2 dominam F3/F4 em {ratio:.2f}x na banda de pestanejo",
            values,
        )
    return CheckResult(
        "dominancia_frontopolar",
        False,
        (
            f"Fp1/Fp2 só excedem F3/F4 em {ratio:.2f}x (mínimo "
            f"{v.frontopolar_dominance_min_ratio:.2f}x) na banda de pestanejo. "
            f"Ou os índices 4 e 5 não são o par frontopolar, ou a pessoa não "
            f"pestanejou durante a janela analisada."
        ),
        values,
    )


# --------------------------------------------------------------------------- #
# Olhar lateral: resolve esquerda/direita
# --------------------------------------------------------------------------- #
def check_lateral_gaze(eeg: np.ndarray, cfg: Config) -> CheckResult:
    """Fp1 e Fp2 devem deflectir em **polaridades opostas** no olhar lateral.

    É a única verificação automática que distingue esquerda de direita, e por
    isso a que sustenta a convenção hemisférica de F3 vs F4.
    """
    v = cfg.montage.validation
    m = cfg.montage
    try:
        i1, i2 = m.index_of("Fp1"), m.index_of("Fp2")
    except KeyError as exc:
        return CheckResult("olhar_lateral", None, str(exc))

    band = _bandpassed(eeg, cfg.device.sfreq_nominal, v.blink_band_hz)
    with np.errstate(invalid="ignore"):
        r = float(np.corrcoef(band[i1], band[i2])[0, 1])
    values = {"r_fp1_fp2": r}
    if not np.isfinite(r):
        return CheckResult("olhar_lateral", None, "correlação indefinida", values)
    if r < 0:
        return CheckResult(
            "olhar_lateral",
            True,
            f"Fp1 e Fp2 em anti-fase (r = {r:.2f}): componente horizontal presente",
            values,
        )
    return CheckResult(
        "olhar_lateral",
        None,
        (
            f"Fp1 e Fp2 em fase (r = {r:.2f}). Só se vê pestanejo vertical; a "
            f"lateralidade não fica confirmada. Peça um olhar esquerda-direita "
            f"e repita, ou use o teste de toque."
        ),
        values,
    )


# --------------------------------------------------------------------------- #
# Teste de toque
# --------------------------------------------------------------------------- #
def identify_tapped_channel(eeg: np.ndarray, cfg: Config) -> tuple[int | None, float]:
    """Índice do canal que reagiu a um toque, e a razão de destaque.

    O canal tocado tem de exceder a mediana dos restantes pelo fator
    ``tap_test_min_ratio``; caso contrário devolve ``None``.
    """
    amp = _detrended(eeg).std(axis=-1)
    if amp.size == 0:
        return None, 0.0
    idx = int(np.argmax(amp))
    others = np.delete(amp, idx)
    baseline = float(np.median(others)) if others.size else 0.0
    ratio = float(amp[idx] / baseline) if baseline > 0 else float("inf")
    if ratio < cfg.montage.validation.tap_test_min_ratio:
        return None, ratio
    return idx, ratio


# --------------------------------------------------------------------------- #
# Conjunto automático
# --------------------------------------------------------------------------- #
def auto_checks(eeg: np.ndarray, cfg: Config) -> list[CheckResult]:
    """As verificações que correm sozinhas no arranque de cada sessão."""
    v = cfg.montage.validation
    results = [
        check_channel_independence(
            eeg, v.no_contact_min_interchannel_r, v.no_contact_max_pair_diff_ratio
        )
    ]
    # Sem canais independentes, as restantes não significam nada.
    if results[0].passed is False:
        for name in ("dominancia_frontopolar", "olhar_lateral"):
            results.append(
                CheckResult(name, None, "não avaliada: os canais não são independentes")
            )
        return results
    results.append(check_frontopolar_dominance(eeg, cfg))
    results.append(check_lateral_gaze(eeg, cfg))
    return results


def summarize(results: list[CheckResult]) -> dict[str, Any]:
    """Forma serializável para ``raw_meta.json``."""
    return {
        "checks": [r.as_dict() for r in results],
        "any_failed": any(r.passed is False for r in results),
        "all_passed": all(r.passed is True for r in results),
    }
