"""Pré-processamento post-hoc (SPEC 3.1).

Corre uma vez sobre a gravação completa, depois da sessão terminar. É por isso
que se pode usar filtragem de **fase zero** (``sosfiltfilt``) e regressão de EOG
— nenhuma das duas é possível em tempo real, e são a razão de ser desta
arquitetura.

Ordem: detrend → notch → passa-banda → correção de EOG.

Duas coisas que a SPEC não diz e que estão aqui:

**Margem de aresta.** Um Butterworth de ordem 4 com corte a 1 Hz aplicado com
``filtfilt`` contamina os primeiros e últimos segundos do registo. Sem descartar
essa margem, as primeiras épocas de calibração — que fixam a referência de
normalização — são transiente de filtro.

**Cópia só detrended.** Os índices de EMG e de rede medem 30–45 Hz e 48–52 Hz.
Depois de um passa-banda de 1–45 Hz a banda da rede está atenuada dezenas de dB
e o índice nunca dispararia. :class:`Preprocessed` transporta as duas versões.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from scipy import signal as sp_signal

from .config import Config


@dataclass(frozen=True)
class Preprocessed:
    """Resultado do pré-processamento, com a proveniência toda anexada."""

    #: ``(n_ch, n)`` filtrado 1–45 Hz, com notch e EOG corrigido. Para marcadores.
    clean: np.ndarray
    #: ``(n_ch, n)`` apenas detrended. Para medir o AMBIENTE de rede, isto é,
    #: quanta rede o local tem — que é diferente de quanta sobra na análise.
    detrended: np.ndarray
    #: ``(n_ch, n)`` com notch mas SEM passa-banda. É esta a cópia certa para o
    #: gate de qualidade: o EMG (30–45 Hz) não é atenuado pelo corte a 45 Hz, e
    #: o índice de rede mede o RESÍDUO que sobrevive ao notch — que é o que
    #: contamina a análise. Medir a rede antes do notch e usá-la como critério
    #: rejeitava 100 % das épocas em dados reais.
    notched: np.ndarray
    sfreq: float
    #: Amostras descartadas em cada ponta por causa do transiente de filtro.
    edge_samples: int
    #: Instante, no registo original, a que corresponde a amostra 0 destes arrays.
    t0_s: float
    eog_applied: bool
    eog_report: Mapping[str, Any] = field(default_factory=dict)
    #: Máscara das amostras em que o amplificador saturou, já alinhada com os
    #: arrays acima. O gate de qualidade rejeita as épocas que lhe tocam.
    saturation: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=bool)
    )
    #: Fração do registo inteiro que estava saturada, para o diagnóstico e
    #: para o aviso ao operador.
    saturated_fraction: float = 0.0

    @property
    def n_samples(self) -> int:
        return int(self.clean.shape[1])

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sfreq

    def time(self) -> np.ndarray:
        """Eixo temporal em segundos, na base de tempo do registo original."""
        return self.t0_s + np.arange(self.n_samples) / self.sfreq


# --------------------------------------------------------------------------- #
# Blocos
# --------------------------------------------------------------------------- #
def find_saturation(
    raw_eeg: np.ndarray, rail_uv: float, pad_s: float, sfreq: float
) -> np.ndarray:
    """Máscara ``(n_ch, n)`` das amostras em que o amplificador bateu no fundo.

    O Unicorn satura em ±750 000 µV. Uma sessão real trazida da banca
    (dispositivo 17) tinha 0,2 % das amostras exatamente nesse valor, em seis
    dos oito canais, concentradas em cerca de um minuto: um pico de 166 mV
    pico a pico no meio de uma gravação que, fora disso, era utilizável.

    **O estrago não fica no minuto mau.** O passa-banda é aplicado com
    ``filtfilt``, que é não causal: um degrau de 750 mV faz o filtro oscilar
    para os dois lados durante muitos segundos, e o resíduo entra no cálculo
    do MAD e dos limiares, que são globais. Nessa sessão a cobertura final foi
    **0 %** — a análise inteira foi perdida por causa de um minuto.

    Por isso as amostras são marcadas **antes** de filtrar, e substituídas
    pela mediana do canal em :func:`excise`. Não é inventar sinal: é impedir
    que sinal que não existe contamine o que existe. As épocas afetadas são
    rejeitadas a jusante na mesma, pelo gate de qualidade.

    ``pad_s`` alarga a máscara para cada lado, porque a aproximação ao limite
    e a recuperação também não são sinal.
    """
    raw = np.asarray(raw_eeg, dtype=np.float64)
    mask = np.abs(raw) >= rail_uv
    pad = int(round(pad_s * sfreq))
    if pad <= 0 or not mask.any():
        return mask
    # Dilatação por convolução: mais barato do que percorrer os intervalos.
    kernel = np.ones(2 * pad + 1)
    for i in range(mask.shape[0]):
        if mask[i].any():
            mask[i] = np.convolve(mask[i].astype(float), kernel, mode="same") > 0
    return mask


def excise(raw_eeg: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Substitui as amostras marcadas pela mediana do próprio canal.

    Um canal inteiro marcado fica a zeros — não há mediana de nada, e um canal
    saturado do princípio ao fim não tem informação a preservar.
    """
    out = np.array(raw_eeg, dtype=np.float64, copy=True)
    for i in range(out.shape[0]):
        bad = mask[i]
        if not bad.any():
            continue
        good = out[i][~bad]
        out[i][bad] = float(np.median(good)) if good.size else 0.0
    return out


def detrend(x: np.ndarray, kind: str = "linear") -> np.ndarray:
    """Remove média e tendência linear por canal.

    Obrigatório antes de tudo: o Unicorn entrega µV com offset de eletrodo de
    centenas de mV, diferente em cada canal.
    """
    x = np.asarray(x, dtype=np.float64)
    if kind == "none":
        return x.copy()
    if kind not in ("linear", "constant"):
        raise ValueError(f"detrend {kind!r} desconhecido ('linear'|'constant'|'none')")
    return sp_signal.detrend(x, axis=-1, type=kind)


def notch(
    x: np.ndarray,
    sfreq: float,
    freqs: tuple[float, ...],
    q: float,
    passes: int = 1,
) -> np.ndarray:
    """Notch IIR de fase zero em cada frequência indicada.

    ``passes`` aplica o filtro repetidamente. Com a rede muito forte, uma só
    passagem deixa resíduo apreciável nas saias do notch (49 e 51 Hz), onde a
    energia da rede de facto se espalha. Duas passagens ganham 9 dB aí e custam
    0,2 dB adicionais no pior ponto da banda útil — muito melhor troca do que
    baixar o Q, que alargaria o notch para dentro da banda.
    """
    out = np.asarray(x, dtype=np.float64)
    for _ in range(max(passes, 1)):
        for f in freqs:
            b, a = sp_signal.iirnotch(f, q, fs=sfreq)
            out = sp_signal.filtfilt(b, a, out, axis=-1)
    return out


def bandpass(
    x: np.ndarray, sfreq: float, band: tuple[float, float], order: int
) -> np.ndarray:
    """Butterworth passa-banda de fase zero (``sosfiltfilt``)."""
    nyq = sfreq / 2.0
    sos = sp_signal.butter(
        order, [band[0] / nyq, band[1] / nyq], btype="bandpass", output="sos"
    )
    return sp_signal.sosfiltfilt(sos, np.asarray(x, dtype=np.float64), axis=-1)


def bandlimited_variance(
    x: np.ndarray, sfreq: float, band: tuple[float, float]
) -> np.ndarray:
    """Variância de ``x`` restrita a uma banda, por canal.

    Usada para quantificar quanta variância a regressão de EOG removeu **em cada
    banda**, que é o que decide se um marcador ficou comprometido.
    """
    x = np.asarray(x, dtype=np.float64)
    nyq = sfreq / 2.0
    hi = min(band[1] / nyq, 0.99)
    sos = sp_signal.butter(4, [band[0] / nyq, hi], btype="bandpass", output="sos")
    return sp_signal.sosfiltfilt(sos, x, axis=-1).var(axis=-1)


# --------------------------------------------------------------------------- #
# Correção de EOG por regressão
# --------------------------------------------------------------------------- #
def _blink_mask(
    eog: np.ndarray, sfreq: float, threshold_uv: float, pad_s: float
) -> np.ndarray:
    """Amostras dentro de um pestanejo, com margem em torno de cada evento."""
    amplitude = np.abs(eog).max(axis=0)
    hit = amplitude > threshold_uv
    pad = int(round(pad_s * sfreq))
    if pad > 0 and hit.any():
        kernel = np.ones(2 * pad + 1, dtype=bool)
        hit = np.convolve(hit, kernel, mode="same") > 0
    return hit


def regress_eog(
    data: np.ndarray,
    sfreq: float,
    target_idx: list[int],
    eog_idx: list[int],
    cfg_eog: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Subtrai dos canais de métrica a componente prevista pelos canais oculares.

    Os regressores são filtrados à banda ocular (1–5 Hz por defeito), portanto o
    que se subtrai vive só nessa banda: alfa, SMR e beta ficam intactos. **O teta
    (4–7 Hz) não fica** — sobrepõe-se à banda do regressor e é o marcador em
    risco. O relatório devolvido quantifica exatamente quanta variância foi
    removida em cada banda, para o diagnóstico poder sinalizá-lo.

    Em vez de Fp1 e Fp2 em bruto, usa por defeito a média e a diferença dos dois:
    os mesmos dois graus de liberdade, mas mais perto dos geradores oculares
    (média = pestanejo vertical, diferença = sacádico horizontal), e o canal de
    diferença transporta bem menos atividade cortical comum.
    """
    data = np.asarray(data, dtype=np.float64)
    out = data.copy()
    band = tuple(cfg_eog.regressor_band_hz)

    eog_raw = data[eog_idx]
    eog_band = bandpass(eog_raw, sfreq, band, order=4)

    if cfg_eog.use_mean_diff_regressors and len(eog_idx) == 2:
        regressors = np.vstack(
            [eog_band.mean(axis=0), eog_band[1] - eog_band[0]]
        )
        regressor_names = ["(Fp1+Fp2)/2", "Fp2-Fp1"]
    else:
        regressors = eog_band
        regressor_names = [f"eog{i}" for i in eog_idx]

    # Os coeficientes são estimados só onde há pestanejo. Estimados sobre tudo
    # seriam diluídos pelos períodos sem pestanejo, onde a covariância entre Fp e
    # F é cortical genuína — e é daí que vem a sobre-correção.
    fit_mask = np.ones(data.shape[1], dtype=bool)
    n_blink = 0
    if cfg_eog.fit_on_blink_segments_only:
        blink = _blink_mask(eog_band, sfreq, cfg_eog.blink_detect_uv, cfg_eog.blink_pad_s)
        n_blink = int(blink.sum())
        # Com olhos fechados os pestanejos são raros. Sem amostras suficientes a
        # regressão é pior que não fazer nada, por isso cai para todo o registo
        # e diz que o fez.
        min_samples = int(regressors.shape[0] * sfreq)
        if n_blink >= min_samples:
            fit_mask = blink

    design = np.vstack([regressors, np.ones(data.shape[1])])
    coefficients: dict[str, dict[str, float]] = {}
    variance_removed: dict[str, dict[str, float]] = {}

    for idx in target_idx:
        beta, *_ = np.linalg.lstsq(design[:, fit_mask].T, data[idx, fit_mask], rcond=None)
        predicted = design.T @ beta
        # O termo constante não é artefacto ocular; o detrend já tratou disso.
        predicted -= beta[-1]
        out[idx] = data[idx] - predicted
        coefficients[str(idx)] = dict(zip(regressor_names, beta[:-1].tolist()))

    return out, {
        "regressor_band_hz": list(band),
        "regressors": regressor_names,
        "fit_on_blinks_only": bool(cfg_eog.fit_on_blink_segments_only),
        "blink_samples": n_blink,
        "fit_samples": int(fit_mask.sum()),
        "fell_back_to_all_samples": bool(
            cfg_eog.fit_on_blink_segments_only and fit_mask.all()
        ),
        "coefficients": coefficients,
        "variance_removed": variance_removed,
    }


# --------------------------------------------------------------------------- #
# Orquestração
# --------------------------------------------------------------------------- #
def preprocess(
    raw_eeg: np.ndarray,
    cfg: Config,
    apply_eog: bool | None = None,
    t0_s: float = 0.0,
) -> Preprocessed:
    """Aplica a cadeia completa da SPEC 3.1 a ``(n_ch, n)`` de EEG em µV.

    ``apply_eog`` sobrepõe-se à config — é o que permite correr a segunda
    passagem de controlo e comparar as conclusões com e sem regressão.
    """
    pp = cfg.preprocess
    sfreq = cfg.device.sfreq_nominal
    raw_eeg = np.asarray(raw_eeg, dtype=np.float64)

    # Saturação primeiro, antes de qualquer filtro: ver find_saturation.
    saturation = find_saturation(
        raw_eeg, pp.saturation_uv, pp.saturation_pad_s, sfreq
    )
    saturated_fraction = float(saturation.mean())
    if saturated_fraction > 0:
        raw_eeg = excise(raw_eeg, saturation)

    detrended = detrend(raw_eeg, pp.detrend)
    notched = notch(detrended, sfreq, pp.notch_freqs_hz, pp.notch_q, pp.notch_passes)
    filtered = bandpass(notched, sfreq, pp.bandpass_hz, pp.bandpass_order)

    use_eog = pp.eog_regression.enabled if apply_eog is None else apply_eog
    eog_report: dict[str, Any] = {}
    if use_eog:
        montage = cfg.montage
        target_idx = [montage.index_of(n) for n in montage.metric_channels]
        eog_idx = [montage.index_of(n) for n in montage.eog_channels]
        before = filtered
        filtered, eog_report = regress_eog(
            filtered, sfreq, target_idx, eog_idx, pp.eog_regression
        )
        eog_report["variance_removed"] = _variance_removed_report(
            before, filtered, sfreq, cfg, target_idx
        )

    edge = int(round(pp.filter_edge_s * sfreq))
    if filtered.shape[1] <= 2 * edge:
        raise ValueError(
            f"registo demasiado curto ({filtered.shape[1] / sfreq:.1f} s) para "
            f"descartar {pp.filter_edge_s:.1f} s de margem em cada ponta"
        )
    span = slice(edge, filtered.shape[1] - edge) if edge else slice(None)

    return Preprocessed(
        saturation=np.ascontiguousarray(saturation[:, span]),
        saturated_fraction=saturated_fraction,
        clean=np.ascontiguousarray(filtered[:, span]),
        detrended=np.ascontiguousarray(detrended[:, span]),
        notched=np.ascontiguousarray(notched[:, span]),
        sfreq=sfreq,
        edge_samples=edge,
        t0_s=t0_s + edge / sfreq,
        eog_applied=bool(use_eog),
        eog_report=eog_report,
    )


def _variance_removed_report(
    before: np.ndarray,
    after: np.ndarray,
    sfreq: float,
    cfg: Config,
    target_idx: list[int],
) -> dict[str, dict[str, float]]:
    """Fração de variância que a regressão removeu, por canal e por banda.

    É esta tabela que diz se o ``theta_rel`` ficou comprometido nesta sessão: a
    banda do regressor (1–5 Hz) sobrepõe-se ao teta (4–7 Hz), e o teta frontal
    medial é largamente partilhado entre Fp e F.
    """
    names = {v: k for k, v in cfg.montage.name_to_index.items()}
    report: dict[str, dict[str, float]] = {}
    for band_name in ("theta", "alpha", "beta", "smr"):
        band = cfg.bands[band_name]
        v_before = bandlimited_variance(before[target_idx], sfreq, band)
        v_after = bandlimited_variance(after[target_idx], sfreq, band)
        with np.errstate(divide="ignore", invalid="ignore"):
            fraction = np.where(v_before > 0, 1.0 - v_after / v_before, np.nan)
        report[band_name] = {
            names[idx]: float(f) for idx, f in zip(target_idx, fraction)
        }
    return report
