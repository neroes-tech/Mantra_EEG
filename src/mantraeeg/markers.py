"""Registo de biomarcadores e as funções que os calculam.

**Fonte única** (SPEC 4.1). O relatório, a tabela de diagnóstico, a seleção no
painel do investigador e a agregação entre sessões iteram este registo.
Acrescentar um marcador é acrescentar uma entrada — não editar cinco ficheiros.

As funções são **puras**, com as assinaturas de ``BIOMARKERS.md``: recebem
dados e um objeto de bandas, devolvem um ``float`` ou ``nan``. Sem estado, sem
I/O, sem leitura de config global.

Onde os dados não chegam, devolvem ``nan`` e propagam. Nunca se inventa valor,
nunca se interpola sobre épocas rejeitadas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Literal, Mapping

import numpy as np

from .bands import Bands
from .connectivity import (
    coherence_band,
    debiased_wpli_band,
    imaginary_coherence_band,
)
from .spectral import band_power, periodic_power, relative_power, spectral_slope

Kind = Literal["psd", "pair", "timeseries"]
Direction = Literal["up", "down", "unknown"]
Role = Literal["marker", "extra"]


@dataclass(frozen=True)
class Marker:
    """Uma entrada do registo."""

    id: str
    label: str                       # rótulo em português, para o gráfico
    kind: Kind
    channels: tuple[str, ...]
    fn: Callable[..., float]
    expected_direction: Direction
    in_composite: bool
    notes: str                       # cautelas, para o relatório de investigação
    #: Uma frase para o participante: o que a medida capta, em linguagem
    #: corrente. Vive aqui e não no gerador de relatório para o registo se
    #: manter fonte única — acrescentar um marcador traz o texto com ele.
    explanation: str = ""
    #: ``extra`` é calculado e reportado mas não entra no ranking (SPEC 4).
    role: Role = "marker"
    #: Rótulo amigável para o participante, quando o marcador é apresentado.
    friendly: str = ""
    unit: str = ""
    params: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_connectivity(self) -> bool:
        return self.kind == "pair"


# --------------------------------------------------------------------------- #
# Funções de marcador — kind="psd"
# --------------------------------------------------------------------------- #
def _mean_relative(psd, freqs, chans, bands, names, band_name) -> float:
    idx = [chans[n] for n in names if n in chans]
    if not idx:
        return float("nan")
    values = relative_power(psd[idx], freqs, bands[band_name], bands.total)
    return float(np.mean(values)) if np.all(np.isfinite(values)) else float("nan")


def alpha_rel(psd, freqs, chans, bands) -> float:
    """Alfa relativo frontocentral. Sobe com relaxamento (B1)."""
    return _mean_relative(psd, freqs, chans, bands, ("F3", "F4", "C3", "C4"), "alpha")


def beta_rel(psd, freqs, chans, bands) -> float:
    """Beta relativo frontal. Desce com relaxamento (B2)."""
    return _mean_relative(psd, freqs, chans, bands, ("F3", "F4"), "beta")


def smr_rel(psd, freqs, chans, bands) -> float:
    """SMR relativo central. Sobe com calma alerta e quietude motora (B4)."""
    return _mean_relative(psd, freqs, chans, bands, ("C3", "C4"), "smr")


def theta_rel(psd, freqs, chans, bands) -> float:
    """Teta frontal. Atenção internalizada — direção contestada (B5)."""
    return _mean_relative(psd, freqs, chans, bands, ("F3", "F4"), "theta")


def alpha_beta(psd, freqs, chans, bands) -> float:
    """Razão alfa/beta. Sobe com relaxamento (B3)."""
    numerator = alpha_rel(psd, freqs, chans, bands)
    denominator = beta_rel(psd, freqs, chans, bands)
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def faa(psd, freqs, chans, bands) -> float:
    """ALAY: ``ln P_abs_alpha[F4] - ln P_abs_alpha[F3]`` (B6).

    Única métrica em potência absoluta — é um log-ratio, a normalização
    cancela. Convenção de sinal confirmada contra o código interno da Neroes
    (``F4Alpha - F3Alpha``): o sentido coincide, não há inversão.
    """
    if "F3" not in chans or "F4" not in chans:
        return float("nan")
    left = band_power(psd[[chans["F3"]]], freqs, bands["alpha"])[0]
    right = band_power(psd[[chans["F4"]]], freqs, bands["alpha"])[0]
    if left <= 0 or right <= 0:
        return float("nan")
    return float(np.log(right) - np.log(left))


def ap_exponent(psd, freqs, chans, bands, fit_range=(2.0, 40.0)) -> float:
    """Expoente aperiódico, média dos canais de métrica (B7).

    Estimador rápido em log-log: o specparam completo custa dezenas de ms por
    ajuste e não cabe no orçamento de 10 s quando corre por época.
    """
    idx = [chans[n] for n in ("F3", "F4", "C3", "C4") if n in chans]
    if not idx:
        return float("nan")
    slopes = spectral_slope(
        psd[idx], freqs, fit_range=fit_range, peak_mask_bands=(bands["alpha"],)
    )
    return float(-np.nanmean(slopes)) if np.any(np.isfinite(slopes)) else float("nan")


def alpha_peak_power(psd, freqs, chans, bands) -> float:
    """Altura do pico alfa acima do 1/f (extra, ``corrected_band_power``).

    Imune ao denominador da potência relativa e a deslocamentos de banda larga.
    Medido nos dados desta montagem, é o que distingue "o alfa subiu" de "o
    resto desceu".
    """
    idx = [chans[n] for n in ("F3", "F4", "C3", "C4") if n in chans]
    if not idx:
        return float("nan")
    values = [periodic_power(psd[i], freqs, bands["alpha"]) for i in idx]
    return float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")


def spectral_entropy(psd, freqs, chans, bands) -> float:
    """Entropia espectral normalizada em 1–45 Hz (extra, B8)."""
    idx = [chans[n] for n in ("F3", "F4", "C3", "C4") if n in chans]
    if not idx:
        return float("nan")
    lo, hi = bands.total
    mask = (freqs >= lo) & (freqs <= hi)
    out = []
    for i in idx:
        power = np.asarray(psd[i])[mask]
        total = power.sum()
        if total <= 0:
            continue
        p = power / total
        p = p[p > 0]
        out.append(float(-(p * np.log(p)).sum() / np.log(len(p))))
    return float(np.mean(out)) if out else float("nan")


# --------------------------------------------------------------------------- #
# kind="pair" — recebem o sinal no tempo
# --------------------------------------------------------------------------- #
def _pair(fn, x, sfreq, chans, band, min_segments):
    if "F3" not in chans or "F4" not in chans:
        return float("nan")
    return fn(
        np.asarray(x[chans["F3"]], dtype=np.float64),
        np.asarray(x[chans["F4"]], dtype=np.float64),
        sfreq,
        band,
        min_segments=min_segments,
    )


def coh_alpha1(x, sfreq, chans, bands, min_segments=19) -> float:
    """MSC F3–F4 em alfa1. O marcador mais específico de mantra (B10)."""
    return _pair(coherence_band, x, sfreq, chans, bands["alpha1"], min_segments)


def coh_theta2(x, sfreq, chans, bands, min_segments=19) -> float:
    """MSC F3–F4 em teta2. Atenção a processos mentais internos (B10)."""
    return _pair(coherence_band, x, sfreq, chans, bands["theta2"], min_segments)


def imcoh_alpha1(x, sfreq, chans, bands, min_segments=19) -> float:
    """Coerência imaginária F3–F4 em alfa1. Imune à condução de volume."""
    return _pair(imaginary_coherence_band, x, sfreq, chans, bands["alpha1"], min_segments)


def wpli_alpha1(x, sfreq, chans, bands, min_segments=19) -> float:
    """wPLI² desenviesado F3–F4 em alfa1. Verificação de robustez (B10)."""
    return _pair(debiased_wpli_band, x, sfreq, chans, bands["alpha1"], min_segments)


# --------------------------------------------------------------------------- #
# kind="timeseries"
# --------------------------------------------------------------------------- #
def lziv(x, chans, bands) -> float:
    """Complexidade de Lempel-Ziv, binarização pela mediana, média dos canais.

    Normalizada por ``n / log2(n)``, que é a normalização padrão e a que torna
    os valores comparáveis com a literatura (B8).
    """
    idx = [chans[n] for n in ("F3", "F4", "C3", "C4") if n in chans]
    if not idx:
        return float("nan")
    out = []
    for i in idx:
        signal = np.asarray(x[i], dtype=np.float64)
        bits = (signal > np.median(signal)).astype(np.uint8)
        out.append(_lz76(bits) / (len(bits) / np.log2(len(bits))))
    return float(np.mean(out)) if out else float("nan")


def _lz76(bits: np.ndarray) -> int:
    """Contagem de Lempel-Ziv 1976 sobre uma sequência binária."""
    text = bits.tobytes()
    n = len(text)
    i, k, l, k_max, complexity = 0, 1, 1, 1, 1
    while l + k <= n:
        if text[i + k - 1] == text[l + k - 1]:
            k += 1
        else:
            k_max = max(k, k_max)
            i += 1
            if i == l:
                complexity += 1
                l += k_max
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    return complexity + 1 if k != 1 else complexity


# --------------------------------------------------------------------------- #
# O REGISTO
# --------------------------------------------------------------------------- #
REGISTRY: tuple[Marker, ...] = (
    Marker(
        id="alpha_rel",
        label="Alfa relativo frontocentral",
        friendly="Calma (alfa)",
        kind="psd",
        channels=("F3", "F4", "C3", "C4"),
        fn=alpha_rel,
        expected_direction="up",
        in_composite=True,
        unit="fração de 1–45 Hz",
        notes=(
            "O marcador mais sólido do conjunto. Meta-análise de stress: alfa "
            "desce de forma consistente (g = 0,60). Sobe em todos os estados "
            "meditativos."
        ),
        explanation=(
            "Ritmo que o cérebro produz quando não está a processar o exterior."
        ),
    ),
    Marker(
        id="beta_rel",
        label="Beta relativo frontal",
        friendly="Esforço mental",
        kind="psd",
        channels=("F3", "F4"),
        fn=beta_rel,
        expected_direction="down",
        in_composite=True,
        unit="fração de 1–45 Hz",
        notes=(
            "EMG mandibular contamina 15–30 Hz. Medido nesta montagem: trincar "
            "multiplica a energia por 34x em 20–30 Hz, abaixo da banda onde o "
            "índice de EMG mede. Um r_emg baixo NÃO exonera este marcador."
        ),
        explanation=(
            "Ritmo do pensamento ativo e do esforço. Pode subir se estiveres a "
            "interpretar o significado do mantra em vez de o repetir."
        ),
    ),
    Marker(
        id="alpha_beta",
        label="Razão alfa/beta",
        friendly="Relaxamento",
        kind="psd",
        channels=("F3", "F4", "C3", "C4"),
        fn=alpha_beta,
        expected_direction="up",
        in_composite=False,
        unit="razão",
        notes=(
            "Numerador sobre F3 F4 C3 C4 e denominador sobre F3 F4: os "
            "conjuntos de canais diferem, portanto NÃO é uma razão alfa/beta "
            "dentro do canal e não é comparável aos valores de referência "
            "publicados."
        ),
        explanation=(
            "O resultado da relação entre a calma e o esforço mental: quanta "
            "calma há por cada unidade de esforço. Se as duas subirem ao mesmo "
            "tempo, o relaxamento pode descer, e isso não é um erro de medição."
        ),
    ),
    Marker(
        id="smr_rel",
        label="SMR relativo central",
        friendly="Quietude corporal",
        kind="psd",
        channels=("C3", "C4"),
        fn=smr_rel,
        expected_direction="up",
        in_composite=True,
        unit="fração de 1–45 Hz",
        notes=(
            "Protocolo validado usa exatamente potência relativa de SMR em C3. "
            "Cai com movimento, portanto é também indicador de quietude."
        ),
        explanation=(
            "Ritmo das áreas motoras, que sobe quando o corpo está parado e a "
            "mente alerta."
        ),
    ),
    Marker(
        id="theta_rel",
        label="Teta frontal",
        friendly="Foco interior",
        kind="psd",
        channels=("F3", "F4"),
        fn=theta_rel,
        expected_direction="unknown",
        in_composite=False,
        unit="fração de 1–45 Hz",
        notes=(
            "Direção contestada, por isso fora do índice. Além disso é o "
            "marcador em risco pela regressão de EOG: a banda do regressor "
            "(1–5 Hz) sobrepõe-se ao teta (4–7 Hz)."
        ),
        explanation=(
            "Ritmo associado a atenção virada para dentro em vez de para o "
            "exterior. A direção deste é debatida na literatura: há trabalho que "
            "o associa a bem-estar profundo e há trabalho que o relaciona "
            "inversamente com a profundidade da meditação."
        ),
    ),
    Marker(
        id="faa",
        label="Assimetria alfa frontal (ALAY)",
        friendly="Controlo emocional",
        kind="psd",
        channels=("F3", "F4"),
        fn=faa,
        expected_direction="unknown",
        in_composite=False,
        unit="ln(µV²) F4 − F3",
        notes=(
            "NÃO usar como índice de stress: efeito agregado nulo em "
            "meta-análise (g = 0,01). Único marcador em potência absoluta, "
            "logo o único exposto a deriva diferencial de impedância. Em 4 min "
            "só a variação intra-sujeito é interpretável."
        ),
        explanation=(
            "A atividade frontal inclina-se para um dos lados conforme o estado "
            "emocional. Durante o mantra tende para o lado que a investigação "
            "associa a aproximação e a afeto positivo."
        ),
    ),
    Marker(
        id="coh_alpha1",
        label="Coerência alfa1 F3–F4",
        friendly="Sintonia com o mantra",
        kind="pair",
        channels=("F3", "F4"),
        fn=coh_alpha1,
        expected_direction="up",
        in_composite=False,
        unit="MSC 0–1",
        notes=(
            "O marcador com maior especificidade para mantra: a literatura de "
            "MT convergiu na coerência, não na potência. Referência comum e "
            "condução de volume inflacionam-na — ler junto com imcoh e wpli."
        ),
        explanation=(
            "O quanto os dois lados do cérebro oscilam em conjunto. É o marcador "
            "que a investigação sobre meditação com mantra identificou como o "
            "mais sensível — em prática com mantra a potência das ondas muda "
            "pouco, o que muda é esta sintonia."
        ),
    ),
    Marker(
        id="coh_theta2",
        label="Coerência teta2 F3–F4",
        friendly="Sintonia meditativa",
        kind="pair",
        channels=("F3", "F4"),
        fn=coh_theta2,
        expected_direction="up",
        in_composite=False,
        unit="MSC 0–1",
        notes="Sobe na escuta de recitação védica. Mesmas cautelas da coerência alfa1.",
        explanation=(
            "A mesma sintonia, mas no ritmo da atenção virada para dentro. Sobe "
            "em quem ouve recitação em sânscrito."
        ),
    ),
    Marker(
        id="imcoh_alpha1",
        label="Coerência imaginária alfa1",
        friendly="Sintonia — verificação sem contaminação",
        kind="pair",
        channels=("F3", "F4"),
        fn=imcoh_alpha1,
        expected_direction="up",
        role="extra",
        in_composite=False,
        unit="0–1",
        notes=(
            "Insensível por construção à condução de volume e à referência "
            "comum. Se a MSC sobe e esta não, o aumento pode ser artefacto — "
            "ou sincronia genuína a lag zero, e as duas não se distinguem."
        ),
        explanation=(
            "Confirma que a sintonia é real e não um efeito da forma como os "
            "elétrodos estão ligados."
        ),
    ),
    Marker(
        id="wpli_alpha1",
        label="wPLI alfa1 (desenviesado)",
        friendly="Sintonia — verificação por desfasamento",
        kind="pair",
        channels=("F3", "F4"),
        fn=wpli_alpha1,
        expected_direction="unknown",
        role="extra",
        in_composite=False,
        unit="0–1",
        notes=(
            "Cego a acoplamento a lag zero por construção — é isso que o torna "
            "verificação e não substituto. Um wPLI nulo não refuta um efeito."
        ),
        explanation=(
            "Segunda confirmação da sintonia, por um método diferente."
        ),
    ),
    Marker(
        id="ap_exponent",
        label="Expoente aperiódico",
        friendly="Perfil global da atividade",
        kind="psd",
        channels=("F3", "F4", "C3", "C4"),
        fn=ap_exponent,
        expected_direction="unknown",
        role="extra",
        in_composite=False,
        unit="expoente 1/f",
        notes=(
            "Não afirmar direção: a validação farmacológica concluiu que não é "
            "marcador fiável do rácio excitação/inibição. Com a regressão de "
            "EOG ligada, o ajuste começa acima da banda do regressor."
        ),
        params={"fit_range": (3.0, 40.0)},
        explanation=(
            "A inclinação geral do espectro. Muda entre estados de consciência, "
            "mas a investigação ainda não estabeleceu qual a direção desejável."
        ),
    ),
    Marker(
        id="lziv",
        label="Complexidade de Lempel-Ziv",
        friendly="Variedade da atividade",
        kind="timeseries",
        channels=("F3", "F4", "C3", "C4"),
        fn=lziv,
        expected_direction="unknown",
        role="extra",
        in_composite=False,
        unit="normalizada",
        notes="Largamente redundante com o expoente aperiódico. Exploratório.",
        explanation=(
            "O quanto o sinal é variado em vez de repetitivo. Sem direção "
            "estabelecida."
        ),
    ),
    # -- extras: reportados, não rankeados --------------------------------- #
    Marker(
        id="alpha_peak_power",
        label="Pico alfa acima do 1/f",
        friendly="Alfa oscilatório puro",
        kind="psd",
        channels=("F3", "F4", "C3", "C4"),
        fn=alpha_peak_power,
        expected_direction="up",
        in_composite=False,
        role="extra",
        unit="µV²",
        notes=(
            "Isola a oscilação do deslocamento de banda larga. Imune ao "
            "denominador da potência relativa, que encolhe de olhos fechados "
            "por haver menos pestanejos."
        ),
        explanation=(
            "A parte das ondas de calma que é oscilação verdadeira, separada do "
            "fundo do sinal."
        ),
    ),
    Marker(
        id="spectral_entropy",
        label="Entropia espectral",
        friendly="Dispersão espectral",
        kind="psd",
        channels=("F3", "F4", "C3", "C4"),
        fn=spectral_entropy,
        expected_direction="unknown",
        in_composite=False,
        role="extra",
        unit="0–1",
        notes="Exploratório, redundante com a complexidade.",
        explanation=(
            "O quanto a energia está espalhada por várias frequências em vez de "
            "concentrada numa."
        ),
    ),
)

BY_ID: dict[str, Marker] = {m.id: m for m in REGISTRY}


def markers(role: Role | None = None, kind: Kind | None = None) -> Iterator[Marker]:
    """Itera o registo, opcionalmente filtrado. É por aqui que tudo passa."""
    for marker in REGISTRY:
        if role is not None and marker.role != role:
            continue
        if kind is not None and marker.kind != kind:
            continue
        yield marker


def get(marker_id: str) -> Marker:
    try:
        return BY_ID[marker_id]
    except KeyError:
        raise KeyError(
            f"marcador {marker_id!r} não está no registo; disponíveis: "
            f"{sorted(BY_ID)}"
        ) from None
