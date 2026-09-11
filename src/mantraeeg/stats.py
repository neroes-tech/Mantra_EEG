"""Estatística robusta para séries de épocas sobrepostas (SPEC 5.1).

As épocas sobrepõem-se 50 % e a envolvente de potência alfa tem dependência de
longo alcance. Tratá-las como independentes produz intervalos de confiança
demasiado estreitos e, por consequência, deteções que não existem.

Isto não é teórico. Num controlo negativo — dois segmentos com exatamente a
mesma distribuição — o delta de Cliff sozinho declarava "efeito grande" em 3 de
8 canais. O bootstrap por blocos com o comprimento estimado dos dados elimina
esses falsos positivos.

O comprimento de bloco é **estimado da autocorrelação**, não fixado: um marcador
com tempo de descorrelação de 60 s numa gravação de 240 s tem cerca de quatro
amostras independentes e não tem direito a um IC apertado.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Estimate:
    """Uma estimativa com intervalo de confiança por bootstrap."""

    value: float
    ci_low: float
    ci_high: float
    block_len: int
    n: int

    @property
    def n_effective(self) -> float:
        """Amostras independentes, aproximadamente ``n / comprimento do bloco``.

        Com poucas, nenhum IC é de confiar: a decisão certa é recusar-se a
        concluir, não produzir um intervalo apertado a partir de nada.
        """
        return self.n / self.block_len if self.block_len else float("nan")

    @property
    def crosses_zero(self) -> bool:
        return bool(self.ci_low <= 0.0 <= self.ci_high)

    @property
    def significant(self) -> bool:
        """O IC exclui zero. Nunca "melhorou" quando isto é falso (SPEC 5.1)."""
        return not self.crosses_zero and np.isfinite(self.ci_low)


# --------------------------------------------------------------------------- #
# Dimensão de efeito
# --------------------------------------------------------------------------- #
def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """``P(b > a) - P(a > b)``. Robusto, não paramétrico, invariante a monótonas.

    Por ser invariante a qualquer reescalonamento monótono, normalizar os dados
    antes de o calcular não faz absolutamente nada — a inferência corre em
    unidades cruas (SPEC 5.1).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan")
    greater = int((b[:, None] > a[None, :]).sum())
    less = int((b[:, None] < a[None, :]).sum())
    return (greater - less) / (a.size * b.size)


def median_log_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """``log2(mediana(b) / mediana(a))``. Simétrico em torno de zero."""
    med_a, med_b = float(np.median(a)), float(np.median(b))
    if med_a <= 0 or med_b <= 0:
        return float("nan")
    return float(np.log2(med_b / med_a))


def interpret_delta(delta: float) -> str:
    """Rótulos convencionais para o delta de Cliff."""
    magnitude = abs(delta)
    if not np.isfinite(magnitude):
        return "indeterminado"
    if magnitude < 0.147:
        return "desprezável"
    if magnitude < 0.33:
        return "pequeno"
    if magnitude < 0.474:
        return "médio"
    return "grande"


# --------------------------------------------------------------------------- #
# Comprimento de bloco
# --------------------------------------------------------------------------- #
def decorrelation_lag(x: np.ndarray, threshold: float = 0.3679) -> int:
    """Primeiro lag em que a autocorrelação cai abaixo de ``threshold`` (1/e)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4:
        return 1
    centred = x - x.mean()
    denom = float(np.dot(centred, centred))
    if denom <= 0:
        return 1
    acf = np.correlate(centred, centred, mode="full")[n - 1 :] / denom
    below = np.nonzero(acf < threshold)[0]
    return int(below[0]) if below.size else n // 2


def auto_block_length(
    x: np.ndarray,
    hop_s: float,
    min_length_s: float,
    max_length_s: float,
    multiplier: float = 2.0,
    threshold: float = 0.3679,
) -> int:
    """Comprimento de bloco em épocas, estimado da autocorrelação e limitado.

    O piso existe porque, com 50 % de sobreposição, épocas adjacentes partilham
    metade das amostras: nenhum bloco honesto pode ser mais curto do que isso.
    """
    lag = decorrelation_lag(x, threshold)
    epochs = int(np.ceil(multiplier * lag))
    lo = max(int(np.ceil(min_length_s / hop_s)), 2)
    hi = max(int(np.floor(max_length_s / hop_s)), lo)
    # Não se limita o bloco em função de ``n``. Um bloco comparável ao
    # comprimento da série torna o bootstrap degenerado — as reamostras ficam
    # quase iguais à original e o IC sai absurdamente estreito — mas a resposta
    # a isso é **denunciar**, não encolher o bloco: limitá-lo a ``n/4`` fixaria
    # ``n_effective`` em 4 por construção e tornaria o critério inútil.
    # Quem denuncia é :attr:`Estimate.n_effective`.
    return int(np.clip(epochs, lo, hi))


# --------------------------------------------------------------------------- #
# Bootstrap por blocos
# --------------------------------------------------------------------------- #
def _stationary_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Índices de uma reamostra pelo bootstrap estacionário (Politis-Romano).

    Blocos de comprimento geométrico, e não fixo: evita os artefactos das
    fronteiras rígidas e mantém a série reamostrada estacionária.
    """
    if n <= 0:
        return np.zeros(0, dtype=int)
    p = 1.0 / max(block_len, 1)
    out = np.empty(n, dtype=int)
    i = 0
    while i < n:
        out[i] = rng.integers(n)
        i += 1
        while i < n and rng.random() >= p:
            out[i] = (out[i - 1] + 1) % n
            i += 1
    return out


def block_bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    statistic,
    hop_s: float,
    min_length_s: float,
    max_length_s: float,
    n_boot: int = 2000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """IC por bootstrap estacionário por blocos, para duas séries de épocas.

    Cada réplica reamostra **as duas** séries por blocos, preservando a
    dependência temporal que resta depois da sobreposição de 50 %.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 3 or b.size < 3:
        return Estimate(float("nan"), float("nan"), float("nan"), 0, min(a.size, b.size))

    block = max(
        auto_block_length(a, hop_s, min_length_s, max_length_s),
        auto_block_length(b, hop_s, min_length_s, max_length_s),
    )
    rng = np.random.default_rng(seed)
    replicates = np.empty(n_boot)
    for i in range(n_boot):
        ra = a[_stationary_indices(a.size, block, rng)]
        rb = b[_stationary_indices(b.size, block, rng)]
        replicates[i] = statistic(ra, rb)

    finite = replicates[np.isfinite(replicates)]
    if finite.size < n_boot // 10:
        return Estimate(
            statistic(a, b), float("nan"), float("nan"), block, min(a.size, b.size)
        )
    alpha = (1.0 - ci_level) / 2.0
    lo, hi = np.quantile(finite, [alpha, 1.0 - alpha])
    return Estimate(
        value=float(statistic(a, b)),
        ci_low=float(lo),
        ci_high=float(hi),
        block_len=block,
        n=min(a.size, b.size),
    )
