"""Estatística: o bootstrap por blocos tem de rejeitar o que não existe.

Estes testes existem porque as primeiras versões falharam. Com épocas
sobrepostas 50 % e uma envolvente de alfa com dependência de longo alcance, o
delta de Cliff sozinho declarava "efeito grande" em 3 de 8 canais num controlo
negativo, e um bootstrap ingénuo piorava para 5 de 8.
"""

from __future__ import annotations

import numpy as np
import pytest

from mantraeeg.stats import (
    auto_block_length,
    block_bootstrap_ci,
    cliffs_delta,
    decorrelation_lag,
    interpret_delta,
    median_log_ratio,
)

HOP_S, MIN_BLOCK_S, MAX_BLOCK_S = 2.0, 20.0, 60.0


def correlated_series(n: int, tau: int, level: float, seed: int) -> np.ndarray:
    """Série positiva com autocorrelação controlada, como a potência de banda."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, n + 10 * tau)
    kernel = np.exp(-np.arange(5 * tau) / tau)
    smooth = np.convolve(noise, kernel / kernel.sum(), mode="same")[:n]
    return level * np.exp(0.3 * smooth / (smooth.std() or 1.0))


def ci(a, b, n_boot: int = 600):
    return block_bootstrap_ci(
        a, b, median_log_ratio, HOP_S, MIN_BLOCK_S, MAX_BLOCK_S, n_boot=n_boot
    )


# --------------------------------------------------------------------------- #
# Dimensão de efeito
# --------------------------------------------------------------------------- #
def test_cliffs_delta_bounds():
    a = np.arange(10.0)
    assert cliffs_delta(a, a + 100) == pytest.approx(1.0)
    assert cliffs_delta(a, a - 100) == pytest.approx(-1.0)
    assert abs(cliffs_delta(a, a)) < 0.15


def test_cliffs_delta_is_invariant_to_monotone_rescaling():
    """Por isso normalizar antes de o calcular não faz nada (SPEC 5.1)."""
    rng = np.random.default_rng(0)
    a, b = rng.lognormal(0, 1, 60), rng.lognormal(0.5, 1, 60)
    assert cliffs_delta(a, b) == pytest.approx(cliffs_delta(a * 7.3, b * 7.3))
    assert cliffs_delta(a, b) == pytest.approx(cliffs_delta(np.log(a), np.log(b)))


def test_median_log_ratio_is_symmetric():
    a, b = np.full(20, 2.0), np.full(20, 4.0)
    assert median_log_ratio(a, b) == pytest.approx(1.0)
    assert median_log_ratio(b, a) == pytest.approx(-1.0)
    assert np.isnan(median_log_ratio(np.zeros(5), b))


def test_interpret_delta_labels():
    assert interpret_delta(0.05) == "desprezável"
    assert interpret_delta(0.5) == "grande"
    assert interpret_delta(float("nan")) == "indeterminado"


# --------------------------------------------------------------------------- #
# Comprimento de bloco
# --------------------------------------------------------------------------- #
def test_decorrelation_lag_grows_with_smoothing():
    fast = correlated_series(600, tau=2, level=1.0, seed=1)
    slow = correlated_series(600, tau=30, level=1.0, seed=1)
    assert decorrelation_lag(fast) < decorrelation_lag(slow)


def test_block_length_respects_the_floor():
    """Com 50 % de sobreposição, épocas vizinhas partilham metade das amostras."""
    white = np.random.default_rng(0).normal(10, 1, 400)
    block = auto_block_length(white, HOP_S, MIN_BLOCK_S, MAX_BLOCK_S)
    assert block >= MIN_BLOCK_S / HOP_S


def test_effective_samples_grow_with_the_recording():
    """Regressão: limitar o bloco a n/4 fixava n_efectivo em 4 por construção.

    Com esse limite, nem um efeito de Berger 4x com 90 s por segmento passava,
    porque o critério ``n_effective >= 5`` era matematicamente inatingível.
    Gravar mais tempo tem de comprar mais amostras independentes.
    """
    short = correlated_series(40, tau=8, level=1.0, seed=3)
    long = correlated_series(400, tau=8, level=1.0, seed=3)
    n_eff_short = short.size / auto_block_length(short, HOP_S, MIN_BLOCK_S, MAX_BLOCK_S)
    n_eff_long = long.size / auto_block_length(long, HOP_S, MIN_BLOCK_S, MAX_BLOCK_S)
    assert n_eff_long > 3 * n_eff_short


# --------------------------------------------------------------------------- #
# Bootstrap: falsos positivos e verdadeiros positivos
# --------------------------------------------------------------------------- #
def test_effective_samples_reflect_the_data_available():
    short = correlated_series(20, tau=6, level=1.0, seed=5)
    long = correlated_series(200, tau=6, level=1.0, seed=6)
    assert ci(short, short).n_effective < ci(long, long).n_effective


BERGER_MIN_LOG2 = 0.5  # o mesmo piso que a config usa: +41 %


def detected(estimate) -> bool:
    """A regra de decisão real: as três condições em conjunto."""
    return (
        estimate.significant
        and estimate.ci_low > 0
        and estimate.n_effective >= 5
        and estimate.value >= BERGER_MIN_LOG2
    )


def test_bootstrap_underestimates_variance_with_dependent_data():
    """Documenta a limitação conhecida, com o número medido.

    Comparado com a dispersão verdadeira entre realizações independentes, o
    bootstrap por blocos devolve ~0,84x o desvio padrão. É um viés conhecido dos
    bootstraps por blocos com dependência, e não desaparece aumentando o bloco:
    medida a cobertura para multiplicadores de 2 a 8, estagna nos 78-81 %.

    É por isso que a decisão não pode assentar só no IC — ver
    :func:`test_null_is_rejected_by_the_full_decision_rule`.
    """
    truth = [
        median_log_ratio(
            correlated_series(120, 6, 0.5, 1000 + s),
            correlated_series(120, 6, 0.5, 5000 + s),
        )
        for s in range(200)
    ]
    a = correlated_series(120, 6, 0.5, 1000)
    b = correlated_series(120, 6, 0.5, 5000)
    estimate = ci(a, b, n_boot=2000)
    bootstrap_sd = (estimate.ci_high - estimate.ci_low) / (2 * 1.96)
    assert 0.6 < bootstrap_sd / np.std(truth) < 1.15


def test_null_is_rejected_by_the_full_decision_rule():
    """O nulo tem de ser rejeitado pela regra completa, em todas as sementes.

    O IC sozinho deixa passar cerca de 3 em 10 (o viés acima). O piso de
    magnitude é o que fecha a porta: no nulo a estatística fica em +-0,25, muito
    abaixo dos 0,5 que um efeito de Berger produz.
    """
    escapes = [
        seed
        for seed in range(10)
        if detected(
            ci(
                correlated_series(120, tau=6, level=0.5, seed=100 + seed),
                correlated_series(120, tau=6, level=0.5, seed=200 + seed),
            )
        )
    ]
    assert not escapes, f"falsos positivos nas sementes {escapes}"


def test_large_real_effect_is_detected_with_enough_data():
    a = correlated_series(120, tau=6, level=0.5, seed=42)
    b = correlated_series(120, tau=6, level=2.0, seed=43)  # 4x
    estimate = ci(a, b)
    assert detected(estimate)
    assert estimate.value == pytest.approx(2.0, abs=0.6)  # log2(4) = 2


def test_same_effect_is_refused_when_the_segment_is_too_short():
    """A resposta honesta a 40 s por segmento é "dados insuficientes".

    O mesmo efeito 4x que é detetado com 120 épocas tem de ser recusado com 16.
    """
    a = correlated_series(16, tau=6, level=0.5, seed=42)
    b = correlated_series(16, tau=6, level=2.0, seed=43)
    estimate = ci(a, b)
    assert estimate.n_effective < 5
    assert not detected(estimate)


def test_degenerate_input_returns_nan_not_a_fake_interval():
    estimate = ci(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
    assert np.isnan(estimate.value)
    assert np.isnan(estimate.ci_low)


def test_bootstrap_is_reproducible():
    a = correlated_series(80, tau=5, level=1.0, seed=9)
    b = correlated_series(80, tau=5, level=1.5, seed=10)
    assert ci(a, b).ci_low == ci(a, b).ci_low
