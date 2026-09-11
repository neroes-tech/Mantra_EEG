"""Fixtures partilhadas."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mantraeeg.config import Config, load_config  # noqa: E402
from mantraeeg.sources.replay import load_recording  # noqa: E402

#: Excerto de 30 s de uma gravação Unicorn **real** (offsets DC de eletrodo de
#: 185–711 mV, um canal quase morto). É sobre isto que se testa o que precisa de
#: sinal genuíno; a fonte sintética chega no M1.
FIXTURE_CSV = Path(__file__).parent / "fixtures" / "unicorn_sample.csv"


@pytest.fixture(scope="session")
def cfg() -> Config:
    return load_config(ROOT / "config" / "default.yaml")


@pytest.fixture
def cfg_factory(tmp_path: Path):
    """Constrói uma config a partir da default com alterações pontuais.

    ``patch`` é aplicado em profundidade, para um teste poder mexer num único
    limiar sem reescrever o YAML inteiro.
    """
    base = yaml.safe_load(
        (ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )

    def build(patch: dict | None = None) -> Config:
        data = _deep_merge(base, patch or {})
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        return load_config(path)

    return build


def _deep_merge(base: dict, patch: dict) -> dict:
    import copy

    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@pytest.fixture(scope="session")
def real_recording() -> np.ndarray:
    """``(17, n)`` de dados Unicorn reais."""
    if not FIXTURE_CSV.exists():
        pytest.skip(f"fixture em falta: {FIXTURE_CSV}")
    return load_recording(FIXTURE_CSV)


@pytest.fixture
def synthetic_eeg():
    """Gerador determinístico de EEG plausível, para testes que não precisam de hardware.

    Não substitui a fonte sintética do M1 (SPEC 6.2) — é só ruído 1/f com um
    pico alfa, o suficiente para exercitar buffers e limiares.
    """

    def build(
        n_channels: int = 8,
        seconds: float = 10.0,
        sfreq: float = 250.0,
        alpha_uv: float = 8.0,
        noise_uv: float = 5.0,
        dc_offset_uv: float = 200_000.0,
        seed: int = 0,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n = int(seconds * sfreq)
        t = np.arange(n) / sfreq
        out = np.empty((n_channels, n), dtype=np.float32)
        for ch in range(n_channels):
            alpha = alpha_uv * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 2 * np.pi))
            pink = np.cumsum(rng.normal(0, 1, n))
            pink = pink / (pink.std() or 1.0) * noise_uv
            white = rng.normal(0, noise_uv * 0.3, n)
            out[ch] = alpha + pink + white + dc_offset_uv * (1.0 + 0.1 * ch)
        return out

    return build
