"""Bandas de frequência.

Este módulo define a *estrutura* das bandas. Os valores vêm do YAML — não há
nenhuma frequência escrita aqui. As funções de marcador de ``markers.py``
recebem uma instância de :class:`Bands` e nunca leem config global.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping

Band = tuple[float, float]

#: Bandas que têm de existir na config para o pipeline arrancar.
REQUIRED_BANDS: tuple[str, ...] = (
    "theta",
    "theta2",
    "alpha",
    "alpha1",
    "alpha2",
    "smr",
    "beta",
    "total",
    "emg",
    "emg_wide",
    "line",
)


@dataclass(frozen=True)
class Bands:
    """Conjunto imutável de bandas, indexável por nome.

    Passado explicitamente a todas as funções puras de marcador, conforme o
    contrato de ``BIOMARKERS.md``.
    """

    values: Mapping[str, Band]

    def __getitem__(self, name: str) -> Band:
        try:
            return self.values[name]
        except KeyError:
            raise KeyError(
                f"banda {name!r} não definida; disponíveis: {sorted(self.values)}"
            ) from None

    def __contains__(self, name: str) -> bool:
        return name in self.values

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    @property
    def total(self) -> Band:
        """Denominador da potência relativa (1–45 Hz por defeito)."""
        return self["total"]

    @classmethod
    def from_config(cls, raw: Mapping[str, object]) -> "Bands":
        """Constrói a partir do bloco ``bands`` do YAML, validando cada entrada."""
        parsed: dict[str, Band] = {}
        for name, value in raw.items():
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(
                    f"banda {name!r}: esperado par [lo, hi], obtido {value!r}"
                )
            lo, hi = float(value[0]), float(value[1])
            if not lo < hi:
                raise ValueError(f"banda {name!r}: lo ({lo}) tem de ser < hi ({hi})")
            if lo < 0.0:
                raise ValueError(f"banda {name!r}: limite inferior negativo ({lo})")
            parsed[name] = (lo, hi)

        missing = [b for b in REQUIRED_BANDS if b not in parsed]
        if missing:
            raise ValueError(f"bandas em falta na config: {missing}")

        # BIOMARKERS.md: beta começa em 15 Hz para não sobrepor ao SMR.
        smr_hi = parsed["smr"][1]
        beta_lo = parsed["beta"][0]
        if beta_lo < smr_hi:
            raise ValueError(
                f"beta começa em {beta_lo} Hz e sobrepõe-se ao SMR (termina em "
                f"{smr_hi} Hz); BIOMARKERS.md exige separação"
            )

        # alpha1 e alpha2 têm de caber dentro de alpha.
        a_lo, a_hi = parsed["alpha"]
        for sub in ("alpha1", "alpha2"):
            s_lo, s_hi = parsed[sub]
            if s_lo < a_lo or s_hi > a_hi:
                raise ValueError(
                    f"{sub} ({s_lo}–{s_hi}) tem de estar contida em alpha ({a_lo}–{a_hi})"
                )

        return cls(values=dict(parsed))
