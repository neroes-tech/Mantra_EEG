"""Contrato comum das fontes de aquisição.

Toda a fonte entrega o **mesmo layout canónico de 17 linhas** (SPEC 1.1), em
**µV** para o EEG, independentemente do que o backend por baixo devolva. Trocar
de fonte é uma linha na config (critério de aceitação 11), e isso só é verdade
se a conversão de layout e de unidades acontecer aqui.

Sobre unidades: uma discrepância silenciosa de escala entre backends quebraria
todos os limiares absolutos (120 µV, 30 µV, 0,5 µV, 75 µV) sem produzir um único
erro. Cada fonte declara ``units_to_uv`` e a base converte; além disso,
:func:`plausibility_warning` verifica em runtime que a amplitude aterra numa
gama fisiologicamente possível.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Layout canónico (SPEC 1.1)
# --------------------------------------------------------------------------- #
# Esta é a nossa estrutura de saída, não a de nenhum backend. Cada fonte mapeia
# o que recebe para aqui — o BrainFlow, por exemplo, apresenta 19 linhas e é
# esta camada que as reduz às 17 do dispositivo.
N_CANONICAL_CHANNELS = 17
EEG_SLICE = slice(0, 8)
ACCEL_SLICE = slice(8, 11)
GYRO_SLICE = slice(11, 14)
BATTERY_ROW = 14
COUNTER_ROW = 15
VALIDATION_ROW = 16

CANONICAL_ROW_NAMES: tuple[str, ...] = (
    "eeg0", "eeg1", "eeg2", "eeg3", "eeg4", "eeg5", "eeg6", "eeg7",
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z",
    "battery", "counter", "validation",
)

#: Gama plausível para o desvio padrão de EEG de escalpe **após detrend**, em µV.
#: Fora disto, ou as unidades estão erradas ou não há contacto.
PLAUSIBLE_EEG_STD_UV = (1.0, 100.0)


class SourceError(RuntimeError):
    """Falha ao abrir, ler ou fechar uma fonte. A mensagem é para o operador."""


@dataclass(frozen=True)
class DeviceInfo:
    """O que a fonte diz sobre si própria, lido do backend e não assumido."""

    source_name: str
    sfreq_nominal: float
    n_channels: int
    device_id: str = ""
    units_to_uv: float = 1.0
    #: Nomes de canal reportados pelo *backend*. Guardados só para diagnóstico —
    #: nunca usados para montagem. O BrainFlow devolve a montagem default do
    #: Unicorn (Fz, C3, Cz, C4, Pz, PO7, Oz, PO8), que não é a nossa.
    backend_channel_names: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


def plausibility_warning(eeg_uv: np.ndarray) -> str | None:
    """Devolve um aviso se a amplitude não parecer EEG em µV, ou ``None``.

    Chamado depois da conversão de unidades. Não levanta: a decisão de continuar
    é do operador, mas o aviso tem de ser alto.
    """
    if eeg_uv.size == 0:
        return None
    detrended = eeg_uv - eeg_uv.mean(axis=1, keepdims=True)
    med_std = float(np.median(detrended.std(axis=1)))
    lo, hi = PLAUSIBLE_EEG_STD_UV
    if med_std < lo:
        return (
            f"desvio padrão mediano de {med_std:.3g} µV, abaixo de {lo} µV: "
            f"elétrodos mortos, ou o fator de unidades está errado"
        )
    if med_std > hi:
        return (
            f"desvio padrão mediano de {med_std:.3g} µV, acima de {hi} µV: "
            f"sem contacto, deriva do amplificador, ou o fator de unidades "
            f"está errado"
        )
    return None


class EegSource(abc.ABC):
    """Fonte de dados de EEG.

    Ciclo de vida: ``open`` -> ``start`` -> ``read``* -> ``stop`` -> ``close``.
    Utilizável como context manager, que garante ``stop``/``close`` mesmo em erro.
    """

    def __init__(self) -> None:
        self._info: DeviceInfo | None = None
        self._started = False

    # -- ciclo de vida ------------------------------------------------------- #
    @abc.abstractmethod
    def open(self) -> DeviceInfo:
        """Liga ao dispositivo e devolve o que ele reporta sobre si."""

    @abc.abstractmethod
    def start(self) -> None:
        """Começa a aquisição."""

    @abc.abstractmethod
    def _read_raw(self) -> np.ndarray:
        """Devolve o que estiver disponível, já no layout canónico e em µV.

        Implementado por cada backend. Pode devolver ``(17, 0)``.
        """

    @abc.abstractmethod
    def stop(self) -> None:
        """Para a aquisição. Idempotente."""

    @abc.abstractmethod
    def close(self) -> None:
        """Liberta o dispositivo. Idempotente."""

    # -- API pública --------------------------------------------------------- #
    @property
    def info(self) -> DeviceInfo:
        if self._info is None:
            raise SourceError("fonte não aberta; chamar open() primeiro")
        return self._info

    @property
    def sfreq(self) -> float:
        return self.info.sfreq_nominal

    @property
    def n_channels(self) -> int:
        return N_CANONICAL_CHANNELS

    def read(self) -> np.ndarray:
        """Amostras disponíveis, ``(17, k)`` float32, EEG em µV."""
        data = self._read_raw()
        if data.ndim != 2 or data.shape[0] != N_CANONICAL_CHANNELS:
            raise SourceError(
                f"{type(self).__name__} devolveu {data.shape}, esperado "
                f"({N_CANONICAL_CHANNELS}, k)"
            )
        return np.ascontiguousarray(data, dtype=np.float32)

    def __enter__(self) -> "EegSource":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            self.stop()
        finally:
            self.close()


def empty_chunk() -> np.ndarray:
    """Chunk canónico vazio, para quando não há dados novos."""
    return np.zeros((N_CANONICAL_CHANNELS, 0), dtype=np.float32)
