"""Aquisição via BrainFlow (``BoardIds.UNICORN_BOARD``). Primeira escolha (SPEC 1.3).

Verificado empiricamente contra hardware real: o BrainFlow abre o Unicorn e
auto-deteta o dispositivo emparelhado sem serial. Traz a sua própria
``Unicorn.dll`` (distinta da do Suite), pelo que não depende do caminho do Suite.

Dois detalhes que só se veem com o dispositivo ligado e que esta camada trata:

1. O BrainFlow apresenta **19 linhas**, não as 17 do dispositivo — acrescenta
   ``package_num``, ``timestamp`` e ``marker``. Reduzimos ao layout canónico.
2. ``get_eeg_names()`` devolve a montagem **default** do Unicorn
   (``Fz, C3, Cz, C4, Pz, PO7, Oz, PO8``), que não é a nossa. Guardamos só para
   diagnóstico; a montagem vem exclusivamente da config.

Os timestamps do BrainFlow chegam em rajadas de pacotes Bluetooth (mediana
medida de 2,5 ms entre amostras, p99 de 27 ms, máximo de 64 ms, com uma taxa
global de 250,5 Hz). Não servem de base temporal — servem para medir a taxa real
e detetar buracos. A base temporal é o contador do dispositivo.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import (
    ACCEL_SLICE,
    BATTERY_ROW,
    COUNTER_ROW,
    EEG_SLICE,
    GYRO_SLICE,
    N_CANONICAL_CHANNELS,
    VALIDATION_ROW,
    DeviceInfo,
    EegSource,
    SourceError,
    empty_chunk,
)


def _discover_serial() -> str:
    """Numero de serie real do headset, pela lista de dispositivos da DLL.

    O BrainFlow auto-deteta o dispositivo mas nao diz qual: sem serial na
    config, o device_id ficava "(auto-detetado)" e a pasta da sessao saia
    chamada "detetado)". O serial e o que nomeia a pasta (UN-2022.01.16 -> 16),
    por isso vale a pena ir busca-lo.

    Se nao der, devolve string vazia — o resto do sistema trata disso.
    """
    try:
        from ..config import load_config
        from .unicorn_dll import list_available_serials

        serials = list_available_serials()
        return serials[0] if serials else ""
    except Exception:
        return ""


def _open_device_hints(exc: Exception) -> str:
    """Sugestões acionáveis para uma falha de abertura, na banca e às 3 da manhã.

    O erro 4 de ``UNICORN_OpenDevice`` é o mais comum e quase sempre significa
    que o headset se desligou sozinho por inatividade — aparece emparelhado no
    Windows mesmo depois de desligado, portanto a lista de dispositivos não
    denuncia nada.
    """
    text = str(exc)
    hints = [
        "  - o headset está LIGADO? (auto-desliga por inatividade e continua a",
        "    aparecer emparelhado no Windows depois de desligado)",
        "  - o dongle Bluetooth da g.tec está espetado?",
        "  - outra aplicação (Unicorn Suite, UnicornLSL, outra sessão desta app)",
        "    tem o dispositivo aberto?",
    ]
    if "UNICORN_OpenDevice 4" in text or "ANOMALY" in text.upper():
        hints.insert(
            0,
            "  - erro 4 = não foi possível abrir. Desligar e voltar a ligar o\n"
            "    headset resolve na esmagadora maioria dos casos.",
        )
    return "\n".join(hints)


class BrainflowSource(EegSource):
    """Unicorn Hybrid Black através do BrainFlow."""

    def __init__(self, cfg: Mapping[str, Any], sfreq_nominal: float) -> None:
        super().__init__()
        self._cfg = dict(cfg)
        self._sfreq_nominal = float(sfreq_nominal)
        self._board: Any = None
        self._rows: dict[str, Any] = {}
        self._last_timestamps: np.ndarray = np.zeros(0, dtype=np.float64)

    # -- ciclo de vida ------------------------------------------------------- #
    def open(self) -> DeviceInfo:
        try:
            from brainflow.board_shim import (
                BoardIds,
                BoardShim,
                BrainFlowInputParams,
            )
        except ImportError as exc:  # pragma: no cover - ambiente sem brainflow
            raise SourceError(
                "brainflow não está instalado. `pip install brainflow`. "
                "Alternativas: device.source = unicorn_dll ou lsl."
            ) from exc

        board_id = BoardIds.UNICORN_BOARD

        # SPEC 1.1: ler os índices das constantes da API, nunca hardcoded.
        try:
            self._rows = {
                "eeg": list(BoardShim.get_eeg_channels(board_id)),
                "accel": list(BoardShim.get_accel_channels(board_id)),
                "gyro": list(BoardShim.get_gyro_channels(board_id)),
                "battery": BoardShim.get_battery_channel(board_id),
                "counter": BoardShim.get_package_num_channel(board_id),
                "other": list(BoardShim.get_other_channels(board_id)),
                "timestamp": BoardShim.get_timestamp_channel(board_id),
            }
            sfreq = float(BoardShim.get_sampling_rate(board_id))
            backend_names = tuple(BoardShim.get_eeg_names(board_id))
        except Exception as exc:
            raise SourceError(
                f"não foi possível ler a descrição do UNICORN_BOARD do BrainFlow: {exc}"
            ) from exc

        n_eeg = len(self._rows["eeg"])
        if n_eeg != 8:
            raise SourceError(
                f"o BrainFlow reporta {n_eeg} canais de EEG para o Unicorn, "
                f"esperados 8; a versão do BrainFlow pode ser incompatível"
            )
        if not self._rows["other"]:
            raise SourceError(
                "o BrainFlow não expõe o canal 'other' (indicador de validação) "
                "para este board"
            )
        if abs(sfreq - self._sfreq_nominal) > 1e-6:
            raise SourceError(
                f"o dispositivo reporta {sfreq} Hz mas a config diz "
                f"{self._sfreq_nominal} Hz; corrigir device.sfreq_nominal"
            )

        params = BrainFlowInputParams()
        serial = str(self._cfg.get("serial_number", "") or "")
        if serial:
            params.serial_number = serial
        timeout = int(self._cfg.get("timeout_s", 0) or 0)
        if timeout:
            params.timeout = timeout

        self._board = BoardShim(board_id, params)
        try:
            self._board.prepare_session()
        except Exception as exc:
            self._board = None
            raise SourceError(
                f"prepare_session() falhou: {exc}\n"
                f"{_open_device_hints(exc)}"
            ) from exc

        device_id = serial or _discover_serial()

        self._info = DeviceInfo(
            source_name="brainflow",
            sfreq_nominal=sfreq,
            n_channels=N_CANONICAL_CHANNELS,
            device_id=device_id,
            units_to_uv=1.0,  # o BrainFlow entrega µV para o Unicorn
            backend_channel_names=backend_names,
            extra={
                "brainflow_rows": dict(self._rows),
                "n_backend_rows": int(BoardShim.get_num_rows(board_id)),
                # Registado explicitamente para que fique claro no raw_meta.json
                # que estes nomes NÃO foram usados para a montagem.
                "backend_names_unused": True,
            },
        )
        return self._info

    def start(self) -> None:
        if self._board is None:
            raise SourceError("start() antes de open()")
        try:
            self._board.start_stream()
        except Exception as exc:
            raise SourceError(f"start_stream() falhou: {exc}") from exc
        self._started = True

    def _read_raw(self) -> np.ndarray:
        if self._board is None or not self._started:
            return empty_chunk()
        try:
            data = self._board.get_board_data()
        except Exception as exc:
            raise SourceError(f"get_board_data() falhou: {exc}") from exc
        if data.size == 0:
            self._last_timestamps = np.zeros(0, dtype=np.float64)
            return empty_chunk()

        k = data.shape[1]
        out = np.zeros((N_CANONICAL_CHANNELS, k), dtype=np.float32)
        out[EEG_SLICE] = data[self._rows["eeg"]]
        out[ACCEL_SLICE] = data[self._rows["accel"]]
        out[GYRO_SLICE] = data[self._rows["gyro"]]
        out[BATTERY_ROW] = data[self._rows["battery"]]
        out[COUNTER_ROW] = data[self._rows["counter"]]
        out[VALIDATION_ROW] = data[self._rows["other"][0]]

        # Guardado para medir a taxa real e o jitter; não é base temporal.
        self._last_timestamps = np.asarray(
            data[self._rows["timestamp"]], dtype=np.float64
        )
        return out

    @property
    def last_timestamps(self) -> np.ndarray:
        """Timestamps do backend do último ``read``, para diagnóstico de jitter."""
        return self._last_timestamps

    def stop(self) -> None:
        if self._board is not None and self._started:
            try:
                self._board.stop_stream()
            except Exception:
                pass
        self._started = False

    def close(self) -> None:
        if self._board is not None:
            try:
                self._board.release_session()
            except Exception:
                pass
        self._board = None
