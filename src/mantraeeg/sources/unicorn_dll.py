"""Aquisição por ``ctypes`` sobre ``Unicorn.dll``. Segunda escolha (SPEC 1.3).

A C API vem incluída no Unicorn Suite e não exige a licença adicional da
``UnicornPy``. Esta implementação segue as assinaturas documentadas de
``unicorn.h``.

**Estado:** o BrainFlow abre o dispositivo neste sistema, portanto este caminho
é um fallback que ainda não foi exercitado contra hardware. As assinaturas estão
conforme a documentação da API mas não foram confirmadas contra um ``unicorn.h``
— o Suite instalado aqui não traz o header. Validar com
``check_unicorn.py --source unicorn_dll`` antes de depender dele no evento.

O dispositivo entrega escans intercalados de ``n_channels`` floats. A ordem dos
canais é a canónica da SPEC 1.1 e é confirmada contra
``UNICORN_GetNumberOfAcquiredChannels``.
"""

from __future__ import annotations

import ctypes
from ctypes import POINTER, byref, c_char, c_char_p, c_float, c_int, c_uint32
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .base import N_CANONICAL_CHANNELS, DeviceInfo, EegSource, SourceError, empty_chunk

UNICORN_ERROR_SUCCESS = 0
UNICORN_SERIAL_LENGTH_MAX = 14
#: Escans pedidos por chamada a ``UNICORN_GetData``. Bloqueia até os ter.
_SCANS_PER_READ = 25


class _DeviceSerial(ctypes.Structure):
    _fields_ = [("serial", c_char * UNICORN_SERIAL_LENGTH_MAX)]


def find_dll(candidates: Iterable[Path]) -> Path:
    """Primeiro candidato existente, ou erro com todos os caminhos tentados."""
    tried: list[str] = []
    for path in candidates:
        tried.append(str(path))
        if path.exists():
            return path
    raise SourceError(
        "Unicorn.dll não encontrada. Caminhos tentados:\n  "
        + "\n  ".join(tried)
        + "\nDefina a variável de ambiente MANTRA_UNICORN_DLL ou instale o "
        "Unicorn Suite."
    )


def list_available_serials(dll_paths: Iterable[Path] | None = None) -> list[str]:
    """Numeros de serie dos headsets emparelhados, sem abrir nenhum.

    Usado pelo caminho do BrainFlow, que auto-deteta o dispositivo mas nao
    revela qual. O serial e o que nomeia a pasta da sessao.
    """
    if dll_paths is None:
        from ..config import load_config

        dll_paths = load_config().device.resolved_dll_paths()
    lib = ctypes.CDLL(str(find_dll(dll_paths)))
    lib.UNICORN_GetAvailableDevices.restype = c_int
    lib.UNICORN_GetAvailableDevices.argtypes = [
        POINTER(_DeviceSerial), POINTER(c_uint32), c_int
    ]
    count = c_uint32(0)
    if lib.UNICORN_GetAvailableDevices(None, byref(count), 1) != UNICORN_ERROR_SUCCESS:
        return []
    if count.value == 0:
        return []
    devices = (_DeviceSerial * count.value)()
    if lib.UNICORN_GetAvailableDevices(devices, byref(count), 1) != UNICORN_ERROR_SUCCESS:
        return []
    return [d.serial.decode("utf-8", "replace").strip(chr(0)) for d in devices]


class UnicornDllSource(EegSource):
    """Unicorn Hybrid Black através da C API do Suite."""

    def __init__(
        self,
        cfg: Mapping[str, Any],
        sfreq_nominal: float,
        dll_paths: Iterable[Path],
    ) -> None:
        super().__init__()
        self._cfg = dict(cfg)
        self._sfreq_nominal = float(sfreq_nominal)
        self._dll_paths = list(dll_paths)
        self._lib: ctypes.CDLL | None = None
        self._handle = ctypes.c_void_p()
        self._n_dev_channels = 0
        self._buf: np.ndarray | None = None

    # -- utilitários --------------------------------------------------------- #
    def _check(self, code: int, what: str) -> None:
        if code == UNICORN_ERROR_SUCCESS:
            return
        text = ""
        if self._lib is not None:
            try:
                raw = self._lib.UNICORN_GetLastErrorText()
                text = ctypes.c_char_p(raw).value.decode("utf-8", "replace")
            except Exception:
                text = ""
        raise SourceError(f"{what} falhou (código {code}){f': {text}' if text else ''}")

    def _bind(self, lib: ctypes.CDLL) -> None:
        """Declara as assinaturas. Sem isto, o ctypes assume int e corrompe pointers."""
        lib.UNICORN_GetApiVersion.restype = c_float
        lib.UNICORN_GetApiVersion.argtypes = []
        lib.UNICORN_GetLastErrorText.restype = c_char_p
        lib.UNICORN_GetLastErrorText.argtypes = []
        lib.UNICORN_GetAvailableDevices.restype = c_int
        lib.UNICORN_GetAvailableDevices.argtypes = [
            POINTER(_DeviceSerial), POINTER(c_uint32), c_int
        ]
        lib.UNICORN_OpenDevice.restype = c_int
        lib.UNICORN_OpenDevice.argtypes = [c_char_p, POINTER(ctypes.c_void_p)]
        lib.UNICORN_CloseDevice.restype = c_int
        lib.UNICORN_CloseDevice.argtypes = [POINTER(ctypes.c_void_p)]
        lib.UNICORN_StartAcquisition.restype = c_int
        lib.UNICORN_StartAcquisition.argtypes = [ctypes.c_void_p, c_int]
        lib.UNICORN_StopAcquisition.restype = c_int
        lib.UNICORN_StopAcquisition.argtypes = [ctypes.c_void_p]
        lib.UNICORN_GetData.restype = c_int
        lib.UNICORN_GetData.argtypes = [
            ctypes.c_void_p, c_uint32, POINTER(c_float), c_uint32
        ]
        lib.UNICORN_GetNumberOfAcquiredChannels.restype = c_int
        lib.UNICORN_GetNumberOfAcquiredChannels.argtypes = [
            ctypes.c_void_p, POINTER(c_uint32)
        ]

    # -- ciclo de vida ------------------------------------------------------- #
    def open(self) -> DeviceInfo:
        dll_path = find_dll(self._dll_paths)
        try:
            lib = ctypes.CDLL(str(dll_path))
        except OSError as exc:
            raise SourceError(
                f"não foi possível carregar {dll_path}: {exc}\n"
                f"  - o Python é x64 e a DLL também?"
            ) from exc
        self._bind(lib)
        self._lib = lib

        count = c_uint32(0)
        self._check(
            lib.UNICORN_GetAvailableDevices(None, byref(count), 1),
            "UNICORN_GetAvailableDevices (contagem)",
        )
        if count.value == 0:
            raise SourceError(
                "nenhum dispositivo Unicorn emparelhado encontrado. "
                "Emparelhe o headset por Bluetooth com o dongle da g.tec."
            )
        devices = (_DeviceSerial * count.value)()
        self._check(
            lib.UNICORN_GetAvailableDevices(devices, byref(count), 1),
            "UNICORN_GetAvailableDevices (lista)",
        )
        serials = [d.serial.decode("utf-8", "replace").strip("\x00") for d in devices]

        wanted = str(self._cfg.get("serial_number", "") or "")
        serial = wanted or serials[0]
        if wanted and wanted not in serials:
            raise SourceError(
                f"dispositivo {wanted!r} não está entre os emparelhados: {serials}"
            )

        self._check(
            lib.UNICORN_OpenDevice(serial.encode("utf-8"), byref(self._handle)),
            f"UNICORN_OpenDevice({serial})",
        )

        n_ch = c_uint32(0)
        self._check(
            lib.UNICORN_GetNumberOfAcquiredChannels(self._handle, byref(n_ch)),
            "UNICORN_GetNumberOfAcquiredChannels",
        )
        self._n_dev_channels = int(n_ch.value)
        if self._n_dev_channels != N_CANONICAL_CHANNELS:
            raise SourceError(
                f"o dispositivo reporta {self._n_dev_channels} canais, esperados "
                f"{N_CANONICAL_CHANNELS} (SPEC 1.1)"
            )
        self._buf = np.zeros(self._n_dev_channels * _SCANS_PER_READ, dtype=np.float32)

        self._info = DeviceInfo(
            source_name="unicorn_dll",
            sfreq_nominal=self._sfreq_nominal,
            n_channels=N_CANONICAL_CHANNELS,
            device_id=serial,
            units_to_uv=1.0,
            extra={
                "dll_path": str(dll_path),
                "api_version": float(lib.UNICORN_GetApiVersion()),
                "available_devices": serials,
                "scans_per_read": _SCANS_PER_READ,
            },
        )
        return self._info

    def start(self) -> None:
        if self._lib is None:
            raise SourceError("start() antes de open()")
        self._check(
            self._lib.UNICORN_StartAcquisition(self._handle, 0),
            "UNICORN_StartAcquisition",
        )
        self._started = True

    def _read_raw(self) -> np.ndarray:
        if self._lib is None or not self._started or self._buf is None:
            return empty_chunk()
        ptr = self._buf.ctypes.data_as(POINTER(c_float))
        self._check(
            self._lib.UNICORN_GetData(
                self._handle, _SCANS_PER_READ, ptr, self._buf.size
            ),
            "UNICORN_GetData",
        )
        # Escans intercalados -> (n_channels, n_scans)
        return self._buf.reshape(_SCANS_PER_READ, self._n_dev_channels).T.copy()

    def stop(self) -> None:
        if self._lib is not None and self._started:
            try:
                self._lib.UNICORN_StopAcquisition(self._handle)
            except Exception:
                pass
        self._started = False

    def close(self) -> None:
        if self._lib is not None and self._handle:
            try:
                self._lib.UNICORN_CloseDevice(byref(self._handle))
            except Exception:
                pass
        self._handle = ctypes.c_void_p()
        self._lib = None
