"""Deixas áudio e reprodução do mantra.

De olhos fechados a pessoa não vê o ecrã, portanto o início e o fim de cada fase
têm de ser assinalados por som (``config.protocol.cues``).

As deixas são **sintetizadas** se os ficheiros não existirem: são duas
campainhas curtas, e fazer depender o protocolo de um asset que se pode perder
antes do evento seria frágil. Um ficheiro fornecido pelo operador tem sempre
precedência.

O áudio do mantra e o vídeo são **independentes**. O vídeo é para quem assiste à
banca, e a tecla ``V`` esconde-o sem nunca interromper o som — a pessoa está de
olhos fechados e o que importa é o que ela ouve.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

#: Duração e envelope das campainhas sintetizadas.
CUE_SFREQ = 44_100
CUE_DURATION_S = 0.45
CUE_FADE_S = 0.06
#: Início: quinta ascendente. Fim: a mesma, descendente. Distinguíveis de olhos
#: fechados sem serem sobressaltantes.
CUE_START_HZ = (660.0, 990.0)
CUE_END_HZ = (990.0, 660.0)


def _tone(frequencies: tuple[float, float], duration_s: float) -> bytes:
    """Duas notas encadeadas, com fade para não estalar."""
    frames = bytearray()
    total = int(CUE_SFREQ * duration_s)
    half = total // 2
    fade = max(int(CUE_SFREQ * CUE_FADE_S), 1)
    for i in range(total):
        freq = frequencies[0] if i < half else frequencies[1]
        phase = 2.0 * math.pi * freq * (i % half) / CUE_SFREQ
        envelope = min(i / fade, (total - i) / fade, 1.0)
        value = int(0.35 * envelope * math.sin(phase) * 32767)
        frames += struct.pack("<h", max(-32768, min(32767, value)))
    return bytes(frames)


def _write_wav(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(CUE_SFREQ)
        handle.writeframes(payload)


def ensure_cue_files(start: Path, end: Path) -> tuple[Path, Path]:
    """Garante que as duas deixas existem, sintetizando as que faltarem."""
    if not start.exists():
        _write_wav(start, _tone(CUE_START_HZ, CUE_DURATION_S))
    if not end.exists():
        _write_wav(end, _tone(CUE_END_HZ, CUE_DURATION_S))
    return start, end


class AudioPlayer:
    """Reprodução de deixas e do mantra, com degradação limpa.

    Se o QtMultimedia não estiver disponível ou faltar um ficheiro, a aplicação
    continua a funcionar e regista o que não conseguiu tocar — numa banca, uma
    sessão sem som é má, mas uma aplicação que rebenta é pior.
    """

    def __init__(self, cue_start: Path, cue_end: Path, mantra: Path | None) -> None:
        self._problems: list[str] = []
        self._cue_start, self._cue_end = ensure_cue_files(cue_start, cue_end)
        self._mantra_path = mantra
        self._effects: dict[str, object] = {}
        self._mantra_player = None
        self._mantra_output = None
        self._available = self._build()

    def _build(self) -> bool:
        try:
            from PyQt6.QtCore import QUrl
            from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer, QSoundEffect
        except Exception as exc:  # pragma: no cover - ambiente sem QtMultimedia
            self._problems.append(f"QtMultimedia indisponível: {exc}")
            return False

        for name, path in (("start", self._cue_start), ("end", self._cue_end)):
            effect = QSoundEffect()
            effect.setSource(QUrl.fromLocalFile(str(path.resolve())))
            effect.setVolume(0.6)
            self._effects[name] = effect

        if self._mantra_path is not None and self._mantra_path.exists():
            self._mantra_player = QMediaPlayer()
            self._mantra_output = QAudioOutput()
            self._mantra_player.setAudioOutput(self._mantra_output)
            self._mantra_player.setSource(
                QUrl.fromLocalFile(str(self._mantra_path.resolve()))
            )
        elif self._mantra_path is not None:
            self._problems.append(
                f"áudio do mantra não encontrado: {self._mantra_path}"
            )
        return True

    # -- estado --------------------------------------------------------------- #
    @property
    def problems(self) -> list[str]:
        return list(self._problems)

    @property
    def has_mantra(self) -> bool:
        return self._mantra_player is not None

    # -- reprodução ------------------------------------------------------------ #
    def cue_start(self) -> None:
        self._play_effect("start")

    def cue_end(self) -> None:
        self._play_effect("end")

    def _play_effect(self, name: str) -> None:
        effect = self._effects.get(name)
        if effect is None:
            return
        try:
            effect.play()
        except Exception as exc:  # pragma: no cover
            self._problems.append(f"falha ao tocar a deixa {name}: {exc}")

    def play_mantra(self) -> bool:
        """Arranca o mantra. Devolve ``False`` se não houver áudio."""
        if self._mantra_player is None:
            return False
        self._mantra_player.setPosition(0)
        self._mantra_player.play()
        return True

    def stop_mantra(self) -> None:
        if self._mantra_player is not None:
            self._mantra_player.stop()

    def set_volume(self, volume: float) -> None:
        if self._mantra_output is not None:
            self._mantra_output.setVolume(max(0.0, min(1.0, volume)))
