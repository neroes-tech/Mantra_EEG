"""Ring buffer de escritor único / leitor único para a thread de aquisição.

A thread de aquisição escreve; a consola do operador e o recorder leem. A UI
nunca toca no dispositivo (SPEC 1.4).

Um overrun — o leitor ficar para trás e ser ultrapassado — é contado e exposto,
nunca silenciado. Se a UI atrasar durante uma sessão, isso tem de aparecer no
``events.json``, não desaparecer.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RingStats:
    total_written: int
    total_read: int
    overruns: int
    capacity: int

    @property
    def available(self) -> int:
        return self.total_written - self.total_read


class RingBuffer:
    """Buffer circular de ``(n_channels, capacity)`` amostras.

    As amostras são escritas em colunas. ``read`` devolve uma cópia contígua em
    ordem cronológica.
    """

    def __init__(self, n_channels: int, capacity: int, dtype: type = np.float32):
        if n_channels <= 0:
            raise ValueError(f"n_channels tem de ser > 0, obtido {n_channels}")
        if capacity <= 0:
            raise ValueError(f"capacity tem de ser > 0, obtido {capacity}")
        self._buf = np.zeros((n_channels, capacity), dtype=dtype)
        self._capacity = capacity
        self._n_channels = n_channels
        self._write_pos = 0          # total de amostras escritas (monotónico)
        self._read_pos = 0           # total de amostras lidas (monotónico)
        self._overruns = 0
        self._lock = threading.Lock()

    # -- propriedades -------------------------------------------------------- #
    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def n_channels(self) -> int:
        return self._n_channels

    def stats(self) -> RingStats:
        with self._lock:
            return RingStats(
                total_written=self._write_pos,
                total_read=self._read_pos,
                overruns=self._overruns,
                capacity=self._capacity,
            )

    def available(self) -> int:
        """Amostras escritas e ainda não lidas (limitado pela capacidade)."""
        with self._lock:
            return min(self._write_pos - self._read_pos, self._capacity)

    # -- escrita ------------------------------------------------------------- #
    def write(self, chunk: np.ndarray) -> None:
        """Escreve ``(n_channels, k)``. Escritas maiores que a capacidade truncam."""
        if chunk.ndim != 2 or chunk.shape[0] != self._n_channels:
            raise ValueError(
                f"esperado chunk ({self._n_channels}, k), obtido {chunk.shape}"
            )
        k = chunk.shape[1]
        if k == 0:
            return
        dropped = 0
        if k > self._capacity:
            # Mais do que o buffer inteiro: só as últimas `capacity` sobrevivem.
            # As restantes são perda real e são contadas — nunca silenciadas.
            dropped = k - self._capacity
            chunk = chunk[:, -self._capacity :]
            k = self._capacity

        with self._lock:
            self._overruns += dropped
            start = self._write_pos % self._capacity
            end = start + k
            if end <= self._capacity:
                self._buf[:, start:end] = chunk
            else:
                split = self._capacity - start
                self._buf[:, start:] = chunk[:, :split]
                self._buf[:, : end - self._capacity] = chunk[:, split:]
            self._write_pos += k

            # O leitor foi ultrapassado?
            unread = self._write_pos - self._read_pos
            if unread > self._capacity:
                self._overruns += unread - self._capacity
                self._read_pos = self._write_pos - self._capacity

    # -- leitura ------------------------------------------------------------- #
    def read(self, n: int | None = None) -> np.ndarray:
        """Consome e devolve até ``n`` amostras (todas as disponíveis se ``None``)."""
        with self._lock:
            avail = min(self._write_pos - self._read_pos, self._capacity)
            k = avail if n is None else min(n, avail)
            if k <= 0:
                return np.zeros((self._n_channels, 0), dtype=self._buf.dtype)
            out = self._extract(self._read_pos, k)
            self._read_pos += k
            return out

    def peek_last(self, n: int) -> np.ndarray:
        """Últimas ``n`` amostras **sem** consumir. Para o monitor de contacto."""
        with self._lock:
            avail = min(self._write_pos, self._capacity)
            k = min(n, avail)
            if k <= 0:
                return np.zeros((self._n_channels, 0), dtype=self._buf.dtype)
            return self._extract(self._write_pos - k, k)

    def _extract(self, from_abs: int, k: int) -> np.ndarray:
        """Cópia contígua de ``k`` amostras a partir da posição absoluta. Sob lock."""
        start = from_abs % self._capacity
        end = start + k
        if end <= self._capacity:
            return self._buf[:, start:end].copy()
        split = self._capacity - start
        out = np.empty((self._n_channels, k), dtype=self._buf.dtype)
        out[:, :split] = self._buf[:, start:]
        out[:, split:] = self._buf[:, : end - self._capacity]
        return out

    def clear(self) -> None:
        with self._lock:
            self._write_pos = 0
            self._read_pos = 0
            self._overruns = 0
            self._buf.fill(0)
