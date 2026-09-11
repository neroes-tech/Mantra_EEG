"""Ring buffer: wrap-around, overrun contado, integridade sob concorrência."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from mantraeeg.ring_buffer import RingBuffer


def ramp(n_channels: int, start: int, k: int) -> np.ndarray:
    """Chunk cujos valores são o índice absoluto da amostra, por canal."""
    idx = np.arange(start, start + k, dtype=np.float32)
    return np.tile(idx, (n_channels, 1)) + np.arange(n_channels, dtype=np.float32)[:, None]


def test_rejects_bad_geometry():
    with pytest.raises(ValueError):
        RingBuffer(0, 10)
    with pytest.raises(ValueError):
        RingBuffer(4, 0)
    rb = RingBuffer(4, 10)
    with pytest.raises(ValueError, match="esperado chunk"):
        rb.write(np.zeros((3, 5), dtype=np.float32))


def test_write_read_roundtrip():
    rb = RingBuffer(4, 100)
    chunk = ramp(4, 0, 30)
    rb.write(chunk)
    assert rb.available() == 30
    np.testing.assert_array_equal(rb.read(), chunk)
    assert rb.available() == 0
    assert rb.read().shape == (4, 0)


def test_partial_read():
    rb = RingBuffer(2, 50)
    rb.write(ramp(2, 0, 20))
    first = rb.read(5)
    assert first.shape == (2, 5)
    np.testing.assert_array_equal(first, ramp(2, 0, 5))
    np.testing.assert_array_equal(rb.read(), ramp(2, 5, 15))


def test_wrap_around_preserves_order():
    rb = RingBuffer(3, 16)
    rb.write(ramp(3, 0, 10))
    rb.read()
    rb.write(ramp(3, 10, 12))  # atravessa a fronteira
    np.testing.assert_array_equal(rb.read(), ramp(3, 10, 12))


def test_overrun_is_counted_never_silent():
    """Se a UI atrasar, isso tem de aparecer no events.json, não desaparecer."""
    rb = RingBuffer(2, 10)
    rb.write(ramp(2, 0, 25))
    stats = rb.stats()
    assert stats.overruns == 15
    # Sobrevivem as últimas `capacity` amostras.
    np.testing.assert_array_equal(rb.read(), ramp(2, 15, 10))


def test_write_larger_than_capacity_keeps_the_tail():
    rb = RingBuffer(2, 8)
    rb.write(ramp(2, 0, 20))
    np.testing.assert_array_equal(rb.read(), ramp(2, 12, 8))


def test_peek_last_does_not_consume():
    rb = RingBuffer(2, 100)
    rb.write(ramp(2, 0, 40))
    peeked = rb.peek_last(10)
    np.testing.assert_array_equal(peeked, ramp(2, 30, 10))
    assert rb.available() == 40  # o monitor de contacto não rouba dados ao recorder
    np.testing.assert_array_equal(rb.peek_last(10), peeked)


def test_peek_last_clamps_to_available():
    rb = RingBuffer(2, 100)
    rb.write(ramp(2, 0, 5))
    assert rb.peek_last(50).shape == (2, 5)
    assert RingBuffer(2, 100).peek_last(10).shape == (2, 0)


def test_clear_resets_everything():
    rb = RingBuffer(2, 20)
    rb.write(ramp(2, 0, 50))
    rb.clear()
    assert rb.available() == 0
    assert rb.stats().overruns == 0


def test_concurrent_writer_and_reader_lose_nothing():
    """Escritor e leitor em threads distintas: nada duplicado, nada perdido.

    A capacidade cobre o total, para o teste medir integridade sob concorrência
    e não a velocidade relativa das threads (o overrun tem o seu próprio teste).
    """
    n_ch, total, capacity = 2, 20_000, 32_768
    rb = RingBuffer(n_ch, capacity)
    received: list[np.ndarray] = []
    done = threading.Event()

    def writer() -> None:
        pos = 0
        while pos < total:
            k = min(97, total - pos)
            rb.write(ramp(n_ch, pos, k))
            pos += k
        done.set()

    def reader() -> None:
        while not done.is_set() or rb.available():
            chunk = rb.read()
            if chunk.shape[1]:
                received.append(chunk)

    t_w = threading.Thread(target=writer)
    t_r = threading.Thread(target=reader)
    t_r.start()
    t_w.start()
    t_w.join(timeout=30)
    t_r.join(timeout=30)

    got = np.concatenate(received, axis=1) if received else np.zeros((n_ch, 0))
    assert rb.stats().overruns == 0, "o leitor devia ter acompanhado"
    assert got.shape[1] == total
    np.testing.assert_array_equal(got, ramp(n_ch, 0, total))
