"""Fonte de replay sobre a gravação Unicorn real, e o contrato da camada de fontes."""

from __future__ import annotations

import numpy as np
import pytest

from mantraeeg.sources import make_source
from mantraeeg.sources.base import (
    ACCEL_SLICE,
    COUNTER_ROW,
    EEG_SLICE,
    N_CANONICAL_CHANNELS,
    SourceError,
    plausibility_warning,
)
from mantraeeg.sources.replay import UNICORN_CSV_COLUMNS, ReplaySource, load_recording

from conftest import FIXTURE_CSV


def test_csv_columns_match_the_canonical_layout():
    """O UnicornRecorder já escreve na ordem da SPEC 1.1."""
    assert len(UNICORN_CSV_COLUMNS) == N_CANONICAL_CHANNELS
    assert UNICORN_CSV_COLUMNS[:8] == tuple(f"EEG {i}" for i in range(1, 9))
    assert UNICORN_CSV_COLUMNS[COUNTER_ROW] == "Counter"


def test_load_real_recording(real_recording):
    assert real_recording.shape[0] == N_CANONICAL_CHANNELS
    assert real_recording.shape[1] == 30 * 250
    assert real_recording.dtype == np.float32


def test_counter_is_continuous_in_the_real_recording(real_recording):
    steps = np.diff(real_recording[COUNTER_ROW].astype(np.float64))
    assert np.all(steps == 1.0), "a gravação de referência não devia ter falhas"


def test_accelerometer_reads_about_one_g(real_recording):
    magnitude = np.linalg.norm(real_recording[ACCEL_SLICE], axis=0)
    assert 0.8 < magnitude.mean() < 1.2


def test_replay_delivers_everything_then_stops(cfg):
    source = make_source(cfg, override="replay", path=FIXTURE_CSV)
    info = source.open()
    assert info.source_name == "replay"
    assert info.extra["n_samples"] == 30 * 250

    source.start()
    total = 0
    for _ in range(10):
        chunk = source.read()
        assert chunk.shape[0] == N_CANONICAL_CHANNELS
        total += chunk.shape[1]
        if source.exhausted:
            break
    assert total == 30 * 250
    assert source.read().shape[1] == 0
    source.stop()
    source.close()


def test_replay_as_context_manager(cfg):
    with make_source(cfg, override="replay", path=FIXTURE_CSV) as source:
        source.start()
        assert source.read().shape[1] > 0


def test_missing_file_is_a_clear_error(cfg):
    source = make_source(cfg, override="replay", path="nao_existe.csv")
    with pytest.raises(SourceError, match="não encontrada"):
        source.open()


def test_empty_path_is_a_clear_error(cfg):
    with pytest.raises(SourceError, match="path"):
        ReplaySource({"path": ""}, 250.0).open()


def test_wrong_column_count_is_rejected(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    with pytest.raises(SourceError, match="colunas"):
        load_recording(bad)


def test_unsupported_extension_is_rejected(tmp_path):
    bad = tmp_path / "x.txt"
    bad.write_text("nada", encoding="utf-8")
    with pytest.raises(SourceError, match="extensão"):
        load_recording(bad)


def test_npy_roundtrip(tmp_path, real_recording):
    path = tmp_path / "raw.npy"
    np.save(path, real_recording)
    np.testing.assert_array_equal(load_recording(path), real_recording)


def test_npy_with_wrong_shape_is_rejected(tmp_path):
    path = tmp_path / "raw.npy"
    np.save(path, np.zeros((5, 100), dtype=np.float32))
    with pytest.raises(SourceError, match="esperado"):
        load_recording(path)


def test_synthetic_source_says_it_arrives_in_m1(cfg):
    with pytest.raises(SourceError, match="M1"):
        make_source(cfg, override="synthetic")


def test_unknown_source_is_rejected(cfg):
    with pytest.raises(SourceError, match="desconhecido"):
        make_source(cfg, override="telepatia")


# --------------------------------------------------------------------------- #
# Salvaguarda de unidades
# --------------------------------------------------------------------------- #
def test_plausibility_warning_catches_a_unit_scale_error():
    """Uma escala errada quebraria todos os limiares absolutos sem dar erro."""
    rng = np.random.default_rng(0)
    good = rng.normal(0, 10.0, (8, 2500))
    assert plausibility_warning(good) is None

    in_volts = good * 1e-6
    assert "abaixo" in (plausibility_warning(in_volts) or "")

    too_big = good * 1e4
    assert "acima" in (plausibility_warning(too_big) or "")


def test_plausibility_warning_on_real_recording_flags_no_contact(real_recording):
    """A gravação de referência tem deriva enorme: tem de ser sinalizada."""
    warning = plausibility_warning(real_recording[EEG_SLICE])
    assert warning is not None and "acima" in warning
