"""Gravação: streaming à prova de crash, e metadados completos."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mantraeeg.recorder import (
    RAW_ARRAY_NAME,
    RAW_META_NAME,
    EVENTS_NAME,
    RAW_STREAM_NAME,
    Recorder,
    next_session_id,
)
from mantraeeg.sources.base import N_CANONICAL_CHANNELS, DeviceInfo


@pytest.fixture
def device_info() -> DeviceInfo:
    return DeviceInfo(
        source_name="brainflow",
        sfreq_nominal=250.0,
        n_channels=N_CANONICAL_CHANNELS,
        device_id="UN-2022.01.16",
        units_to_uv=1.0,
        backend_channel_names=("Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"),
        extra={"n_backend_rows": 19},
    )


@pytest.fixture
def cfg_tmp(cfg_factory, tmp_path):
    return cfg_factory({"paths": {"sessions_dir": str(tmp_path / "sessions")}})


def chunk(k: int, start: int = 0) -> np.ndarray:
    out = np.zeros((N_CANONICAL_CHANNELS, k), dtype=np.float32)
    out[:] = np.arange(start, start + k, dtype=np.float32)
    out[15] = np.arange(start, start + k, dtype=np.float32)  # contador
    return out


def test_session_ids_are_sequential_and_anonymous(tmp_path):
    root = tmp_path / "sessions"
    root.mkdir()
    assert next_session_id(root) == "p001"
    (root / "2026-01-01T10-00-00_p001").mkdir()
    (root / "2026-01-01T11-00-00_p007").mkdir()
    assert next_session_id(root) == "p008"


def test_roundtrip_write_and_finalize(cfg_tmp, device_info):
    rec = Recorder(cfg_tmp)
    rec.write(chunk(100, 0))
    rec.write(chunk(150, 100))
    assert rec.n_samples == 250

    out = rec.finalize(device_info, measured_sfreq=250.5)
    saved = np.load(out / RAW_ARRAY_NAME)
    assert saved.shape == (N_CANONICAL_CHANNELS, 250)
    np.testing.assert_array_equal(saved[15], np.arange(250, dtype=np.float32))
    assert not (out / RAW_STREAM_NAME).exists()


def test_metadata_records_what_the_analysis_needs(cfg_tmp, device_info):
    rec = Recorder(cfg_tmp)
    rec.write(chunk(500))
    out = rec.finalize(device_info, measured_sfreq=250.523)
    meta = json.loads((out / RAW_META_NAME).read_text(encoding="utf-8"))

    assert meta["sfreq_measured"] == pytest.approx(250.523)
    assert meta["sfreq_nominal"] == 250.0
    assert meta["units_eeg"] == "uV"
    assert meta["n_samples"] == 500
    # A condição ocular e as durações viajam com os dados (SPEC 2).
    assert meta["eye_condition"] in ("open", "closed")
    assert meta["durations_s"]["calibration"] == cfg_tmp.protocol.calibration_s
    assert meta["durations_s"]["calib_use_last"] == cfg_tmp.protocol.calib_use_last_s
    assert "git_hash" in meta and "code_version" in meta


def test_metadata_records_the_montage_and_disowns_the_backend_names(
    cfg_tmp, device_info
):
    """Os nomes do backend são a montagem default do Unicorn e não podem ser usados."""
    rec = Recorder(cfg_tmp)
    rec.write(chunk(10))
    meta = json.loads(
        (rec.finalize(device_info, 250.0) / RAW_META_NAME).read_text(encoding="utf-8")
    )

    assert meta["montage"]["channels"]["0"] == "F3"
    assert meta["montage"]["channels"]["1"] == "F4"
    assert meta["montage"]["car_applied"] is False
    assert meta["source"]["backend_channel_names"][0] == "Fz"
    assert meta["source"]["backend_names_used_for_montage"] is False


def test_metadata_records_reference_adequacy(cfg_tmp, device_info):
    """A qualidade da referência que a duração escolhida comprou fica gravada."""
    rec = Recorder(cfg_tmp)
    rec.write(chunk(10))
    meta = json.loads(
        (rec.finalize(device_info, 250.0) / RAW_META_NAME).read_text(encoding="utf-8")
    )
    adequacy = meta["reference_adequacy"]
    # 60 s de calibracao usados por inteiro. Eram 30 (metade) e davam 14
    # epocas e 5 janelas — a coerencia nascia morta em qualquer sessao com a
    # duracao por omissao.
    assert adequacy["calib_used_s"] == 60.0
    assert adequacy["n_power_epochs"] == 29
    assert adequacy["n_conn_windows"] == 11
    assert adequacy["power_quality"] == "ok"


def test_events_are_persisted(cfg_tmp, device_info):
    rec = Recorder(cfg_tmp)
    rec.write(chunk(10))
    rec.add_event("phase", t_s=0.0, detail="CALIBRATION")
    rec.add_event("video_start", t_s=60.0, detail="mantra.mp4", monotonic=1234.5)
    rec.add_event("operator_note", t_s=90.0, detail="pessoa mexeu-se")

    out = rec.finalize(device_info, 250.0)
    events = json.loads((out / EVENTS_NAME).read_text(encoding="utf-8"))
    assert [e["kind"] for e in events] == ["phase", "video_start", "operator_note"]
    assert events[1]["payload"]["monotonic"] == 1234.5


def test_acquisition_stats_are_persisted(cfg_tmp, device_info):
    from mantraeeg.acquisition import AcquisitionStats

    stats = AcquisitionStats()
    stats.counter_gaps = [(120, 500.0, 507.0)]
    stats.overruns = 3
    stats.discarded_settle_samples = 2500
    stats.plausibility_warnings = ["desvio padrão mediano de 3000 µV"]

    rec = Recorder(cfg_tmp)
    rec.write(chunk(10))
    meta = json.loads(
        (rec.finalize(device_info, 250.0, acquisition_stats=stats) / RAW_META_NAME)
        .read_text(encoding="utf-8")
    )
    acq = meta["acquisition"]
    assert acq["counter_gaps"] == [{"at_sample": 120, "from": 500.0, "to": 507.0}]
    assert acq["overruns"] == 3
    assert acq["discarded_settle_samples"] == 2500
    assert acq["plausibility_warnings"]


def test_montage_validation_travels_with_the_data(cfg_tmp, device_info):
    from mantraeeg.montage import CheckResult, summarize

    validation = summarize([CheckResult("teste_de_toque", True, "confirmada")])
    rec = Recorder(cfg_tmp)
    rec.write(chunk(10))
    meta = json.loads(
        (rec.finalize(device_info, 250.0, montage_validation=validation) / RAW_META_NAME)
        .read_text(encoding="utf-8")
    )
    assert meta["montage"]["validation"]["all_passed"] is True


def test_truncated_stream_is_recovered_not_lost(cfg_tmp, device_info):
    """Uma sessão interrompida a meio tem de deixar dados aproveitáveis."""
    rec = Recorder(cfg_tmp)
    rec.write(chunk(300))
    rec.abort()

    # Simula um crash: o stream tem menos bytes do que o contador diz.
    stream = rec.dir / RAW_STREAM_NAME
    raw = np.fromfile(stream, dtype=np.float32)
    raw[: len(raw) // 2].tofile(stream)

    out = rec.finalize(device_info, 250.0)
    saved = np.load(out / RAW_ARRAY_NAME)
    assert saved.shape[1] == 150
    events = json.loads((out / EVENTS_NAME).read_text(encoding="utf-8"))
    assert any(e["kind"] == "truncated_recording" for e in events)


def test_wrong_shape_is_rejected(cfg_tmp):
    rec = Recorder(cfg_tmp)
    with pytest.raises(ValueError, match="esperado"):
        rec.write(np.zeros((8, 10), dtype=np.float32))
    rec.abort()


def test_write_after_close_is_rejected(cfg_tmp, device_info):
    rec = Recorder(cfg_tmp)
    rec.write(chunk(10))
    rec.finalize(device_info, 250.0)
    with pytest.raises(RuntimeError, match="fechado"):
        rec.write(chunk(10))
