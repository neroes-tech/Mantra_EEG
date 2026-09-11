"""Gestão do dispositivo e nomes de sessão.

A aplicação passa horas numa banca sem ninguém, com o headset desligado. A
versão anterior bloqueava nessa situação; estes testes fixam o comportamento
que a impede de voltar a bloquear.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from mantraeeg.device import DeviceManager, DeviceState
from mantraeeg.recorder import session_folder_name

from conftest import FIXTURE_CSV


@pytest.fixture
def manager(cfg_factory, tmp_path):
    built = cfg_factory({"paths": {"sessions_dir": str(tmp_path / "sessions")}})
    return DeviceManager(
        built, source_override="replay", replay_file=str(FIXTURE_CSV)
    )


# --------------------------------------------------------------------------- #
# Nome da pasta de sessão
# --------------------------------------------------------------------------- #
def test_session_folder_name_format():
    """``<id do unicorn>_<HHMMSSDDMMAAAA>``."""
    name = session_folder_name("16", datetime(2026, 9, 10, 3, 40, 20))
    assert name == "16_03402010092026"


def test_session_folder_name_has_no_personal_data():
    name = session_folder_name("16", datetime(2026, 9, 10, 3, 40, 20))
    assert name.replace("_", "").isdigit()


def test_short_id_extracts_the_unicorn_number(manager):
    """``UN-2022.01.16`` -> ``16``, que é o que nomeia a pasta."""
    from mantraeeg.sources.base import DeviceInfo

    manager._info = DeviceInfo("brainflow", 250.0, 17, "UN-2022.01.16")
    assert manager.short_id == "16"


def test_short_id_without_device(manager):
    assert manager.short_id == "sem-id"


# --------------------------------------------------------------------------- #
# Ciclo de vida
# --------------------------------------------------------------------------- #
def test_starts_disconnected(manager):
    status = manager.status()
    assert status.state == DeviceState.DISCONNECTED
    assert not status.usable
    assert status.needs_restart


def test_connect_then_disconnect(manager):
    assert manager.connect()
    assert manager.status().state == DeviceState.READY
    assert manager.status().usable
    manager.disconnect()
    assert manager.status().state == DeviceState.DISCONNECTED


def test_streaming_is_separate_from_being_connected(manager):
    """O LED só fica fixo durante a recolha: ligar não é adquirir."""
    manager.connect()
    assert manager.status().state == DeviceState.READY
    assert manager.start_streaming()
    assert manager.status().state == DeviceState.STREAMING
    manager.stop_streaming()
    assert manager.status().state == DeviceState.READY
    manager.disconnect()


def test_start_streaming_connects_if_needed(manager):
    assert manager.status().state == DeviceState.DISCONNECTED
    assert manager.start_streaming()
    assert manager.status().state == DeviceState.STREAMING
    manager.disconnect()


def test_failed_connection_reports_actionable_advice(cfg_factory, tmp_path):
    built = cfg_factory({"paths": {"sessions_dir": str(tmp_path / "s")}})
    manager = DeviceManager(
        built, source_override="replay", replay_file=str(tmp_path / "nao_existe.csv")
    )
    assert not manager.connect()
    status = manager.status()
    assert status.state == DeviceState.DISCONNECTED
    assert status.needs_restart
    assert "headset" in status.detail


def test_disconnect_is_idempotent(manager):
    manager.disconnect()
    manager.disconnect()
    assert manager.status().state == DeviceState.DISCONNECTED


def test_reconnect_after_disconnect(manager):
    """Depois de horas em espera, voltar a ligar tem de funcionar."""
    manager.connect()
    manager.start_streaming()
    manager.disconnect()
    assert manager.status().state == DeviceState.DISCONNECTED
    assert manager.connect()
    assert manager.start_streaming()
    manager.disconnect()


def test_peek_last_is_safe_without_device(manager):
    assert manager.peek_last(5.0).shape == (17, 0)
    assert manager.elapsed_s == 0.0
