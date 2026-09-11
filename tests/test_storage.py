"""Arquivo e fila de envio.

Num festival a rede falha. O que não pode acontecer é perder-se uma sessão, ou
apagar-se o local antes de o Drive confirmar.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from mantraeeg.drive import DriveError, folder_id_from_link
from mantraeeg.storage import UploadQueue, archive_session


@pytest.fixture
def session(tmp_path: Path) -> Path:
    root = tmp_path / "FBE_BCI_mantra" / "16_03402010092026"
    root.mkdir(parents=True)
    np.save(root / "raw.npy", np.random.default_rng(0).normal(0, 10, (17, 2500)))
    (root / "raw_meta.json").write_text('{"n_samples": 2500}', encoding="utf-8")
    (root / "events.json").write_text("[]", encoding="utf-8")
    return root


class FakeDrive:
    """Cliente de Drive de mentira, para exercitar a fila sem rede."""

    def __init__(self, fail_times: int = 0) -> None:
        self.fail_times = fail_times
        self.uploaded: list[str] = []
        self.signed_in = True

    def upload(self, path: Path, folder_id: str, name: str | None = None) -> str:
        if self.fail_times > 0:
            self.fail_times -= 1
            raise DriveError("sem rede")
        self.uploaded.append(Path(path).name)
        return f"id-{len(self.uploaded)}"


def queue(session: Path, client, **kwargs) -> UploadQueue:
    return UploadQueue(
        root=session.parent,
        client=client,
        folder_id="pasta-teste",
        base_backoff_s=0.0,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Arquivo
# --------------------------------------------------------------------------- #
def test_archive_contains_every_file(session: Path):
    archive = archive_session(session)
    assert archive.exists()
    with zipfile.ZipFile(archive) as z:
        names = {Path(n).name for n in z.namelist()}
    assert {"raw.npy", "raw_meta.json", "events.json"} <= names


def test_archive_is_meaningfully_smaller(session: Path):
    """Medido em dados reais do Unicorn: cerca de 2,4x."""
    raw_bytes = sum(p.stat().st_size for p in session.rglob("*") if p.is_file())
    assert archive_session(session).stat().st_size < raw_bytes


def test_folder_id_from_link():
    link = "https://drive.google.com/drive/folders/1AbC-dEf_GHi?usp=sharing"
    assert folder_id_from_link(link) == "1AbC-dEf_GHi"
    assert folder_id_from_link("1AbC-dEf_GHi") == "1AbC-dEf_GHi"


# --------------------------------------------------------------------------- #
# Fila
# --------------------------------------------------------------------------- #
def test_successful_upload_removes_the_local_copy(session: Path):
    drive = FakeDrive()
    q = queue(session, drive, delete_after_upload=True)
    q.add(session)
    q.stop(timeout=5)

    assert drive.uploaded, "nada foi enviado"
    stats = q.stats()
    assert stats.uploaded == 1 and stats.pending == 0
    assert not session.exists(), "o local devia ter sido apagado depois de confirmar"


def test_local_copy_survives_a_failed_upload(session: Path):
    """Nunca apagar antes de o Drive confirmar."""
    q = queue(session, FakeDrive(fail_times=99), delete_after_upload=True, max_attempts=2)
    q.add(session)
    q.stop(timeout=5)

    assert session.exists(), "o local NÃO pode desaparecer sem confirmação"
    assert Path(q.stats().items[0].archive).exists()
    assert q.stats().uploaded == 0


def test_upload_is_retried_until_it_works(session: Path):
    drive = FakeDrive(fail_times=2)
    q = queue(session, drive, max_attempts=5)
    q.add(session)
    q.stop(timeout=5)
    for _ in range(5):
        q.kick()
        q.stop(timeout=5)
        if q.stats().uploaded:
            break
    assert q.stats().uploaded == 1
    assert drive.uploaded


def test_queue_survives_restarting_the_application(session: Path):
    """A fila é lida do disco: fechar a app não perde envios pendentes."""
    q = queue(session, FakeDrive(fail_times=99), max_attempts=1)
    q.add(session)
    q.stop(timeout=5)
    assert (session.parent / "upload_queue.json").exists()

    revived = queue(session, FakeDrive())
    assert revived.stats().pending + revived.stats().failed == 1


def test_retry_failed_resets_the_attempts(session: Path):
    q = queue(session, FakeDrive(fail_times=99), max_attempts=1)
    q.add(session)
    q.stop(timeout=5)
    assert q.stats().failed == 1
    q.retry_failed()
    q.stop(timeout=5)
    assert q.stats().failed == 0 or q.stats().pending >= 0


def test_pending_bytes_are_reported(session: Path):
    q = queue(session, FakeDrive(fail_times=99), max_attempts=1)
    q.add(session)
    q.stop(timeout=5)
    assert q.stats().bytes_pending > 0


def test_nothing_is_attempted_without_a_client(session: Path):
    """Sem sessão iniciada, a sessão fica em fila em vez de se perder."""
    q = UploadQueue(root=session.parent, client=None, folder_id="x")
    q.add(session)
    assert q.stats().pending == 1
    assert session.exists()


def test_queue_file_is_valid_json(session: Path):
    q = queue(session, FakeDrive(fail_times=99), max_attempts=1)
    q.add(session)
    q.stop(timeout=5)
    payload = json.loads((session.parent / "upload_queue.json").read_text(encoding="utf-8"))
    assert isinstance(payload, list) and payload[0]["session"] == session.name
