"""Arquivo das sessões e fila de envio para o Drive.

Números medidos numa gravação real: uma sessão de 5 min ocupa **5,1 MB** em
bruto e **2,1 MB** comprimida. Cem sessões são 210 MB — não é um problema de
espaço em nenhum Surface. A razão para enviar é outra: tirar os dados das
máquinas, que num festival podem ser levadas, avariar ou ser reiniciadas, e
juntá-los todos num sítio.

A fila é **à prova de falhas de rede**, que num festival são a norma:

- o arquivo é escrito primeiro, e só depois entra na fila;
- o envio corre numa thread, nunca na interface;
- o que falha é tentado outra vez, com espera crescente;
- o local só é apagado **depois** de o Drive confirmar;
- a fila sobrevive a fechar a aplicação — é lida do disco ao arrancar.

O pior caso, sem rede o dia inteiro, é ficar com 210 MB por enviar e uma lista
que se despacha quando houver rede. Nunca se perde uma sessão por não haver
wi-fi.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .drive import DriveClient, DriveError

QUEUE_NAME = "upload_queue.json"
ARCHIVE_SUFFIX = ".zip"


@dataclass
class QueueItem:
    archive: str
    session: str
    attempts: int = 0
    last_error: str = ""
    next_attempt_at: float = 0.0
    uploaded_id: str = ""

    @property
    def done(self) -> bool:
        return bool(self.uploaded_id)


@dataclass
class QueueStats:
    pending: int = 0
    uploaded: int = 0
    failed: int = 0
    #: Bytes ainda por enviar, incluindo os que ja esgotaram as tentativas.
    bytes_pending: int = 0
    last_error: str = ""
    busy: bool = False
    items: list[QueueItem] = field(default_factory=list)


def archive_session(session_dir: Path, destination: Path | None = None) -> Path:
    """Comprime a pasta da sessão para um ``.zip`` ao lado dela.

    Compressão medida: 2,4x sobre os dados do Unicorn.
    """
    session_dir = Path(session_dir)
    target = Path(destination or session_dir.parent / (session_dir.name + ARCHIVE_SUFFIX))
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(session_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(session_dir.parent))
    return target


class UploadQueue:
    """Fila persistente de envios, com um trabalhador em fundo."""

    def __init__(
        self,
        root: Path,
        client: DriveClient | None,
        folder_id: str,
        delete_after_upload: bool = True,
        max_attempts: int = 5,
        base_backoff_s: float = 20.0,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._path = self._root / QUEUE_NAME
        self._client = client
        self._folder_id = folder_id
        self._delete_after = delete_after_upload
        self._max_attempts = max_attempts
        self._backoff = base_backoff_s
        self._on_change = on_change

        self._items: list[QueueItem] = self._load()
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self._busy = False

    # -- persistência ------------------------------------------------------------ #
    def _load(self) -> list[QueueItem]:
        if not self._path.exists():
            return []
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            return [QueueItem(**item) for item in raw]
        except Exception:
            return []

    def _save(self) -> None:
        self._path.write_text(
            json.dumps([asdict(i) for i in self._items], indent=2), encoding="utf-8"
        )
        if self._on_change is not None:
            self._on_change()

    # -- API ----------------------------------------------------------------------- #
    def add(self, session_dir: Path) -> QueueItem:
        """Arquiva a sessão e põe-na na fila. Não envia já."""
        archive = archive_session(session_dir)
        item = QueueItem(archive=str(archive), session=Path(session_dir).name)
        with self._lock:
            self._items.append(item)
            self._save()
        self.kick()
        return item

    def stats(self) -> QueueStats:
        with self._lock:
            items = list(self._items)
        pending = [i for i in items if not i.done and i.attempts < self._max_attempts]
        failed = [i for i in items if not i.done and i.attempts >= self._max_attempts]
        # Conta pendentes E falhados: os bytes dos falhados continuam no disco
        # e continuam por enviar. Para quem esta na banca a pergunta e "quanto
        # falta sair daqui", nao "quanto e que o trabalhador ainda vai tentar".
        size = 0
        for item in pending + failed:
            path = Path(item.archive)
            if path.exists():
                size += path.stat().st_size
        return QueueStats(
            pending=len(pending),
            uploaded=sum(1 for i in items if i.done),
            failed=len(failed),
            bytes_pending=size,
            last_error=next(
                (i.last_error for i in reversed(items) if i.last_error), ""
            ),
            busy=self._busy,
            items=items,
        )

    def kick(self) -> None:
        """Acorda o trabalhador. Seguro chamar a qualquer momento."""
        if self._worker is not None and self._worker.is_alive():
            return
        if self._client is None or not self._client.signed_in:
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def retry_failed(self) -> None:
        with self._lock:
            for item in self._items:
                if not item.done:
                    item.attempts = 0
                    item.next_attempt_at = 0.0
            self._save()
        self.kick()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=timeout)

    # -- trabalhador ------------------------------------------------------------------ #
    def _next_item(self) -> QueueItem | None:
        now = time.time()
        with self._lock:
            for item in self._items:
                if (
                    not item.done
                    and item.attempts < self._max_attempts
                    and item.next_attempt_at <= now
                ):
                    return item
        return None

    def _run(self) -> None:
        self._busy = True
        try:
            while not self._stop.is_set():
                item = self._next_item()
                if item is None:
                    break
                self._upload_one(item)
        finally:
            self._busy = False
            if self._on_change is not None:
                self._on_change()

    def _upload_one(self, item: QueueItem) -> None:
        archive = Path(item.archive)
        if not archive.exists():
            with self._lock:
                item.last_error = "arquivo desaparecido"
                item.attempts = self._max_attempts
                self._save()
            return
        try:
            assert self._client is not None
            file_id = self._client.upload(archive, self._folder_id)
        except DriveError as exc:
            with self._lock:
                item.attempts += 1
                item.last_error = str(exc)
                # Espera crescente: um festival sem rede não deve gerar milhares
                # de tentativas nem encher o disco de logs.
                item.next_attempt_at = time.time() + self._backoff * (2 ** (item.attempts - 1))
                self._save()
            return

        with self._lock:
            item.uploaded_id = file_id or "enviado"
            item.last_error = ""
            self._save()

        if self._delete_after:
            # Só depois de o Drive confirmar. Nunca antes.
            archive.unlink(missing_ok=True)
            session = self._root / item.session
            if session.is_dir():
                shutil.rmtree(session, ignore_errors=True)
            with self._lock:
                self._save()
