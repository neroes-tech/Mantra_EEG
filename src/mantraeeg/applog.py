"""Um registo em ficheiro, porque a banca corre sem consola.

O executável é ``console=False`` — é um quiosque, e uma janela preta atrás da
aplicação seria pior do que nada. Mas isso significa que ``print`` não vai
para lado nenhum: em modo sem consola o CPython descarta silenciosamente tudo
o que se escreve para ``sys.stdout``, que é ``None``.

Consequência que já custou uma sessão: uma exceção dentro de um *slot* do Qt
imprimia um traceback que ninguém via, e a aplicação ficava parada num estado
intermédio sem uma única pista. Este módulo é a pista.

``data/app.log`` fica ao lado do executável, roda quando passa de 2 MB, e
apanha também o que escapa a todos os ``try`` — :func:`install_excepthook`
liga o ``sys.excepthook`` e o ``threading.excepthook`` ao mesmo ficheiro.
"""

from __future__ import annotations

import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

from .apppaths import app_root

MAX_BYTES = 2_000_000

_lock = threading.Lock()
_path: Path | None = None


def log_path() -> Path:
    global _path
    if _path is None:
        _path = app_root() / "data" / "app.log"
    return _path


def log(message: str) -> None:
    """Uma linha no registo, com hora. Nunca levanta."""
    try:
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            if path.exists() and path.stat().st_size > MAX_BYTES:
                path.replace(path.with_suffix(".log.1"))
            with path.open("a", encoding="utf-8") as handle:
                handle.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {message}\n")
    except Exception:
        pass


def log_exception(context: str, exc: BaseException) -> None:
    """A exceção inteira, com pilha, para se poder diagnosticar depois."""
    detail = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    )
    log(f"ERRO em {context}: {exc}\n{detail}")


def install_excepthook() -> None:
    """Manda para o ficheiro tudo o que ninguém apanhou.

    Inclui as threads: sem o ``threading.excepthook``, uma falha na thread de
    envio morria sem deixar rasto e o ecrã ficava à espera para sempre.
    """
    previous = sys.excepthook

    def hook(kind, value, tb) -> None:
        log_exception("nao apanhado", value)
        try:
            previous(kind, value, tb)
        except Exception:
            pass

    sys.excepthook = hook

    def thread_hook(args) -> None:
        log_exception(f"thread {args.thread.name if args.thread else '?'}", args.exc_value)

    threading.excepthook = thread_hook
    log("--- arranque ---")
