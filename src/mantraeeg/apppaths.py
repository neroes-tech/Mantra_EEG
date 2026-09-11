"""Onde é que a aplicação vive, com e sem executável.

Em desenvolvimento a raiz é a pasta do projeto. Depois de empacotada, é a
pasta onde está o ``.exe`` — e **não** a pasta temporária onde o PyInstaller
descomprime o código.

A diferença importa para tudo o que o operador tem de poder ver e mexer na
banca sem reconstruir nada: a configuração, os mantras, o logótipo, e a pasta
das gravações. Se estes fossem para dentro do pacote, mudar de mantra exigiria
um programador, e as sessões desapareceriam com a pasta temporária no fim.

Regra prática: **código vai para dentro do pacote, conteúdo fica ao lado.**
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def frozen() -> bool:
    """Estamos dentro de um executável do PyInstaller?"""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def app_root() -> Path:
    """A pasta a que os caminhos relativos da configuração se referem.

    - Empacotado: a pasta do executável. É onde ficam ``config/``,
      ``assets/`` e ``data/``, ao lado do ``.exe``.
    - Em desenvolvimento: a raiz do repositório, deduzida deste ficheiro, para
      correr de qualquer diretório sem partir os caminhos.
    - ``MANTRA_HOME`` sobrepõe-se às duas, para testes e para instalações onde
      os dados não podem viver ao lado da aplicação.
    """
    override = os.environ.get("MANTRA_HOME", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def resolve(path: str | Path) -> Path:
    """Um caminho da configuração, ancorado na raiz da aplicação.

    Caminhos absolutos passam intactos — quem escreveu um caminho absoluto na
    configuração quis exatamente esse.
    """
    candidate = Path(path)
    return candidate if candidate.is_absolute() else app_root() / candidate
