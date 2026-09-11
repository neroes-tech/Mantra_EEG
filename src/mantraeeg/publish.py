"""Onde é que o relatório fica a viver, para o link abrir no browser.

O Google Drive **não serve HTML**. Medido: um ficheiro ``text/html``
partilhado sai com ``Content-Type: application/octet-stream`` e
``Content-Disposition: attachment``, portanto o link descarrega em vez de
abrir. Serve para arquivo, não para mandar a um amigo.

Este módulo publica a página num alojamento estático, onde o link abre. Três
modos, todos sem dependências novas — o GitHub e o Drive falam HTTP e a
biblioteca padrão chega:

``github_pages``
    Envia o ficheiro para um repositório público pela API de conteúdos do
    GitHub. Fica em ``https://<owner>.github.io/<repo>/r/<ficheiro>``, é
    gratuito, é permanente, e aceita domínio próprio.
``base_url``
    Assume que alguém põe o ficheiro num sítio que já servem. A aplicação
    devolve o endereço mas não publica nada.
``drive``
    O que existia. O link é livre mas descarrega.

**O nome do ficheiro leva um sufixo aleatório de propósito.** Um relatório
num alojamento público é legível por quem souber o endereço, e um nome como
``FBE-2026-110628.html`` adivinha-se a partir da hora da sessão. Com oito
dígitos hexadecimais à frente há 4 mil milhões de hipóteses por código, o que
põe o endereço fora do alcance de quem não o recebeu. Não é o mesmo que
privado: é "não indexado e não adivinhável".
"""

from __future__ import annotations

import base64
import json
import secrets
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

GITHUB_API = "https://api.github.com"


class PublishError(RuntimeError):
    """Falha na publicação, com uma mensagem que o operador possa ler."""


@dataclass(frozen=True)
class GitHubTarget:
    """Um repositório do GitHub com o Pages ligado."""

    owner: str
    repo: str
    branch: str
    folder: str
    token: str
    #: Domínio próprio, se houver. Vazio = o endereço ``github.io``.
    base_url: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.owner and self.repo and self.token)

    def url_for(self, path: str) -> str:
        if self.base_url:
            return f"{self.base_url.rstrip('/')}/{path}"
        # O anfitrião em minúsculas: é DNS, e um endereço com maiúsculas no
        # meio parece um erro de escrita a quem o recebe. O caminho fica como
        # está — esse é sensível a maiúsculas.
        return f"https://{self.owner.lower()}.github.io/{self.repo}/{path}"


def unguessable_name(code: str, suffix: str = ".html") -> str:
    """``FBE-2026-110628-3f9a2c81.html``. Ver a nota no topo do módulo."""
    return f"{code}-{secrets.token_hex(4)}{suffix}"


def _github_request(
    target: GitHubTarget, method: str, path: str, body: dict | None = None
) -> dict:
    request = urllib.request.Request(
        f"{GITHUB_API}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={
            "Authorization": f"Bearer {target.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "mantra-eeg",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        if exc.code == 401:
            raise PublishError(
                "o GitHub recusou o token (401). Está expirado, ou foi copiado "
                "com um espaço à frente."
            ) from exc
        if exc.code == 403:
            raise PublishError(
                "o GitHub recusou (403). O token precisa da permissão "
                "'Contents: Read and write' neste repositório."
            ) from exc
        if exc.code == 404:
            raise PublishError(
                f"o GitHub não encontrou {target.owner}/{target.repo} (404). "
                f"Confirme o nome, e que o token dá acesso a este repositório."
            ) from exc
        raise PublishError(f"o GitHub devolveu {exc.code}: {detail}") from exc
    except OSError as exc:
        raise PublishError(f"sem rede ao contactar o GitHub: {exc}") from exc


def _put_file(
    target: GitHubTarget, path: str, content: bytes, message: str, attempts: int = 4
) -> None:
    """Escreve um ficheiro, com nova tentativa em caso de conflito.

    Com várias bancas a correr ao mesmo tempo, dois portáteis publicam no
    mesmo ramo com segundos de diferença e o segundo apanha um 409: a API de
    conteúdos escreve um commit por ficheiro e recusa quando a cabeça do ramo
    mudou entretanto. Não é um erro do operador — é uma corrida, e resolve-se
    tentando outra vez.
    """
    import time

    encoded = urllib.parse.quote(path)
    body = {
        "message": message,
        "content": base64.b64encode(content).decode("ascii"),
        "branch": target.branch,
    }
    for attempt in range(attempts):
        try:
            _github_request(
                target,
                "PUT",
                f"/repos/{target.owner}/{target.repo}/contents/{encoded}",
                body,
            )
            return
        except PublishError as exc:
            conflict = "409" in str(exc) or "422" in str(exc)
            if not conflict or attempt == attempts - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def _ensure_nojekyll(target: GitHubTarget) -> None:
    """Sem isto o Pages passa o site pelo Jekyll e ignora o que começa por ``_``.

    Os relatórios não começam por ``_``, mas o ficheiro custa um pedido uma
    única vez e evita uma classe inteira de surpresas.
    """
    try:
        _github_request(
            target,
            "GET",
            f"/repos/{target.owner}/{target.repo}/contents/.nojekyll"
            f"?ref={urllib.parse.quote(target.branch)}",
        )
    except PublishError:
        try:
            _put_file(target, ".nojekyll", b"", "Serve os ficheiros tal como estão")
        except PublishError:
            pass


def wait_until_live(url: str, timeout_s: float = 90.0, poll_s: float = 3.0) -> bool:
    """Espera que o Pages sirva o endereço. Devolve se chegou a servir.

    O GitHub Pages não publica no instante do commit: constrói o site
    primeiro, o que leva de dez segundos a um minuto (mais na primeira vez).
    O email chega ao telemóvel da pessoa em segundos, portanto sem esta
    espera havia uma janela em que o link dava 404 — e um 404 no primeiro
    clique é pior do que o email chegar meio minuto mais tarde.

    Não levanta se esgotar: o link vai ficar bom de qualquer forma, e é
    melhor enviar com um aviso no registo do que não enviar.
    """
    import time

    deadline = time.monotonic() + timeout_s
    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": "mantra-eeg"}
    )
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(poll_s)
    return False


def publish_to_github(target: GitHubTarget, path: Path, code: str) -> str:
    """Envia a página e devolve o endereço público."""
    if not target.configured:
        raise PublishError(
            "faltam dados do GitHub: publish.github precisa de owner, repo e "
            "token."
        )
    _ensure_nojekyll(target)
    name = unguessable_name(code)
    folder = target.folder.strip("/")
    remote = f"{folder}/{name}" if folder else name
    _put_file(target, remote, Path(path).read_bytes(), f"Relatório {code}")
    return target.url_for(remote)
