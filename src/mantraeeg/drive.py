"""Envio para o Google Drive, só com a biblioteca padrão.

Sem ``google-api-python-client``: essa cadeia de dependências é pesada, arrasta
``grpc`` e é das piores de empacotar com o PyInstaller. O que é preciso aqui —
OAuth de aplicação instalada e um upload multipart — cabe em ``urllib`` e
``http.server``, e resulta num executável que não tem de resolver nada em
runtime.

**A API do Drive exige sempre autenticação.** Uma pasta partilhada com "qualquer
pessoa com o link pode editar" não permite escrever sem credenciais: o link dá o
identificador da pasta, não o direito de escrita programática. É por isso que há
um passo de início de sessão.

Dois caminhos:

``oauth``
    Início de sessão com a conta Google, uma vez por computador. Abre o browser,
    a pessoa escolhe a conta Neroes, e o *refresh token* fica guardado — nas
    vezes seguintes não há ecrã de login. É o que foi pedido.

``service_account``
    Uma chave JSON, partilhada a pasta com o email da conta de serviço. **Zero**
    interação. Para um quiosque desatendido num festival é mais robusto: não há
    login a expirar a meio do evento nem browser a abrir por cima da demo.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

AUTH_URI = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"
UPLOAD_URI = "https://www.googleapis.com/upload/drive/v3/files"
FILES_URI = "https://www.googleapis.com/drive/v3/files"

#: Só ficheiros criados por esta aplicação. É o âmbito mínimo que serve.
DEFAULT_SCOPE = "https://www.googleapis.com/auth/drive.file"


class DriveError(RuntimeError):
    """Falha de autenticação ou de envio. A mensagem é para o operador."""


@dataclass
class Credentials:
    access_token: str
    refresh_token: str = ""
    expires_at: float = 0.0
    account: str = ""

    @property
    def expired(self) -> bool:
        return time.time() >= self.expires_at - 60.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "account": self.account,
        }


def _post_form(url: str, fields: dict[str, str]) -> dict[str, Any]:
    data = urllib.parse.urlencode(fields).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise DriveError(f"{url} devolveu {exc.code}: {detail}") from exc
    except OSError as exc:
        raise DriveError(f"sem rede ao contactar {url}: {exc}") from exc


class _RedirectHandler(BaseHTTPRequestHandler):
    """Recebe o ``code`` que o Google devolve ao endereço de loopback."""

    code: str | None = None
    error: str | None = None

    def do_GET(self) -> None:  # noqa: N802
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        _RedirectHandler.code = (query.get("code") or [None])[0]
        _RedirectHandler.error = (query.get("error") or [None])[0]
        body = (
            "<html><body style='font-family:sans-serif;background:#07080b;"
            "color:#eef1f7;text-align:center;padding-top:80px'>"
            "<h2>Sessão iniciada</h2><p>Pode fechar esta janela e voltar à "
            "aplicação.</p></body></html>"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:  # silêncio
        pass


class DriveClient:
    """Autentica e envia ficheiros para uma pasta do Drive."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        token_path: Path,
        scope: str = DEFAULT_SCOPE,
        service_account_file: Path | None = None,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._token_path = Path(token_path)
        self._scope = scope
        self._service_account_file = service_account_file
        self._creds: Credentials | None = self._load()

    # -- estado ---------------------------------------------------------------- #
    @property
    def signed_in(self) -> bool:
        return self._creds is not None

    @property
    def account(self) -> str:
        return self._creds.account if self._creds else ""

    @property
    def configured(self) -> bool:
        """Há credenciais suficientes para sequer tentar."""
        if self._service_account_file and self._service_account_file.exists():
            return True
        return bool(self._client_id)

    def sign_out(self) -> None:
        self._creds = None
        self._token_path.unlink(missing_ok=True)

    # -- persistência ------------------------------------------------------------ #
    def _load(self) -> Credentials | None:
        if not self._token_path.exists():
            return None
        try:
            payload = json.loads(self._token_path.read_text(encoding="utf-8"))
            return Credentials(**payload)
        except Exception:
            return None

    def _save(self) -> None:
        if self._creds is None:
            return
        self._token_path.parent.mkdir(parents=True, exist_ok=True)
        self._token_path.write_text(
            json.dumps(self._creds.as_dict(), indent=2), encoding="utf-8"
        )

    # -- autenticação -------------------------------------------------------------- #
    def sign_in(self, timeout_s: float = 180.0) -> str:
        """Fluxo de aplicação instalada com PKCE. Devolve a conta autenticada.

        Abre o browser e espera pelo redirecionamento para ``localhost``. O
        *refresh token* é guardado, portanto isto acontece uma vez por
        computador.
        """
        if not self._client_id:
            raise DriveError(
                "falta o client_id do Google. Criar um cliente OAuth do tipo "
                "'Aplicação de ambiente de trabalho' na Google Cloud Console e "
                "pô-lo em drive.client_id."
            )

        verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).decode().rstrip("=")
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .decode()
            .rstrip("=")
        )

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        redirect = f"http://127.0.0.1:{port}"

        params = {
            "client_id": self._client_id,
            "redirect_uri": redirect,
            "response_type": "code",
            "scope": self._scope,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "access_type": "offline",
            "prompt": "consent",
        }
        _RedirectHandler.code = None
        _RedirectHandler.error = None
        server = HTTPServer(("127.0.0.1", port), _RedirectHandler)
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        webbrowser.open(f"{AUTH_URI}?{urllib.parse.urlencode(params)}")

        deadline = time.time() + timeout_s
        while thread.is_alive() and time.time() < deadline:
            time.sleep(0.2)
        server.server_close()

        if _RedirectHandler.error:
            raise DriveError(f"o Google recusou: {_RedirectHandler.error}")
        if not _RedirectHandler.code:
            raise DriveError("o início de sessão não foi concluído a tempo.")

        payload = _post_form(
            TOKEN_URI,
            {
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "code": _RedirectHandler.code,
                "code_verifier": verifier,
                "grant_type": "authorization_code",
                "redirect_uri": redirect,
            },
        )
        self._creds = Credentials(
            access_token=payload["access_token"],
            refresh_token=payload.get("refresh_token", ""),
            expires_at=time.time() + float(payload.get("expires_in", 3600)),
        )
        self._creds.account = self._whoami()
        self._save()
        return self._creds.account

    def _refresh(self) -> None:
        if self._creds is None or not self._creds.refresh_token:
            raise DriveError("sessão expirada e sem refresh token: iniciar sessão.")
        payload = _post_form(
            TOKEN_URI,
            {
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "refresh_token": self._creds.refresh_token,
                "grant_type": "refresh_token",
            },
        )
        self._creds.access_token = payload["access_token"]
        self._creds.expires_at = time.time() + float(payload.get("expires_in", 3600))
        self._save()

    def _service_account_token(self) -> str:
        """JWT assinado -> token. Precisa de ``cryptography``, se for usado."""
        raise DriveError(
            "conta de serviço ainda não implementada; usar o início de sessão "
            "com a conta Google."
        )

    def _token(self) -> str:
        if self._service_account_file and self._service_account_file.exists():
            return self._service_account_token()
        if self._creds is None:
            raise DriveError("sem sessão iniciada no Google Drive.")
        if self._creds.expired:
            self._refresh()
        return self._creds.access_token

    def access_token(self) -> str:
        """Token válido, renovado se preciso. É o que o envio de email usa.

        A mesma credencial serve o Drive e o Gmail: o operador faz login uma
        vez e a banca não guarda mais nenhuma palavra-passe.
        """
        return self._token()

    def _whoami(self) -> str:
        try:
            request = urllib.request.Request(
                "https://www.googleapis.com/drive/v3/about?fields=user(emailAddress)",
                headers={"Authorization": f"Bearer {self._creds.access_token}"},
            )
            with urllib.request.urlopen(request, timeout=20) as response:
                data = json.loads(response.read().decode())
            return data.get("user", {}).get("emailAddress", "")
        except Exception:
            return ""

    # -- envio ----------------------------------------------------------------------- #
    def upload(
        self,
        path: Path,
        folder_id: str,
        name: str | None = None,
        mime: str | None = None,
    ) -> str:
        """Envia um ficheiro para a pasta. Devolve o id do ficheiro no Drive.

        ``mime`` por omissão é ``application/zip``, que é o que a fila de
        sessões envia. O relatório em HTML precisa do seu, senão o Drive
        guarda-o como um zip e o browser recusa-se a abri-lo.
        """
        path = Path(path)
        if not path.exists():
            raise DriveError(f"ficheiro não encontrado: {path}")
        if not folder_id:
            raise DriveError(
                "falta o id da pasta do Drive. É a parte final do link de "
                "partilha: .../folders/<ID>."
            )

        token = self._token()
        content_type = mime or "application/zip"
        metadata: dict[str, Any] = {"name": name or path.name, "parents": [folder_id]}
        if mime:
            metadata["mimeType"] = mime
        boundary = "----mantraeeg" + secrets.token_hex(12)
        body = b"".join(
            [
                f"--{boundary}\r\n".encode(),
                b"Content-Type: application/json; charset=UTF-8\r\n\r\n",
                json.dumps(metadata).encode(),
                f"\r\n--{boundary}\r\n".encode(),
                f"Content-Type: {content_type}\r\n\r\n".encode(),
                path.read_bytes(),
                f"\r\n--{boundary}--\r\n".encode(),
            ]
        )
        request = urllib.request.Request(
            f"{UPLOAD_URI}?uploadType=multipart&supportsAllDrives=true",
            data=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": f"multipart/related; boundary={boundary}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.loads(response.read().decode()).get("id", "")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            if exc.code in (401, 403):
                raise DriveError(
                    f"o Google recusou o envio ({exc.code}). Confirme que a conta "
                    f"tem permissão de edição na pasta. Detalhe: {detail}"
                ) from exc
            raise DriveError(f"envio falhou ({exc.code}): {detail}") from exc
        except OSError as exc:
            raise DriveError(f"sem rede: {exc}") from exc


    def share_with_link(self, file_id: str) -> str:
        """Dá acesso de leitura a quem tiver o link, e devolve o link.

        Um relatório enviado por email tem de abrir sem a pessoa iniciar
        sessão em nada. Sem esta permissão o link pede autenticação e o
        participante vê um ecrã de erro do Google.
        """
        if not file_id:
            raise DriveError("sem id de ficheiro para partilhar.")
        token = self._token()
        request = urllib.request.Request(
            f"{FILES_URI}/{file_id}/permissions?supportsAllDrives=true",
            data=json.dumps({"role": "reader", "type": "anyone"}).encode(),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        try:
            urllib.request.urlopen(request, timeout=60).close()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise DriveError(
                f"não consegui tornar o ficheiro público ({exc.code}): {detail}"
            ) from exc
        except OSError as exc:
            raise DriveError(f"sem rede: {exc}") from exc
        return web_link(file_id)


def web_link(file_id: str) -> str:
    """O link de visualização de um ficheiro do Drive."""
    return f"https://drive.google.com/file/d/{file_id}/view"


def folder_id_from_link(link: str) -> str:
    """Extrai o id de um link de partilha do Drive.

    Aceita ``https://drive.google.com/drive/folders/<ID>?usp=sharing`` e também
    o id já isolado.
    """
    text = link.strip()
    if "/folders/" in text:
        text = text.split("/folders/", 1)[1]
    return text.split("?")[0].split("/")[0].strip()
