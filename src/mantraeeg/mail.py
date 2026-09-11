"""Envio do relatório por email, pela conta Google já autenticada.

Reaproveita a sessão do Drive: o operador faz login uma vez no painel e a
mesma credencial serve para guardar a gravação e para enviar o email. Sem
servidor SMTP, sem palavra-passe de aplicação guardada no disco da banca, sem
dependências novas — a API do Gmail é um POST com o RFC 822 em base64url, e
``email.message`` é da biblioteca padrão.

**O âmbito ``gmail.send`` tem de estar em ``drive.scope``.** É o âmbito mais
estreito que a Google oferece para isto: permite enviar, não permite ler nada.
Acrescentá-lo invalida o token guardado — o primeiro arranque depois da
mudança pede login outra vez, uma vez por computador.

O corpo do email vive na configuração e não aqui: é texto de marketing, muda
sem código, e ninguém deve ter de abrir um ficheiro Python para corrigir uma
vírgula na comunicação com o participante.
"""

from __future__ import annotations

import base64
import html as html_mod
import json
import mimetypes
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path

from .drive import DriveClient, DriveError

SEND_URI = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
GMAIL_SCOPE = "https://www.googleapis.com/auth/gmail.send"

#: Um endereço tem de passar isto antes de a aplicação tentar enviar. Não é
#: validação de RFC — é o filtro que apanha o dedo enganado na banca.
ADDRESS_RE = re.compile(r"^[^@\s]+@[^@\s.]+(\.[^@\s.]+)+$")


class MailError(RuntimeError):
    """Falha no envio, com uma mensagem que o operador possa ler."""


@dataclass(frozen=True)
class Message:
    """Um email pronto a enviar."""

    to: str
    subject: str
    text: str
    html: str
    attachments: tuple[Path, ...] = ()
    inline_images: tuple[tuple[str, Path], ...] = ()


def valid_address(address: str) -> bool:
    return bool(ADDRESS_RE.match(address.strip()))


def _paragraphs_to_html(text: str) -> str:
    """O corpo em texto, convertido para HTML sem perder a estrutura.

    Uma só fonte para as duas versões: o texto é o original, e o HTML é
    derivado dele. Assim não há forma de as duas discordarem quando alguém
    editar a configuração.
    """
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    out = []
    for block in blocks:
        lines = block.splitlines()
        if all(line.lstrip().startswith("→") for line in lines):
            items = "".join(
                f'<li style="margin:0 0 10px;padding-left:2px;">'
                f"{html_mod.escape(line.lstrip()[1:].strip())}</li>"
                for line in lines
            )
            out.append(
                f'<ul style="margin:0 0 20px;padding-left:20px;color:#A9BFC5;'
                f'font-size:15px;line-height:1.6;">{items}</ul>'
            )
            continue
        body = "<br>".join(html_mod.escape(line) for line in lines)
        out.append(
            f'<p style="margin:0 0 18px;color:#A9BFC5;font-size:15px;'
            f'line-height:1.65;">{body}</p>'
        )
    return "".join(out)


def compose(
    *,
    to: str,
    name: str,
    link: str,
    template: str,
    subject: str,
    logo: Path | None = None,
    attachments: tuple[Path, ...] = (),
) -> Message:
    """Preenche o modelo com o nome e o link, e monta as duas versões.

    O link é escapado antes de entrar no HTML, e o nome também: são texto
    escrito por uma pessoa numa caixa de texto de um quiosque.
    """
    # Só o primeiro nome: "Olá Ana Maria Silva," lê-se como um formulário e
    # não como uma pessoa a falar com outra. Sem nome, a saudação fica sozinha
    # em vez de deixar um buraco ou um "Olá ,".
    first_name = name.strip().split(" ")[0] if name.strip() else ""
    text = template.replace("{nome}", first_name).replace("{link}", link)
    if not first_name:
        text = text.replace("Olá ,", "Olá,").replace("Olá , ", "Olá, ")

    body = _paragraphs_to_html(text)
    # O link no HTML é um botão, e o URL cru fica só na versão em texto.
    body = body.replace(
        html_mod.escape(link),
        f'<a href="{html_mod.escape(link, quote=True)}" '
        f'style="color:#43BEC3;text-decoration:none;font-weight:600;">'
        f"Abrir o meu relatório</a>",
    )

    inline: tuple[tuple[str, Path], ...] = ()
    logo_html = ""
    if logo is not None and logo.exists():
        inline = (("logo", logo),)
        logo_html = (
            '<img src="cid:logo" alt="neroes" '
            'style="height:34px;width:auto;display:block;margin-top:22px;">'
        )

    page = (
        '<div style="background:#0C1D24;padding:28px 16px;">'
        '<div style="max-width:560px;margin:0 auto;background:#152E38;'
        "border:1px solid rgba(169,191,197,0.16);border-radius:10px;"
        "padding:32px;font-family:'Sora','Segoe UI',system-ui,sans-serif;\">"
        f"{body}{logo_html}"
        "</div></div>"
    )
    return Message(
        to=to.strip(),
        subject=subject,
        text=text,
        html=page,
        attachments=attachments,
        inline_images=inline,
    )


def _build_mime(message: Message, sender: str, sender_name: str) -> EmailMessage:
    mail = EmailMessage()
    mail["To"] = message.to
    mail["From"] = f"{sender_name} <{sender}>" if sender_name else sender
    mail["Subject"] = message.subject
    mail.set_content(message.text)
    mail.add_alternative(message.html, subtype="html")

    # As imagens inline entram na parte HTML, não no topo: senão os clientes
    # mostram o logótipo como anexo solto em vez de o desenhar no corpo.
    html_part = mail.get_payload()[-1]
    for cid, path in message.inline_images:
        subtype = (mimetypes.guess_type(path.name)[0] or "image/png").split("/")[1]
        html_part.add_related(
            path.read_bytes(), maintype="image", subtype=subtype, cid=f"<{cid}>"
        )

    for path in message.attachments:
        if not path.exists():
            continue
        guessed = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        maintype, _, subtype = guessed.partition("/")
        mail.add_attachment(
            path.read_bytes(),
            maintype=maintype,
            subtype=subtype,
            filename=path.name,
        )
    return mail


class GmailSender:
    """Envia pela conta Google que já está autenticada para o Drive."""

    def __init__(self, drive: DriveClient, sender_name: str = "") -> None:
        self._drive = drive
        self._sender_name = sender_name

    @property
    def ready(self) -> bool:
        return self._drive.signed_in

    def send(self, message: Message) -> str:
        """Envia e devolve o id da mensagem. Levanta ``MailError`` se falhar."""
        if not valid_address(message.to):
            raise MailError(f"endereço inválido: {message.to!r}")
        if not self._drive.signed_in:
            raise MailError(
                "sem sessão iniciada no Google. Abra o painel (Ctrl+I) e "
                "inicie sessão antes de enviar."
            )
        try:
            token = self._drive.access_token()
            sender = self._drive.account
        except DriveError as exc:
            raise MailError(str(exc)) from exc

        mail = _build_mime(message, sender, self._sender_name)
        raw = base64.urlsafe_b64encode(mail.as_bytes()).decode("ascii")
        request = urllib.request.Request(
            SEND_URI,
            data=json.dumps({"raw": raw}).encode(),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode()).get("id", "")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            if "has not been used in project" in detail or "is disabled" in detail:
                # Acontece uma vez por projeto e a mensagem crua do Google
                # esconde o essencial no meio de um JSON. Isto e um clique na
                # consola, nao um problema de codigo.
                raise MailError(
                    "a Gmail API esta desativada no projeto Google Cloud. "
                    "Ative-a em console.cloud.google.com > APIs e Servicos > "
                    "Ativar API > Gmail API, no mesmo projeto do client_id, e "
                    "espere um minuto. O login nao precisa de ser repetido."
                ) from exc
            if exc.code in (401, 403):
                raise MailError(
                    f"o Google recusou o envio ({exc.code}). Falta o âmbito "
                    f"'{GMAIL_SCOPE}' em drive.scope, ou a sessão é de outra "
                    f"conta. Detalhe: {detail}"
                ) from exc
            raise MailError(f"envio falhou ({exc.code}): {detail}") from exc
        except OSError as exc:
            raise MailError(f"sem rede: {exc}") from exc
