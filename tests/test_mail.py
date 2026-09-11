"""O email que sai da banca. Sem rede: só a composição e o MIME."""

from __future__ import annotations

from email import message_from_bytes

import pytest

from mantraeeg.mail import _build_mime, compose, valid_address

LINK = "https://drive.google.com/file/d/abc123/view"


@pytest.fixture()
def template(cfg) -> str:
    return str(cfg.ui.email["body"])


def test_addresses_are_filtered_before_sending():
    assert valid_address("duarte@neroes.tech")
    for bad in ("", "duarte", "duarte@", "@neroes.tech", "duarte@neroes", "a b@c.pt"):
        assert not valid_address(bad), bad


def test_name_and_link_are_substituted(cfg, template):
    message = compose(
        to="ana@exemplo.pt",
        name="Ana Maria Silva",
        link=LINK,
        template=template,
        subject=str(cfg.ui.email["subject"]),
    )
    # Só o primeiro nome: "Olá Ana Maria Silva," lê-se como um formulário.
    assert "Olá Ana," in message.text
    assert LINK in message.text
    assert "{nome}" not in message.text and "{link}" not in message.text
    # No HTML o link é um botão, não o URL cru.
    assert f'href="{LINK}"' in message.html
    assert "Abrir o meu relatório" in message.html


def test_arrows_become_a_list_in_html(cfg, template):
    message = compose(
        to="ana@exemplo.pt",
        name="Ana",
        link=LINK,
        template=template,
        subject="x",
    )
    assert message.html.count("<li") == 3
    # O texto simples mantém as setas — é o que os clientes sem HTML mostram.
    assert message.text.count("→") == 3


def test_mime_carries_both_versions_and_the_attachment(cfg, template, tmp_path):
    report = tmp_path / "FBE-2026-000000.html"
    report.write_text("<p>relatório</p>", encoding="utf-8")
    message = compose(
        to="ana@exemplo.pt",
        name="Ana",
        link=LINK,
        template=template,
        subject="Relatório",
        logo=cfg.paths.logo_lockup,
        attachments=(report,),
    )
    raw = _build_mime(message, "banca@neroes.tech", "Equipa Neroes").as_bytes()
    parsed = message_from_bytes(raw)

    assert parsed["To"] == "ana@exemplo.pt"
    assert parsed["From"] == "Equipa Neroes <banca@neroes.tech>"

    types = {part.get_content_type() for part in parsed.walk()}
    assert "text/plain" in types
    assert "text/html" in types

    names = [p.get_filename() for p in parsed.walk() if p.get_filename()]
    assert report.name in names


def test_unknown_name_does_not_leave_a_hole(cfg, template):
    message = compose(
        to="ana@exemplo.pt", name="  ", link=LINK, template=template, subject="x"
    )
    assert "{nome}" not in message.text
    assert message.text.startswith("Olá,")
