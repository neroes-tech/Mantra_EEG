"""Publicação do relatório. Sem rede: só os nomes e os endereços."""

from __future__ import annotations

import re

from mantraeeg.publish import GitHubTarget, unguessable_name


def _target(**kwargs) -> GitHubTarget:
    base = dict(
        owner="neroes", repo="relatorios", branch="main", folder="r", token="x"
    )
    base.update(kwargs)
    return GitHubTarget(**base)  # type: ignore[arg-type]


def test_filename_is_not_guessable_from_the_session_code():
    """Um relatório num alojamento público não pode ter endereço adivinhável.

    O código deriva da hora da sessão: quem esteve na banca sabe-a com
    precisão de segundos, e podia percorrer os códigos dos vizinhos.
    """
    code = "FBE-2026-110628"
    names = {unguessable_name(code) for _ in range(50)}
    assert len(names) == 50, "o sufixo repetiu-se"
    for name in names:
        assert re.fullmatch(rf"{re.escape(code)}-[0-9a-f]{{8}}\.html", name), name


def test_url_uses_github_io_by_default():
    assert _target().url_for("r/x.html") == "https://neroes.github.io/relatorios/r/x.html"


def test_custom_domain_wins_and_normalises_the_slash():
    target = _target(base_url="https://relatorios.neroes.tech/")
    assert target.url_for("r/x.html") == "https://relatorios.neroes.tech/r/x.html"


def test_configured_needs_owner_repo_and_token():
    assert _target().configured
    assert not _target(token="").configured
    assert not _target(owner="").configured
    assert not _target(repo="").configured


def test_token_never_comes_from_the_yaml(cfg):
    """O YAML viaja nos pacotes que se copiam para os portáteis do evento.

    O token vem da variável de ambiente ou de um ficheiro em ``data/``, que
    está no ``.gitignore`` e que o ``build_exe.py`` não copia.
    """
    import pathlib

    import yaml

    raw = yaml.safe_load(pathlib.Path(cfg.source_path).read_text(encoding="utf-8"))
    github = (raw.get("publish") or {}).get("github") or {}
    assert "token" not in github, "não escrever o token no YAML"
    assert github.get("token_env")
