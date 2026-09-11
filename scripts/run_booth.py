"""Entrypoint da banca. Uma janela, um monitor.

    python scripts/run_booth.py
    python scripts/run_booth.py --windowed
    python scripts/run_booth.py --source replay --file tests/fixtures/unicorn_sample.csv

A aplicacao arranca SEM dispositivo: nao precisa de headset ligado para abrir.
O equipamento e procurado quando se inicia uma sessao, e libertado no fim, o que
lhe permite ficar horas em espera com o headset desligado.

Ctrl+I abre o painel do investigador (estado do dispositivo, sinal, duracoes).
"""

from __future__ import annotations

import argparse
import signal as os_signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PyQt6 import QtCore, QtWidgets  # noqa: E402

from mantraeeg.applog import install_excepthook, log  # noqa: E402
from mantraeeg.config import load_config  # noqa: E402
from mantraeeg.ui.app import MainWindow  # noqa: E402


def selftest(config_path: str) -> int:
    """Verifica que esta maquina consegue correr uma sessao. Nao abre janela.

    Existe para o dia do evento: correr isto num portatil novo diz em cinco
    segundos se falta alguma coisa, em vez de se descobrir com uma pessoa
    sentada na cadeira e o headset na cabeca.
    """
    from mantraeeg.apppaths import app_root, frozen

    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        mark = "[ ok ]" if passed else "[FALHA]"
        print(f"{mark} {label}{('  ' + detail) if detail else ''}")

    print(f"raiz da aplicacao: {app_root()}")
    print(f"empacotado       : {'sim' if frozen() else 'nao'}" + chr(10))

    try:
        cfg = load_config(config_path)
    except Exception as exc:
        check("configuracao", False, str(exc))
        return 1
    check("configuracao", True, str(cfg.source_path))

    for entry in cfg.paths.mantras:
        check(f"mantra {entry.id}", entry.file.exists(), str(entry.file))
    check("logotipo", cfg.paths.logo_lockup.exists(), str(cfg.paths.logo_lockup))

    try:
        from PyQt6.QtMultimedia import QMediaPlayer  # noqa: F401
        from PyQt6.QtMultimediaWidgets import QVideoWidget  # noqa: F401

        check("Qt Multimedia", True, "video e audio disponiveis")
    except Exception as exc:
        check("Qt Multimedia", False, f"{exc} — o mantra nao toca")

    try:
        from mantraeeg.sources.unicorn_dll import list_available_serials

        serials = list_available_serials()
        check(
            "Unicorn emparelhado",
            bool(serials),
            ", ".join(serials) if serials else "nenhum dispositivo UN-* visivel",
        )
    except Exception as exc:
        print(f"[aviso] nao consegui listar dispositivos pela DLL: {exc}")

    try:
        import brainflow  # noqa: F401

        check("BrainFlow", True, brainflow.__file__)
    except Exception as exc:
        check("BrainFlow", False, str(exc))

    # Qual a conta Google em uso. Numa banca com varios membros da equipa, e
    # a pergunta que decide de quem parte o email e para onde vao as sessoes.
    if cfg.drive.enabled:
        from mantraeeg.drive import DriveClient

        client = DriveClient(
            client_id=cfg.drive.client_id,
            client_secret=cfg.drive.client_secret,
            token_path=cfg.drive.token_file,
            scope=cfg.drive.scope,
        )
        check(
            "sessao Google",
            client.signed_in,
            client.account
            if client.signed_in
            else "sem sessao: abrir Ctrl+I e iniciar sessao antes do evento",
        )

    mode = str(cfg.ui.email.get("link_mode", "drive"))
    if mode == "github_pages":
        from mantraeeg.publish import GitHubTarget, PublishError, _github_request

        target = GitHubTarget(
            owner=cfg.publish.owner,
            repo=cfg.publish.repo,
            branch=cfg.publish.branch,
            folder=cfg.publish.folder,
            token=cfg.publish.token,
            base_url=cfg.publish.base_url,
        )
        if not target.configured:
            check(
                "alojamento do relatorio",
                False,
                "link_mode e github_pages mas falta owner, repo ou token",
            )
        else:
            try:
                info = _github_request(
                    target, "GET", f"/repos/{target.owner}/{target.repo}"
                )
                check(
                    "alojamento do relatorio",
                    True,
                    f"{target.owner}/{target.repo} -> {target.url_for('')}",
                )
                if info.get("private"):
                    check(
                        "repositorio publico",
                        False,
                        "o repositorio e privado; o Pages gratuito precisa dele publico",
                    )
            except PublishError as exc:
                check("alojamento do relatorio", False, str(exc))
    else:
        print(
            f"[aviso] link_mode='{mode}': o link do email DESCARREGA o "
            f"relatorio em vez de o abrir. Ver o README, 'O link do relatorio'."
        )

    sessions = cfg.paths.sessions_dir
    try:
        sessions.mkdir(parents=True, exist_ok=True)
        probe = sessions / ".escrita"
        probe.write_text("x", encoding="utf-8")
        probe.unlink()
        check("pasta de gravacoes", True, str(sessions))
    except Exception as exc:
        check("pasta de gravacoes", False, f"{sessions}: {exc}")

    print("\n" + ("tudo pronto." if ok else "ha coisas por resolver acima."))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Estacao de demonstracao mantra-eeg")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument(
        "--source", choices=("brainflow", "unicorn_dll", "lsl", "replay"), default=None
    )
    ap.add_argument("--file", default=None, help="gravacao, para --source replay")
    ap.add_argument("--windowed", action="store_true", help="nao usar ecra inteiro")
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="verificar a instalacao e sair, sem abrir a janela",
    )
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # Antes de tudo: sem consola nao ha para onde imprimir, e o CPython
    # descarta em silencio o que se escreve para um sys.stdout que e None.
    # Tudo o que escapa vai para data/app.log, ao lado do executavel.
    install_excepthook()
    log(f"arranque: selftest={args.selftest} source={args.source}")

    if args.selftest:
        return selftest(args.config)

    cfg = load_config(args.config)
    if args.source == "replay":
        cfg.device.replay["realtime"] = True

    app = QtWidgets.QApplication(sys.argv)
    os_signal.signal(os_signal.SIGINT, os_signal.SIG_DFL)

    window = MainWindow(
        cfg,
        source_override=args.source,
        replay_file=args.file,
        fullscreen=not args.windowed,
    )
    window.show_window()

    keepalive = QtCore.QTimer()
    keepalive.start(200)
    keepalive.timeout.connect(lambda: None)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
