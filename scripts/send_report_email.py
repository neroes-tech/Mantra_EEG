"""Publica o relatorio de uma sessao e envia-o por email.

O mesmo caminho que a banca corre quando o participante carrega em Enviar,
mas isolado, para se poder testar sem gravar nada.

    python scripts/send_report_email.py --to duarte@neroes.tech --name Duarte
    python scripts/send_report_email.py --to ... --name ... --dry-run
    python scripts/send_report_email.py --to ... --name ... --sign-in

--dry-run escreve o email num ficheiro .eml em vez de o enviar. Abre com duplo
clique em qualquer cliente e mostra exactamente o que a pessoa vai receber.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mantraeeg.config import load_config  # noqa: E402
from mantraeeg.drive import DriveClient, DriveError  # noqa: E402
from mantraeeg.mail import (  # noqa: E402
    GmailSender,
    MailError,
    _build_mime,
    compose,
)
from mantraeeg.publish import (  # noqa: E402
    GitHubTarget,
    PublishError,
    publish_to_github,
)
from mantraeeg.report.html import render  # noqa: E402


def setup_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def latest_report(cfg) -> Path:
    """A sessao mais recente que ja tem report.json."""
    roots = [cfg.paths.sessions_dir / cfg.paths.drive_folder_name, cfg.paths.sessions_dir]
    found: list[Path] = []
    for root in roots:
        if root.exists():
            found += [p for p in root.iterdir() if (p / "report.json").exists()]
    if not found:
        raise SystemExit(
            "nenhuma sessao com report.json. Corre primeiro:\n"
            "  python scripts/build_report.py --silent"
        )
    return sorted(found, key=lambda p: p.stat().st_mtime)[-1]


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description="Envio do relatorio por email")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--session", default=None)
    ap.add_argument("--to", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--sign-in", action="store_true", help="forcar novo login")
    ap.add_argument("--dry-run", action="store_true", help="escrever .eml, nao enviar")
    args = ap.parse_args()

    cfg = load_config(args.config)
    session = Path(args.session) if args.session else latest_report(cfg)
    data = json.loads((session / "report.json").read_text(encoding="utf-8"))
    code = data["meta"]["code"]

    report = session / "relatorio.html"
    report.write_text(render(data), encoding="utf-8")
    print(f"sessao : {session}")
    print(f"pagina : {report}  ({report.stat().st_size // 1024} KB)")

    email_cfg = cfg.ui.email
    drive: DriveClient | None = None
    link = ""
    mode = str(email_cfg.get("link_mode", "drive"))

    base = str(email_cfg.get("public_base_url", "")).strip().rstrip("/")
    if mode == "github_pages":
        target = GitHubTarget(
            owner=cfg.publish.owner,
            repo=cfg.publish.repo,
            branch=cfg.publish.branch,
            folder=cfg.publish.folder,
            token=cfg.publish.token,
            base_url=cfg.publish.base_url,
        )
        if args.dry_run:
            link = target.url_for(f"{target.folder}/{code}-EXEMPLO.html")
            print("(dry-run: link de exemplo, nada foi publicado)")
        else:
            link = publish_to_github(target, report, code)
            print(f"link   : {link}")
    elif mode == "base_url" and base:
        link = f"{base}/{report.name}"
    else:
        drive = DriveClient(
            client_id=cfg.drive.client_id,
            client_secret=cfg.drive.client_secret,
            token_path=cfg.drive.token_file,
            scope=cfg.drive.scope,
        )
        if args.sign_in or not drive.signed_in:
            print("a abrir o browser para iniciar sessao…")
            print(f"conta: {drive.sign_in()}")
        if args.dry_run:
            link = "https://drive.google.com/file/d/EXEMPLO/view"
            print("(dry-run: link de exemplo, nada foi enviado para o Drive)")
        else:
            file_id = drive.upload(
                report, cfg.drive.folder_id, name=f"{code}.html", mime="text/html"
            )
            link = drive.share_with_link(file_id)
            print(f"link   : {link}")

    message = compose(
        to=args.to,
        name=args.name,
        link=link,
        template=str(email_cfg.get("body", "")),
        subject=str(email_cfg.get("subject", "Relatorio")),
        logo=cfg.paths.logo_lockup,
        attachments=(
            (report,)
            if str(email_cfg.get("attach_report", "auto")).lower()
            in ("true", "sim", "1")
            or (
                str(email_cfg.get("attach_report", "auto")).lower() == "auto"
                and mode == "drive"
            )
            else ()
        ),
    )

    if args.dry_run:
        out = session / f"email_{datetime.now():%H%M%S}.eml"
        mime = _build_mime(message, "banca@neroes.tech", str(email_cfg["sender_name"]))
        out.write_bytes(mime.as_bytes())
        print(f"\n--- texto ---\n{message.text}")
        print(f"\nescrito: {out}")
        return 0

    if drive is None:
        drive = DriveClient(
            client_id=cfg.drive.client_id,
            client_secret=cfg.drive.client_secret,
            token_path=cfg.drive.token_file,
            scope=cfg.drive.scope,
        )
        if not drive.signed_in:
            print("a abrir o browser para iniciar sessao…")
            print(f"conta: {drive.sign_in()}")
    try:
        sent = GmailSender(drive, str(email_cfg["sender_name"])).send(message)
    except (MailError, DriveError, PublishError) as exc:
        print(f"\nFALHOU: {exc}")
        return 1
    print(f"\nenviado para {args.to} (id {sent})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
