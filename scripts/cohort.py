"""Analise de coorte: junta TODAS as sessoes da Drive e publica uma pagina.

    python scripts/cohort.py

E so isto. Descarrega o que ainda nao tem, analisa cada sessao, junta as que
tem sinal utilizavel, desenha a pagina e publica-a no GitHub Pages. No fim
imprime o link.

    python scripts/cohort.py --min-coverage 0.30   # mais exigente
    python scripts/cohort.py --local               # nao publica, so escreve
    python scripts/cohort.py --refresh             # volta a descarregar tudo

As gravacoes ficam em cache em data/cohort/, portanto a segunda corrida e
muito mais rapida.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.parse
import urllib.request
import zipfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from mantraeeg.analysis import analyse  # noqa: E402
from mantraeeg.cohort import aggregate, summarise_session  # noqa: E402
from mantraeeg.config import load_config  # noqa: E402
from mantraeeg.drive import DriveClient  # noqa: E402
from mantraeeg.publish import (  # noqa: E402
    GitHubTarget,
    PublishError,
    publish_to_github,
    wait_until_live,
)
from mantraeeg.report.cohort_html import render  # noqa: E402
from mantraeeg.sources.base import ACCEL_SLICE, EEG_SLICE, GYRO_SLICE  # noqa: E402


def setup_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def read_phases(session: pathlib.Path, total_s: float, meta: dict) -> list:
    """As fases da sessao, preferindo events.json a raw_meta.json.

    O raw_meta de gravacoes feitas antes de 11/09 pode trazer as fases de
    VARIAS sessoes: o historico do controlador nao era limpo entre sessoes, e
    como a base de tempo reinicia com cada ficheiro, saiam janelas sobrepostas
    todas a comecar em zero. O events.json e escrito por gravacao e nunca teve
    esse problema, portanto e a fonte de confianca.
    """
    events_path = session / "events.json"
    if events_path.exists():
        events = json.loads(events_path.read_text(encoding="utf-8"))
        ordered = sorted(
            [e for e in events if e.get("kind") == "phase"], key=lambda e: e["t_s"]
        )
        out = []
        for i, event in enumerate(ordered):
            end = ordered[i + 1]["t_s"] if i + 1 < len(ordered) else total_s
            if end > event["t_s"]:
                out.append((event["detail"], float(event["t_s"]), float(end)))
        if out:
            return out
    return [
        (p["phase"], float(p["start_s"]), float(p["end_s"]))
        for p in meta.get("phases", [])
    ]


def fetch(cfg, folder: pathlib.Path, refresh: bool) -> list[pathlib.Path]:
    """Traz da Drive o que ainda nao esta em cache."""
    folder.mkdir(parents=True, exist_ok=True)
    drive = DriveClient(
        cfg.drive.client_id, cfg.drive.client_secret, cfg.drive.token_file,
        scope=cfg.drive.scope,
    )
    if not drive.signed_in:
        raise SystemExit(
            "sem sessao iniciada no Google. Corre a aplicacao, abre Ctrl+I e "
            "inicia sessao — o token fica guardado."
        )
    token = drive.access_token()
    params = urllib.parse.urlencode({
        "q": f"'{cfg.drive.folder_id}' in parents and trashed=false",
        "fields": "files(id,name,size,createdTime)",
        "pageSize": "500",
        "orderBy": "createdTime",
    })
    request = urllib.request.Request(
        "https://www.googleapis.com/drive/v3/files?" + params,
        headers={"Authorization": f"Bearer {token}"},
    )
    files = [
        f
        for f in json.loads(urllib.request.urlopen(request, timeout=60).read())["files"]
        if f["name"].endswith(".zip")
    ]
    print(f"{len(files)} gravacoes na Drive")

    out = []
    for entry in files:
        archive = folder / entry["name"]
        session = folder / archive.stem
        if refresh or not archive.exists():
            r = urllib.request.Request(
                f"https://www.googleapis.com/drive/v3/files/{entry['id']}?alt=media",
                headers={"Authorization": f"Bearer {token}"},
            )
            archive.write_bytes(urllib.request.urlopen(r, timeout=900).read())
            print(f"  descarregada {entry['name']}")
        if refresh or not session.exists():
            with zipfile.ZipFile(archive) as z:
                z.extractall(folder)
        out.append(session)
    return out


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description="Analise de coorte")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--min-coverage", type=float, default=None)
    ap.add_argument("--local", action="store_true", help="nao publicar")
    ap.add_argument("--refresh", action="store_true", help="descarregar tudo de novo")
    ap.add_argument("--title", default="O cérebro a ouvir mantras")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cache = cfg.paths.sessions_dir.parent / "cohort"
    started = time.time()
    sessions = fetch(cfg, cache, args.refresh)

    print("\na analisar:")
    entries = []
    for session in sorted(sessions):
        meta_path = session / "raw_meta.json"
        if not meta_path.exists():
            print(f"  {session.name:<26} sem raw_meta.json")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        raw = np.load(session / "raw.npy")
        phases = read_phases(session, raw.shape[1] / cfg.device.sfreq_nominal, meta)
        if not {"CALIBRATION", "MANTRA"} <= {n for n, _, _ in phases}:
            print(f"  {session.name:<26} sem as duas fases")
            continue
        try:
            result = analyse(
                raw[EEG_SLICE].astype(float), cfg, phases,
                accel=raw[ACCEL_SLICE], gyro=raw[GYRO_SLICE],
            )
        except Exception as exc:
            print(f"  {session.name:<26} analise falhou: {exc}")
            continue
        entry = summarise_session(
            session.name,
            session.name.split("_")[0],
            (meta.get("plan") or {}).get("mantra_id", ""),
            raw.shape[1] / cfg.device.sfreq_nominal,
            result,
        )
        entries.append(entry)
        print(f"  {session.name:<26} cobertura {entry.coverage * 100:3.0f}%")

    if not entries:
        raise SystemExit("nenhuma sessao analisavel")

    cohort = aggregate(entries, cfg, args.min_coverage)
    print(
        f"\n{len(cohort.included)} de {len(cohort.sessions)} sessoes incluidas "
        f"(cobertura >= {cohort.min_coverage:.0%})"
    )
    print(f"  {'biomarcador':<26}{'n':>3}{'variacao':>10}{'mesmo sentido':>16}")
    for a in cohort.aggregates:
        percent = (
            f"{a.median_percent:+.0f}%"
            if np.isfinite(a.median_percent)
            else f"{a.median_std:+.2f}"
        )
        mark = "  <- consistente" if a.consistent else ""
        print(
            f"  {a.marker.friendly[:26]:<26}{a.n_sessions:>3}{percent:>10}"
            f"{max(a.n_up, a.n_down)} de {a.n_sessions:<10}{mark}"
        )

    page = cache / "coorte.html"
    page.write_text(render(cohort, cfg, args.title), encoding="utf-8")
    print(f"\npagina: {page}  ({page.stat().st_size // 1024} KB)")

    if args.local:
        print("(--local: nao publicado)")
        return 0

    target = GitHubTarget(
        owner=cfg.publish.owner,
        repo=cfg.publish.repo,
        branch=cfg.publish.branch,
        folder=cfg.publish.folder,
        token=cfg.publish.token,
        base_url=cfg.publish.base_url,
    )
    if not target.configured:
        print(
            "\npublicacao desligada: falta publish.github na config. A pagina\n"
            f"esta escrita em {page} e abre com duplo clique."
        )
        return 0
    try:
        link = publish_to_github(target, page, "coorte")
    except PublishError as exc:
        print(f"\nnao publiquei: {exc}")
        return 1
    print("a aguardar que o site fique no ar…")
    live = wait_until_live(link)
    print(f"\n{'=' * 68}")
    print(f"  {link}")
    print(f"{'=' * 68}")
    if not live:
        print("  (o GitHub Pages ainda esta a construir; abre daqui a um minuto)")
    print(f"\n{time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
