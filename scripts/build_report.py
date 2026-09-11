"""Gera o relatorio do participante a partir de uma sessao gravada.

    python scripts/build_report.py --name "Duarte Rodrigues" --silent
    python scripts/build_report.py --session <pasta> --mantra "Sri Vitthala"

Escreve report.json (os dados) e report.html (a pagina) na pasta da sessao.
O JSON e a fonte tambem do corpo do email, para os dois nunca divergirem.

O mantra tem de ser dito explicitamente: --mantra <nome> ou --silent. Uma
sessao de controlo nao pode sair do relatorio com o nome de um mantra que nao
tocou.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mantraeeg.analysis import analyse, rank_by_variation  # noqa: E402
from mantraeeg.config import load_config  # noqa: E402
from mantraeeg.report.html import render  # noqa: E402
from mantraeeg.report.payload import ReportContext, build  # noqa: E402
from mantraeeg.sources.base import ACCEL_SLICE, EEG_SLICE, GYRO_SLICE  # noqa: E402


def setup_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def latest_session(cfg) -> Path:
    roots = [cfg.paths.sessions_dir / cfg.paths.drive_folder_name, cfg.paths.sessions_dir]
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            candidates += [p for p in root.iterdir() if p.is_dir() and (p / "raw.npy").exists()]
    if not candidates:
        raise SystemExit("nenhuma sessao encontrada em data/sessions")
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def read_phases(session: Path, total_s: float) -> list[tuple[str, float, float]]:
    events = json.loads((session / "events.json").read_text(encoding="utf-8"))
    ordered = sorted([e for e in events if e.get("kind") == "phase"], key=lambda e: e["t_s"])
    out = []
    for i, event in enumerate(ordered):
        end = ordered[i + 1]["t_s"] if i + 1 < len(ordered) else total_s
        if end > event["t_s"]:
            out.append((event["detail"], float(event["t_s"]), float(end)))
    return out


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description="Relatorio do participante")
    ap.add_argument("--config", default=None, help="default: o da propria sessao")
    ap.add_argument("--session", default=None)
    ap.add_argument("--name", default="Participante")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--mantra", default=None, help="nome do mantra ouvido")
    group.add_argument("--silent", action="store_true", help="sessao de controlo")
    ap.add_argument("--festival", default="Festival Bem Estar 2026 · neroes")
    args = ap.parse_args()

    meta_cfg = args.config
    session = Path(args.session) if args.session else None
    if session is None:
        session = latest_session(load_config(meta_cfg or "config/default.yaml"))
    meta = json.loads((session / "raw_meta.json").read_text(encoding="utf-8"))
    cfg = load_config(meta_cfg or meta.get("config_path") or "config/default.yaml")

    raw = np.load(session / "raw.npy")
    total_s = raw.shape[1] / cfg.device.sfreq_nominal
    phases = read_phases(session, total_s)

    print(f"sessao : {session}")
    print(f"duracao: {total_s:.1f} s")
    for name, start, end in phases:
        print(f"  {name:<12} {start:7.1f} -> {end:7.1f} s")

    result = analyse(
        raw[EEG_SLICE], cfg, phases, accel=raw[ACCEL_SLICE], gyro=raw[GYRO_SLICE]
    )
    for note in result.notes:
        print(f"  {note}")

    featured = rank_by_variation(result, cfg, limit=cfg.report.featured_count)
    print(f"\ndestaque: {featured}")

    started_raw = meta.get("created_utc")
    started = (
        datetime.fromisoformat(started_raw).astimezone()
        if started_raw
        else datetime.fromtimestamp(session.stat().st_mtime)
    )
    # O mantra vem da propria gravacao quando esta la; --mantra/--silent
    # sobrepoem-se. Se nao houver nem um nem outro, para: um relatorio nao
    # pode nomear um mantra por defeito quando ninguem disse qual tocou.
    mantra_label: str | None
    if args.silent:
        mantra_label = None
    elif args.mantra:
        mantra_label = args.mantra
    else:
        recorded = (meta.get("plan") or {}).get("mantra_id")
        if not recorded:
            raise SystemExit(
                "esta gravacao nao regista o mantra: usa --mantra <nome> ou "
                "--silent"
            )
        mantra_label = cfg.paths.mantra(str(recorded)).label

    context = ReportContext(
        participant_name=args.name,
        session_started=started,
        duration_s=total_s,
        mantra_label=mantra_label,
        session_code=f"FBE-{started:%Y}-{session.name.split('_')[-1][:6]}",
        festival_label=args.festival,
    )

    data = build(result, cfg, context, featured, phases)
    (session / "report.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    html = render(data)
    (session / "report.html").write_text(html, encoding="utf-8")

    print(f"\n  {data['counts']['reliable']}/{data['counts']['total']} com IC a excluir zero")
    print(f"  {session / 'report.json'}")
    print(f"  {session / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
