"""Analisa uma sessao gravada: series de biomarcadores, graficos e tabela.

    python scripts/analyse_session.py                     # sessao mais recente
    python scripts/analyse_session.py --session <pasta>
    python scripts/analyse_session.py --markers alpha_rel,beta_rel,coh_alpha1

Itera o registo de markers.py. Acrescentar um marcador la faz com que apareca
aqui, no grafico e na tabela, sem tocar neste ficheiro.

Se a gravacao nao tiver fases marcadas (calibracao/mantra), aceita as marcas do
inspetor com --a e --b.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mantraeeg.analysis import PHASE_ACTIVE, PHASE_BASELINE, analyse  # noqa: E402
from mantraeeg.config import load_config  # noqa: E402
from mantraeeg.markers import markers  # noqa: E402
from mantraeeg.report.figures import marker_timeline, summary_bars  # noqa: E402
from mantraeeg.sources.base import ACCEL_SLICE, EEG_SLICE, GYRO_SLICE  # noqa: E402

OK, WARN, BAD = "[ ok ]", "[aviso]", "[FALHA]"


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


def read_phases(session: Path, total_s: float, label_a: str, label_b: str):
    """Fases a partir de events.json, ou das marcas do inspetor."""
    events_path = session / "events.json"
    events = json.loads(events_path.read_text(encoding="utf-8")) if events_path.exists() else []

    phases = [e for e in events if e.get("kind") == "phase"]
    if phases:
        out = []
        ordered = sorted(phases, key=lambda e: e["t_s"])
        for i, event in enumerate(ordered):
            end = ordered[i + 1]["t_s"] if i + 1 < len(ordered) else total_s
            out.append((event["detail"], float(event["t_s"]), float(end)))
        return [p for p in out if p[2] > p[1]]

    marks = sorted(
        [e for e in events if e.get("kind") in ("mark", "mark_end")],
        key=lambda e: e["t_s"],
    )
    out = []
    for i, event in enumerate(marks):
        if event["kind"] != "mark":
            continue
        end = marks[i + 1]["t_s"] if i + 1 < len(marks) else total_s
        name = {label_a: PHASE_BASELINE, label_b: PHASE_ACTIVE}.get(
            event["detail"], event["detail"]
        )
        out.append((name, float(event["t_s"]), float(end)))
    return out


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description="Analise de uma sessao gravada")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--session", default=None)
    ap.add_argument("--markers", default=None, help="ids separados por virgula")
    ap.add_argument("--a", default="olhos_abertos", help="marca que serve de base")
    ap.add_argument("--b", default="olhos_fechados", help="marca de comparacao")
    args = ap.parse_args()

    cfg = load_config(args.config)
    session = Path(args.session) if args.session else latest_session(cfg)
    raw = np.load(session / "raw.npy")
    total_s = raw.shape[1] / cfg.device.sfreq_nominal

    print(f"sessao : {session}")
    print(f"duracao: {total_s:.1f} s")

    phases = read_phases(session, total_s, args.a, args.b)
    if not phases:
        print(f"{BAD} sem fases nem marcas em events.json")
        return 2
    print("\nfases:")
    for name, start, end in phases:
        print(f"  {name:<14} {start:7.1f} -> {end:7.1f} s  ({end - start:6.1f} s)")

    have = {name for name, _, _ in phases}
    if PHASE_BASELINE not in have or PHASE_ACTIVE not in have:
        print(
            f"\n{BAD} faltam as fases {PHASE_BASELINE} e {PHASE_ACTIVE}. "
            f"Use --a e --b para mapear as marcas."
        )
        return 2

    selected = None
    if args.markers:
        wanted = {m.strip() for m in args.markers.split(",") if m.strip()}
        selected = {m.id: (m.id in wanted) for m in markers()}

    print("\na analisar…")
    result = analyse(
        raw[EEG_SLICE],
        cfg,
        phases,
        accel=raw[ACCEL_SLICE],
        gyro=raw[GYRO_SLICE],
        selected=selected,
    )
    for note in result.notes:
        print(f"  {note}")

    print(f"\n{'=' * 104}")
    print(f"BIOMARCADORES: {PHASE_BASELINE} -> {PHASE_ACTIVE}   (mediana por fase)")
    print(f"{'=' * 104}")
    print(
        f"  {'marcador':<26}{'base':>11}{'mantra':>11}{'variacao':>10}"
        f"{'IC 95%':>20}{'n_ef':>6}{'cob':>6}{'r_emg':>7}  veredicto"
    )
    print(f"  {'-' * 100}")
    for s in result.summaries:
        if not np.isfinite(s.baseline):
            print(f"  {s.marker.label[:26]:<26}{'sem dados':>11}")
            continue
        ci = (
            f"[{s.ci_low:+.3f}, {s.ci_high:+.3f}]"
            if np.isfinite(s.ci_low)
            else "—"
        )
        verdict = "FIAVEL" if s.reliable else "estavel"
        if s.direction_ok is False and s.reliable:
            verdict = "FIAVEL (direcao inversa)"
        flag = " <-EMG" if abs(s.r_emg) > cfg.diagnostics.r_emg_red_threshold else ""
        # Quantidades com sinal centradas perto de zero nao levam
        # percentagem: o ALAY foi de -0,030 para -0,048, o que sai "+57 %" e
        # se le como melhoria quando e a direcao oposta.
        change = s.change_text()
        print(
            f"  {s.marker.label[:26]:<26}{s.baseline:>11.4f}{s.active:>11.4f}"
            f"{change:>10}{ci:>20}{s.n_effective:>6.1f}"
            f"{s.coverage * 100:>5.0f}%{s.r_emg:>+7.2f}  {verdict}{flag}"
        )

    reliable = [s for s in result.summaries if s.reliable]
    print(f"\n  {len(reliable)}/{len(result.summaries)} marcadores com IC a excluir zero")
    if reliable:
        print("\n  leitura para o participante:")
        for s in reliable:
            print(f"    · {s.headline()}")

    timeline = marker_timeline(result, phases, session / "biomarcadores.png")
    bars = summary_bars(result, session / "variacao.png")
    print(f"\nfiguras:\n  {timeline}\n  {bars}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
