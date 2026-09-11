"""Compara dois segmentos marcados de uma gravação. O árbitro da qualidade.

    python scripts/compare_segments.py                      # sessão mais recente
    python scripts/compare_segments.py --session data/sessions/2026-...
    python scripts/compare_segments.py --a olhos_abertos --b olhos_fechados

Um semáforo de contacto responde "o sinal parece limpo?". Esta ferramenta
responde à pergunta que interessa: **"consigo recuperar deste canal um efeito
fisiológico que sei que existe?"**. Se sim, o canal mede cérebro, digam os
limiares o que disserem.

O efeito usado por defeito é o de **Berger**: fechar os olhos multiplica a
potência alfa (8–12 Hz) por 2 a 5x. É o efeito mais robusto e mais fácil de
evocar em EEG, e é a mesma coisa de que o protocolo de meditação depende — se
não o conseguimos ver, nenhum biomarcador mais subtil vai aparecer.

Corre o pipeline de pré-processamento completo (SPEC 3.1), portanto valida-o de
ponta a ponta ao mesmo tempo.

A deteção exige **IC por bootstrap por blocos que exclua zero**, não apenas uma
dimensão de efeito grande. Num controlo negativo — dois segmentos com a mesma
distribuição — o delta de Cliff sozinho declarava "efeito grande" em 3 de 8
canais. Com épocas sobrepostas 50 %, ele é anticonservador.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mantraeeg.config import Config, load_config  # noqa: E402
from mantraeeg.preprocess import preprocess  # noqa: E402
from mantraeeg.sources.base import EEG_SLICE  # noqa: E402
from mantraeeg.spectral import (
    band_power,
    band_reactivity,
    individual_alpha_frequency,
    reactivity_peak_frequency,
    relative_power,
    welch_psd,
)
from mantraeeg.stats import (  # noqa: E402
    block_bootstrap_ci,
    cliffs_delta,
    interpret_delta,
    median_log_ratio,
)

OK, WARN, BAD = "[ ok ]", "[aviso]", "[FALHA]"


def _setup_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


@dataclass(frozen=True)
class Segment:
    label: str
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


# --------------------------------------------------------------------------- #
# Carregar
# --------------------------------------------------------------------------- #
def latest_session(sessions_dir: Path) -> Path:
    candidates = sorted(p for p in sessions_dir.glob("*_p*") if p.is_dir())
    if not candidates:
        raise SystemExit(
            f"nenhuma sessão em {sessions_dir}. Grave uma com:\n"
            f"  python scripts/inspect_signal.py --record --seconds 200"
        )
    return candidates[-1]


def read_segments(session: Path, total_s: float) -> list[Segment]:
    """Reconstrói os segmentos a partir das marcas de ``events.json``."""
    events_path = session / "events.json"
    if not events_path.exists():
        return []
    events = json.loads(events_path.read_text(encoding="utf-8"))
    marks = [e for e in events if e.get("kind") in ("mark", "mark_end")]
    marks.sort(key=lambda e: e["t_s"])

    segments: list[Segment] = []
    for i, event in enumerate(marks):
        if event["kind"] != "mark":
            continue
        end = marks[i + 1]["t_s"] if i + 1 < len(marks) else total_s
        segments.append(Segment(event["detail"], float(event["t_s"]), float(end)))
    return segments


# --------------------------------------------------------------------------- #
# Análise
# --------------------------------------------------------------------------- #
def epoch_band_powers(
    clean: np.ndarray, cfg: Config, segment_slice: slice
) -> dict[str, np.ndarray]:
    """Potências por época dentro de um segmento. ``(n_epochs, n_ch)`` por banda."""
    grid = cfg.epochs.power
    sfreq = cfg.device.sfreq_nominal
    win = int(round(grid.win_s * sfreq))
    hop = int(round(grid.hop_s * sfreq))

    data = clean[:, segment_slice]
    starts = range(0, max(data.shape[1] - win + 1, 0), hop)

    wanted = ("alpha", "theta", "beta", "smr")
    collected: dict[str, list[np.ndarray]] = {b: [] for b in wanted}
    collected["alpha_abs"] = []
    collected["alpha_spec"] = []

    for start in starts:
        epoch = data[:, start : start + win]
        freqs, psd = welch_psd(epoch, sfreq, grid.welch_seg_s, grid.welch_overlap)
        for name in wanted:
            collected[name].append(
                relative_power(psd, freqs, cfg.bands[name], cfg.bands.total)
            )
        collected["alpha_abs"].append(band_power(psd, freqs, cfg.bands["alpha"]))
        # MEDIDA PRIMARIA: alfa normalizado pelos FLANCOS, em log2.
        #
        # A relativa (alfa / 1-45 Hz) e precisa mas confundida: o denominador
        # encolhe de olhos fechados porque ha menos pestanejos, e no Fp1 isso
        # inflacionou a subida de +26 % para +121 %.
        #
        # A absoluta e nao-confundida mas ruidosa: normaliza nada, e as
        # flutuacoes globais de amplitude alargam o IC ao ponto de so um canal
        # sobreviver.
        #
        # Normalizar pelos flancos tem as duas propriedades: cancela as
        # flutuacoes globais (como a relativa) mas so com bandas vizinhas do
        # alfa, que nao contem o artefacto ocular (ao contrario de 1-45).
        alpha_p = band_power(psd, freqs, cfg.bands["alpha"])
        flank_p = [band_power(psd, freqs, f) for f in cfg.stats.berger_flanks]
        with np.errstate(divide="ignore", invalid="ignore"):
            spec = np.log2(np.maximum(alpha_p, 1e-12)) - np.mean(
                [np.log2(np.maximum(f, 1e-12)) for f in flank_p], axis=0
            )
        collected["alpha_spec"].append(spec)

    return {
        k: (np.array(v) if v else np.zeros((0, clean.shape[0])))
        for k, v in collected.items()
    }


# --------------------------------------------------------------------------- #
def main() -> int:
    _setup_console()
    ap = argparse.ArgumentParser(description="Compara dois segmentos marcados")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--session", default=None)
    ap.add_argument("--a", default=None, help="rótulo do segmento de referência")
    ap.add_argument("--b", default=None, help="rótulo do segmento de comparação")
    ap.add_argument("--no-eog", action="store_true", help="desligar a regressão de EOG")
    ap.add_argument("--figure", action="store_true", help="gravar espectros.png")
    args = ap.parse_args()

    cfg = load_config(args.config)
    session = (
        Path(args.session) if args.session else latest_session(cfg.paths.sessions_dir)
    )
    label_a = args.a or cfg.ui.validation_pair[0]
    label_b = args.b or cfg.ui.validation_pair[1]

    raw = np.load(session / "raw.npy")
    meta = json.loads((session / "raw_meta.json").read_text(encoding="utf-8"))
    sfreq = cfg.device.sfreq_nominal
    total_s = raw.shape[1] / sfreq

    print(f"sessão   : {session}")
    print(f"duração  : {total_s:.1f} s   ({raw.shape[1]} amostras)")
    print(
        f"sfreq    : nominal {sfreq}, medida "
        f"{meta.get('sfreq_measured', float('nan')):.2f}"
    )

    segments = read_segments(session, total_s)
    if not segments:
        keys = ", ".join(f"[{k}] {v}" for k, v in cfg.ui.marker_keys.items())
        print(
            f"\n{BAD} a gravação não tem marcas. Grave outra com o inspetor aberto\n"
            f"       e use as teclas para marcar os segmentos:\n"
            f"         python scripts/inspect_signal.py --record --seconds 200\n"
            f"       teclas: {keys}"
        )
        return 2

    print(f"\nsegmentos marcados ({len(segments)}):")
    for s in segments:
        print(
            f"  {s.label:<16} {s.start_s:7.1f} -> {s.end_s:7.1f} s  "
            f"({s.duration_s:5.1f} s)"
        )

    chosen: dict[str, Segment | None] = {label_a: None, label_b: None}
    for s in segments:
        current = chosen.get(s.label, "absent")
        if current == "absent":
            continue
        if current is None or s.duration_s > current.duration_s:
            chosen[s.label] = s
    missing = [k for k, v in chosen.items() if v is None]
    if missing:
        print(f"\n{BAD} faltam segmentos marcados como: {missing}")
        return 2

    # ---- pré-processamento (SPEC 3.1), sobre o registo completo ------------- #
    apply_eog = False if args.no_eog else None
    pre = preprocess(raw[EEG_SLICE], cfg, apply_eog=apply_eog)
    print(
        f"\npré-processamento: notch {list(cfg.preprocess.notch_freqs_hz)} Hz, "
        f"passa-banda {list(cfg.preprocess.bandpass_hz)} Hz ordem "
        f"{cfg.preprocess.bandpass_order} (fase zero), margem descartada "
        f"{cfg.preprocess.filter_edge_s:.0f} s por ponta"
    )
    print(f"regressão de EOG : {'ligada' if pre.eog_applied else 'desligada'}")
    if pre.eog_applied and pre.eog_report.get("fell_back_to_all_samples"):
        print(
            f"  {WARN} poucos pestanejos detetados; os coeficientes foram "
            f"estimados sobre todo o registo"
        )

    def to_slice(segment: Segment) -> slice:
        lo = max(int(round((segment.start_s - pre.t0_s) * sfreq)), 0)
        hi = min(int(round((segment.end_s - pre.t0_s) * sfreq)), pre.n_samples)
        return slice(lo, hi)

    powers = {
        label: epoch_band_powers(pre.clean, cfg, to_slice(seg))
        for label, seg in chosen.items()
        if seg is not None
    }
    # Espectros por condicao, sobre o segmento inteiro: dao a reatividade de
    # banda e o pico alfa individual com muito melhor resolucao do que as
    # epocas de 4 s.
    spectra = {
        label: welch_psd(pre.clean[:, to_slice(seg)], sfreq, seg_s=8.0)
        for label, seg in chosen.items()
        if seg is not None
    }

    n_a = powers[label_a]["alpha_abs"].shape[0]
    n_b = powers[label_b]["alpha_abs"].shape[0]
    if min(n_a, n_b) < 3:
        print(
            f"\n{BAD} épocas a menos ({label_a}: {n_a}, {label_b}: {n_b}). "
            f"Grave mais tempo em cada segmento."
        )
        return 2

    # ---- resultado ---------------------------------------------------------- #
    grid = cfg.epochs.power
    bb = cfg.stats.block_bootstrap
    print(f"\n{'=' * 96}")
    lo_a, hi_a = cfg.bands["alpha"]
    print(
        f"ALFA ABSOLUTO ({lo_a:.0f}-{hi_a:.0f} Hz):  {label_a}  ->  {label_b}"
    )
    print(f"{'=' * 96}")
    print(
        f"épocas: {label_a} {n_a}, {label_b} {n_b}   "
        f"(janela {grid.win_s:.0f} s, hop {grid.hop_s:.0f} s)   "
        f"bootstrap: {cfg.stats.bootstrap_n} reamostragens por blocos"
    )
    print(
        f"  {'canal':<6} {'abs A':>10} {'abs F':>10} {'var':>7} "
        f"{'IC 95% (log2)':>17} {'react':>7} {'IAF':>6} {'n_ef':>5}  veredicto"
    )
    print(f"  {'-' * 96}")

    montage = cfg.montage
    detected: list[str] = []
    reactivities: dict[str, float] = {}
    underpowered: list[str] = []
    freqs_a, psd_a = spectra[label_a]
    freqs_b, psd_b = spectra[label_b]
    for idx in sorted(montage.channels):
        name = montage.channels[idx]
        a = powers[label_a]["alpha_spec"][:, idx]
        b = powers[label_b]["alpha_spec"][:, idx]
        abs_a = powers[label_a]["alpha_abs"][:, idx]
        abs_b = powers[label_b]["alpha_abs"][:, idx]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if a.size < 3:
            print(f"  {name:<6} {'sem dados':>10}")
            continue

        med_a, med_b = float(np.median(abs_a)), float(np.median(abs_b))
        change = (med_b / med_a - 1.0) * 100.0 if med_a > 0 else float("nan")
        react = band_reactivity(
            psd_a[idx], psd_b[idx], freqs_a, cfg.bands["alpha"], cfg.stats.berger_flanks
        )
        iaf = individual_alpha_frequency(psd_b[idx], freqs_b)
        reactivities[name] = react

        est = block_bootstrap_ci(
            a, b, lambda x, y: float(np.median(y) - np.median(x)),
            hop_s=grid.hop_s, min_length_s=bb.min_length_s,
            max_length_s=bb.max_length_s, n_boot=cfg.stats.bootstrap_n,
            ci_level=cfg.stats.ci_level,
        )

        # Quatro condicoes. A reatividade carrega a especificidade: sem ela,
        # uma subida modesta e indistinguivel de um deslocamento do espectro.
        enough = est.n_effective >= bb.min_effective_samples
        big = est.value >= cfg.stats.berger_min_log2
        specific = bool(np.isfinite(react) and react >= cfg.stats.berger_min_reactivity)
        found = est.significant and est.ci_low > 0 and enough and big and specific
        if found:
            detected.append(name)

        if not found and specific and big and enough and est.crosses_zero:
            underpowered.append(name)

        if found:
            verdict = "BERGER"
        elif not enough:
            verdict = "dados insuf."
        elif est.crosses_zero:
            verdict = "estável"
        elif not specific:
            verdict = "sobe, mas não é do alfa"
        else:
            verdict = "sobe pouco"

        ci = f"[{est.ci_low:+.2f}, {est.ci_high:+.2f}]" if np.isfinite(est.ci_low) else "—"
        print(
            f"  {name:<6} {med_a:>10.2f} {med_b:>10.2f} {change:>+6.0f}% "
            f"{ci:>17} {react:>+7.2f} {iaf:>6.1f} {est.n_effective:>5.1f}  {verdict}"
        )

    print()
    print(
        f"  alfa em uV2; react = log2(alfa) - log2(flancos "
        f"{[list(f) for f in cfg.stats.berger_flanks]}); 0 = espectro moveu-se todo junto."
    )

    print(f"\n{'=' * 96}")
    metric = [c for c in detected if c in montage.metric_channels]
    positive = [n for n, r in reactivities.items() if np.isfinite(r) and r > 0]
    strong = [n for n, r in reactivities.items() if np.isfinite(r) and r >= cfg.stats.berger_min_reactivity]

    # Evidencia AGREGADA, distinta da inferencia por epoca. Os espectros de
    # segmento inteiro tem muito melhor relacao sinal-ruido do que as epocas de
    # 4 s, e a concordancia entre canais e por si so informativa: sob a hipotese
    # nula esperar-se-iam metade dos canais a subir e metade a descer.
    # Evidencia AGREGADA, distinta da inferencia por epoca. Os espectros de
    # segmento inteiro tem muito melhor relacao sinal-ruido do que as epocas de
    # 4 s, e a concordancia entre canais e por si so informativa: sob a hipotese
    # nula esperar-se-iam metade dos canais a subir e metade a descer.
    print("REATIVIDADE AGREGADA (espectros de segmento inteiro)")
    print(
        f"  {len(positive)}/{len(reactivities)} canais com reatividade alfa "
        f"positiva; {len(strong)} acima de "
        f"{cfg.stats.berger_min_reactivity:+.2f}"
    )
    if reactivities and len(positive) == len(reactivities):
        print(
            "  Todos os canais na mesma direcao. Sob a nula esperar-se-ia "
            "metade para cada lado."
        )
    ordered = sorted(
        reactivities.items(),
        key=lambda kv: -kv[1] if np.isfinite(kv[1]) else 0.0,
    )
    print("  mais reativos: " + ", ".join(f"{n} {r:+.2f}" for n, r in ordered[:3]))

    agreement = len(positive) / len(reactivities) if reactivities else 0.0
    consistent = agreement >= cfg.stats.berger_min_channel_agreement

    print()
    if not consistent:
        print(
            f"{BAD} SEM EFEITO CONSISTENTE: so {len(positive)}/{len(reactivities)} "
            f"canais sobem ({agreement:.0%}, minimo "
            f"{cfg.stats.berger_min_channel_agreement:.0%})."
        )
        print("     Um efeito de Berger real e global; um falso positivo e disperso.")
        if detected:
            print(
                f"     Os {len(detected)} canais que passaram o teste individual "
                f"({chr(44).join(detected)}) nao chegam"
            )
            print("     para concluir contra uma divisao ao meio entre canais.")
    elif len(detected) >= 2:
        print(
            f"{OK} EFEITO DE BERGER CONFIRMADO por epoca em {len(detected)} "
            f"canais: {chr(44).join(detected)}"
        )
        print(
            f"     Dos canais de metrica (F3 F4 C3 C4): "
            f"{chr(44).join(metric) if metric else 0 or chr(110)+chr(101)+chr(110)+chr(104)+chr(117)+chr(109)}"
        )
    elif detected:
        print(f"{OK} EFEITO DE BERGER PRESENTE, confirmado por epoca em {detected[0]}.")
        print(
            f"     Reatividade agregada positiva em {len(positive)}/"
            f"{len(reactivities)} canais, com gradiente posterior>frontal — o"
        )
        print("     padrao fisiologico esperado.")
        print("     O que falta nao e efeito, e POTENCIA: para resolver por")
        print("     epoca um efeito frontal de +0,3 em log2 sao precisas mais")
        print("     amostras independentes. Gravar 4-5 min por condicao.")
    else:
        print(f"{WARN} reatividade consistente mas nenhum canal fecha o IC.")
        print("     Gravar mais tempo por condicao: 4-5 min de cada lado.")

    if underpowered:
        print()
        print(
            f"  {WARN} SEM POTENCIA (reatividade certa, IC largo demais): "
            f"{chr(44).join(underpowered)}"
        )
        print("         Mostram o efeito na direcao e magnitude certas mas o IC")
        print("         nao fecha. A correcao e gravar mais tempo por condicao,")
        print("         nao mexer nos limiares.")

    if args.figure:
        _save_figure(session, pre, cfg, chosen, to_slice)
    return 0


def _save_figure(session, pre, cfg, chosen, to_slice) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    montage = cfg.montage
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=True)
    for idx in sorted(montage.channels):
        ax = axes.flat[idx]
        for label, seg in chosen.items():
            data = pre.clean[:, to_slice(seg)]
            freqs, psd = welch_psd(data[idx], pre.sfreq, seg_s=4.0)
            keep = (freqs >= 1) & (freqs <= 45)
            ax.loglog(freqs[keep], psd[keep], label=label, linewidth=1.2)
        lo, hi = cfg.bands["alpha"]
        ax.axvspan(lo, hi, color="#ffcc66", alpha=0.25)
        ax.set_title(montage.channels[idx])
        ax.grid(True, which="both", alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Espectros por condição (banda alfa sombreada)")
    fig.supxlabel("Hz")
    fig.supylabel("µV²/Hz")
    fig.tight_layout()
    out = session / "espectros.png"
    fig.savefig(out, dpi=130)
    print(f"\nfigura: {out}")


if __name__ == "__main__":
    raise SystemExit(main())
