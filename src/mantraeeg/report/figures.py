"""Figuras do relatório.

Um painel por marcador seleccionado: eixo X é o tempo de sessão, eixo Y é a
unidade do próprio marcador. As fases estão sombreadas e as épocas rejeitadas
aparecem como **buracos** na linha — não são interpoladas, porque interpolar
sobre uma época má é inventar sinal (SPEC 3.3).

O texto por baixo de cada painel diz o que mudou, e diz "estável" quando o
intervalo de confiança cruza zero. Nunca "melhorou" (SPEC 5.1).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..analysis import AnalysisResult, MarkerSeries, MarkerSummary  # noqa: E402

BG = "#0b0d12"
FG = "#e8ecf4"
GRID = "#20252f"
PHASE_COLOURS = {
    "CALIBRATION": "#243044",
    "MANTRA": "#1f3a2c",
    "SETTLE": "#2b2536",
}
LINE = "#7fb2ff"


def _style(ax) -> None:
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors="#8b95a7", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def _shade_phases(ax, phases: list[tuple[str, float, float]]) -> None:
    for name, start, end in phases:
        colour = PHASE_COLOURS.get(name)
        if colour:
            ax.axvspan(start, end, color=colour, alpha=0.55, linewidth=0)


def _verdict(summary: MarkerSummary) -> tuple[str, str]:
    """Frase e cor. Nunca anuncia melhoria sem o IC a sustentar."""
    if not np.isfinite(summary.pct_change):
        return "sem dados suficientes", "#8b95a7"
    if not summary.reliable:
        return f"estável ({summary.change_text()}, IC cruza zero)", "#8b95a7"
    colour = "#8fd6a0" if summary.direction_ok is not False else "#e0b07a"
    return summary.change_text(), colour


def marker_timeline(
    result: AnalysisResult,
    phases: list[tuple[str, float, float]],
    out_path: Path,
    title: str = "Evolução dos biomarcadores",
) -> Path:
    """Um painel por marcador, empilhados no mesmo eixo temporal."""
    series = [s for s in result.series if np.any(np.isfinite(s.values))]
    if not series:
        raise ValueError("nenhum marcador tem valores finitos")

    height = max(2.0 * len(series), 4.0)
    fig, axes = plt.subplots(
        len(series), 1, figsize=(13, height), sharex=True, facecolor=BG
    )
    axes = np.atleast_1d(axes)

    for ax, item in zip(axes, series):
        summary = result.summary_of(item.marker.id)
        _style(ax)
        _shade_phases(ax, phases)

        # Épocas rejeitadas ficam como buracos: nunca interpolar.
        ax.plot(item.times_s, item.values, color=LINE, linewidth=1.4)
        ax.plot(
            item.times_s,
            item.values,
            ".",
            color=LINE,
            markersize=2.5,
            alpha=0.5,
        )
        if summary is not None and np.isfinite(summary.baseline):
            ax.axhline(
                summary.baseline, color="#55617a", linewidth=0.9, linestyle="--"
            )

        ax.set_ylabel(item.marker.unit or "", color="#8b95a7", fontsize=8)
        text, colour = _verdict(summary) if summary else ("", FG)
        ax.set_title(
            f"{item.marker.label}    {text}",
            color=colour,
            fontsize=11,
            loc="left",
            pad=6,
        )
        if summary is not None and abs(summary.r_emg) > 0.5:
            ax.text(
                0.995,
                0.06,
                f"r_emg {summary.r_emg:+.2f} — pode ser músculo",
                transform=ax.transAxes,
                ha="right",
                color="#e08b84",
                fontsize=8,
            )

    axes[-1].set_xlabel("tempo de sessão (s)", color="#8b95a7", fontsize=9)
    handles = [
        plt.Line2D([0], [0], color=colour, linewidth=8, alpha=0.55)
        for colour in PHASE_COLOURS.values()
    ]
    fig.legend(
        handles,
        ["calibração", "mantra", "repouso"],
        loc="upper right",
        facecolor=BG,
        edgecolor=GRID,
        labelcolor=FG,
        fontsize=9,
    )
    fig.suptitle(title, color=FG, fontsize=15, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=130, facecolor=BG)
    plt.close(fig)
    return out_path


def summary_bars(
    result: AnalysisResult, out_path: Path, title: str = "Variação face à calibração"
) -> Path:
    """Barras de variação percentual, ordenadas, com os não-fiáveis apagados."""
    summaries = [s for s in result.summaries if np.isfinite(s.pct_change)]
    if not summaries:
        raise ValueError("nenhum marcador com variação calculável")
    summaries.sort(key=lambda s: s.pct_change)

    fig, ax = plt.subplots(figsize=(11, max(0.5 * len(summaries) + 2, 4)), facecolor=BG)
    _style(ax)
    labels = [s.marker.friendly or s.marker.label for s in summaries]
    values = [s.pct_change for s in summaries]
    colours = [
        ("#8fd6a0" if s.direction_ok is not False else "#e0b07a")
        if s.reliable
        else "#3a4152"
        for s in summaries
    ]
    positions = np.arange(len(labels))
    ax.barh(positions, values, color=colours)
    ax.set_yticks(positions)
    ax.axvline(0, color="#55617a", linewidth=1)
    ax.set_xlabel("variação face à calibração (%)", color="#8b95a7", fontsize=9)
    for i, s in enumerate(summaries):
        note = "" if s.reliable else "  (estável)"
        ax.text(
            s.pct_change,
            positions[i],
            f"  {s.pct_change:+.0f}%{note}",
            va="center",
            color="#c8ccd4",
            fontsize=9,
        )
    ax.set_yticklabels(labels, color=FG, fontsize=10)
    fig.suptitle(title, color=FG, fontsize=15, x=0.02, ha="left")
    fig.text(
        0.02,
        0.01,
        "Barras apagadas: o intervalo de confiança cruza zero — não houve "
        "mudança fiável.",
        color="#5d6779",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=130, facecolor=BG)
    plt.close(fig)
    return out_path
