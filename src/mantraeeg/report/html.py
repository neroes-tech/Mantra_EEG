"""O relatório do participante, em HTML, num único ficheiro.

Consome o dicionário de ``payload.build`` e devolve uma página que não depende
de nada externo além dos tipos de letra do Google — os gráficos são SVG gerado
aqui, não imagens, para a página aguentar impressão e email sem anexos.

**O design é o da Neroes**, não um inventado: a escada de tinta petróleo, o
trio Sora / IBM Plex Mono / Newsreader, e a regra de que o latão só aparece
onde há energia ou visão. Tema escuro único e deliberado — o design system
diz "dark, instrument-grade", e um relatório de leituras de instrumento não
tem versão clara.

Uma decisão de conteúdo, herdada de ``payload``: **a cor é informação**. Verde
é efeito no sentido esperado, latão é efeito no sentido oposto, cinzento é
estável. Nada fica verde por defeito.
"""

from __future__ import annotations

import html
from typing import Any, Sequence

# Escada de tinta petróleo (tokens/colors.css).
INK_900 = "#0C1D24"
INK_700 = "#152E38"
INK_600 = "#1B3A46"
INK_HEAD = "#0B333C"
TEXT_1 = "#E9F2F3"
TEXT_2 = "#A9BFC5"
TEXT_3 = "#6E8B93"
BORDER = "rgba(169,191,197,0.16)"
DIVIDER = "rgba(169,191,197,0.09)"

#: Ordem das linhas do gráfico da sessão. Laranja, verde e azul-bebé: três
#: matizes bem afastadas no círculo cromático, para se saber qual é qual sem
#: consultar a legenda. O teal da marca ficaria a competir com o azul e as
#: duas linhas confundiam-se.
LINE_COLOURS = ["#E8944D", "#5BBC99", "#7FCBE8", "#C9A0DC"]

FONTS = (
    "https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600"
    "&family=IBM+Plex+Mono:ital,wght@0,400;0,500;1,400"
    "&family=Newsreader:ital,opsz,wght@1,6..72,400..500&display=swap"
)

SANS = "'Sora','Avenir Next','Segoe UI',system-ui,sans-serif"
MONO = "'IBM Plex Mono','SFMono-Regular',Menlo,monospace"
SERIF = "'Newsreader',Georgia,serif"


def _esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


# --------------------------------------------------------------------------- #
# Gráficos — SVG gerado, não imagem
# --------------------------------------------------------------------------- #
def _nice_step(span: float) -> float:
    """Um passo de grelha redondo. Sem isto os eixos saem com 0,37 em 0,37."""
    if span <= 0:
        return 1.0
    raw = span / 4.0
    for step in (0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 25.0, 50.0):
        if raw <= step:
            return step
    return 100.0


def _line_chart(chart: dict[str, Any], active_label: str) -> str:
    """A sessão inteira, com todas as métricas em destaque no mesmo eixo.

    O eixo Y está em unidades da variabilidade do próprio repouso: zero é o
    valor típico da calibração, +1 é um desvio acima da oscilação normal. É a
    única escala que aceita percentagens de potência e coerências no mesmo
    gráfico sem uma delas ficar achatada contra a margem.
    """
    series = chart.get("series") or []
    if not series:
        return ""

    width, height = 860.0, 330.0
    left, right, top, bottom = 46.0, 18.0, 20.0, 46.0
    plot_w = width - left - right
    plot_h = height - top - bottom

    t0, t1 = float(chart["t_start"]), float(chart["t_end"])
    if t1 <= t0:
        return ""
    values = [v for s in series for v in s["deviation"]]
    lo, hi = min(values), max(values)
    pad = max((hi - lo) * 0.12, 0.3)
    lo, hi = lo - pad, hi + pad
    step = _nice_step(hi - lo)

    def x_of(t: float) -> float:
        return left + (t - t0) / (t1 - t0) * plot_w

    def y_of(v: float) -> float:
        return top + (hi - v) / (hi - lo) * plot_h

    parts: list[str] = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" '
        f'role="img" aria-label="Evolução das métricas em destaque ao longo '
        f'da sessão" style="display:block;overflow:visible;">'
    ]

    # Grelha horizontal, com rótulo em cada linha desenhada.
    tick = step * round(lo / step)
    while tick <= hi + 1e-9:
        if tick >= lo:
            y = y_of(tick)
            zero = abs(tick) < 1e-9
            parts.append(
                f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{width - right:.1f}" '
                f'y2="{y:.1f}" stroke="rgba(169,191,197,'
                f'{0.28 if zero else 0.09})" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{left - 10:.1f}" y="{y + 3.5:.1f}" text-anchor="end" '
                f'fill="{TEXT_3}" font-family="{MONO}" font-size="10">'
                f'{("+" if tick > 0 else "")}{tick:g}</text>'
            )
        tick += step

    # Eixo do tempo em minutos **da sessão**, não do início do gráfico. O
    # gráfico começa aos 7 s (a margem que o filtro come nas pontas), e contar
    # a partir daí deslocava a marca do início da fase seguinte.
    import math

    minute = math.ceil(t0 / 60)
    while minute * 60 <= t1:
        x = x_of(minute * 60)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top + plot_h:.1f}" x2="{x:.1f}" '
            f'y2="{top + plot_h + 5:.1f}" stroke="rgba(169,191,197,0.28)"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 19:.1f}" text-anchor="middle" '
            f'fill="{TEXT_3}" font-family="{MONO}" font-size="10">'
            f"{minute}′</text>"
        )
        minute += 1

    # A fronteira entre repouso e a fase seguinte.
    mantra_at = chart.get("mantraAt")
    if mantra_at is not None and t0 < float(mantra_at) < t1:
        x = x_of(float(mantra_at))
        parts.append(
            f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" '
            f'y2="{top + plot_h:.1f}" stroke="#D4A455" stroke-width="1" '
            f'stroke-dasharray="3 4" opacity="0.75"/>'
        )
        parts.append(
            f'<text x="{x + 7:.1f}" y="{top + 12:.1f}" fill="#D4A455" '
            f'font-family="{MONO}" font-size="9.5" letter-spacing="0.14em">'
            f"INÍCIO — {_esc(active_label.upper())}</text>"
        )

    for i, s in enumerate(series):
        colour = LINE_COLOURS[i % len(LINE_COLOURS)]
        points = " ".join(
            f"{x_of(t):.1f},{y_of(v):.1f}"
            for t, v in zip(s["times"], s["deviation"])
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{colour}" '
            f'stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/>'
        )

    parts.append("</svg>")

    legend = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:7px;">'
        f'<span style="width:14px;height:2px;background:'
        f'{LINE_COLOURS[i % len(LINE_COLOURS)]};display:block;border-radius:1px;">'
        f"</span>"
        f'<span style="font-family:{MONO};font-size:10.5px;letter-spacing:0.1em;'
        f'text-transform:uppercase;color:{TEXT_2};">{_esc(s["label"])}</span>'
        f"</span>"
        for i, s in enumerate(series)
    )

    return (
        f'<div style="background:{INK_900};border:1px solid {BORDER};'
        f'border-radius:10px;padding:18px 20px 14px;">'
        f'{"".join(parts)}'
        f'<div style="display:flex;flex-wrap:wrap;gap:18px;margin-top:14px;'
        f'padding-top:14px;border-top:1px solid {DIVIDER};">{legend}</div>'
        f"</div>"
    )


def _sparkline(
    values: Sequence[float], mantra_at: float | None, colour: str, index: int
) -> str:
    """A trajetória de um marcador, do tamanho de uma linha de texto."""
    if len(values) < 3:
        return (
            f'<div style="height:44px;display:flex;align-items:center;'
            f'font-family:{MONO};font-size:10px;color:{TEXT_3};">sem série</div>'
        )
    width, height, pad = 240.0, 44.0, 5.0
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        lo, hi = lo - 1.0, hi + 1.0

    def x_of(i: int) -> float:
        return i / (len(values) - 1) * width

    def y_of(v: float) -> float:
        return pad + (hi - v) / (hi - lo) * (height - 2 * pad)

    line = " ".join(f"{x_of(i):.1f},{y_of(v):.1f}" for i, v in enumerate(values))
    area = f"0,{height:.1f} {line} {width:.1f},{height:.1f}"
    # Um id por cartão. hash() de strings é aleatorizado por processo e
    # colidir aqui faria dois cartões partilharem o mesmo gradiente.
    uid = f"spark{index}"

    marker = ""
    if mantra_at is not None:
        x = float(mantra_at) * width
        marker = (
            f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{height:.1f}" '
            f'stroke="#D4A455" stroke-width="1" stroke-dasharray="2 3" '
            f'opacity="0.6"/>'
        )

    return (
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" '
        f'height="{height:.0f}" preserveAspectRatio="none" aria-hidden="true" '
        f'style="display:block;">'
        f'<defs><linearGradient id="{uid}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{colour}" stop-opacity="0.24"/>'
        f'<stop offset="100%" stop-color="{colour}" stop-opacity="0"/>'
        f"</linearGradient></defs>"
        f'<polygon points="{area}" fill="url(#{uid})"/>'
        f"{marker}"
        f'<polyline points="{line}" fill="none" stroke="{colour}" '
        f'stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>'
        f'<circle cx="{width:.1f}" cy="{y_of(values[-1]):.1f}" r="2.6" '
        f'fill="{colour}"/>'
        f"</svg>"
    )


def _spectrum_chart(spectrum: dict[str, Any]) -> str:
    """O espectro do sinal da pessoa, com as bandas nomeadas por baixo.

    É o que se mostra quando a comparação entre fases não se sustenta: não
    depende de linha de base nenhuma, e continua a ser verdade com poucas
    épocas — só fica com aspeto mais ruidoso, que é o aspeto que deve ter.
    """
    freqs = spectrum.get("freqs") or []
    values = spectrum.get("db") or []
    if len(freqs) < 8 or len(values) != len(freqs):
        return ""

    width, height = 860.0, 300.0
    left, right, top, bottom = 46.0, 18.0, 20.0, 52.0
    plot_w, plot_h = width - left - right, height - top - bottom
    f0, f1 = freqs[0], freqs[-1]
    lo, hi = min(values), max(values)
    pad = max((hi - lo) * 0.1, 1.0)
    lo, hi = lo - pad, hi + pad

    def x_of(f: float) -> float:
        return left + (f - f0) / (f1 - f0) * plot_w

    def y_of(v: float) -> float:
        return top + (hi - v) / (hi - lo) * plot_h

    parts = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" role="img" '
        f'aria-label="Espectro do sinal" style="display:block;overflow:visible;">'
    ]

    # As bandas, sombreadas e nomeadas: e o que torna o grafico legivel para
    # quem nunca viu um espectro.
    for name, colour in (
        ("theta", "#5B8AD4"),
        ("alpha", "#43BEC3"),
        ("beta", "#5BBC99"),
    ):
        band = (spectrum.get("bands") or {}).get(name)
        if not band:
            continue
        x0, x1 = x_of(max(band[0], f0)), x_of(min(band[1], f1))
        parts.append(
            f'<rect x="{x0:.1f}" y="{top:.1f}" width="{max(x1 - x0, 0):.1f}" '
            f'height="{plot_h:.1f}" fill="{colour}" opacity="0.10"/>'
        )
        parts.append(
            f'<text x="{(x0 + x1) / 2:.1f}" y="{top + plot_h + 32:.1f}" '
            f'text-anchor="middle" fill="{colour}" font-family="{MONO}" '
            f'font-size="9.5" letter-spacing="0.14em">{name.upper()}</text>'
        )

    for hz in (5, 10, 15, 20, 25, 30, 35, 40):
        if not f0 <= hz <= f1:
            continue
        x = x_of(hz)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top + plot_h:.1f}" x2="{x:.1f}" '
            f'y2="{top + plot_h + 5:.1f}" stroke="rgba(169,191,197,0.28)"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 18:.1f}" text-anchor="middle" '
            f'fill="{TEXT_3}" font-family="{MONO}" font-size="10">{hz}</text>'
        )

    points = " ".join(
        f"{x_of(f):.1f},{y_of(v):.1f}" for f, v in zip(freqs, values)
    )
    parts.append(
        f'<polyline points="{points}" fill="none" stroke="#43BEC3" '
        f'stroke-width="1.8" stroke-linejoin="round"/>'
    )
    parts.append(
        f'<text x="{width / 2:.1f}" y="{height - 4:.1f}" text-anchor="middle" '
        f'fill="{TEXT_3}" font-family="{MONO}" font-size="9.5" '
        f'letter-spacing="0.14em">OSCILACOES POR SEGUNDO (HZ)</text>'
    )
    parts.append("</svg>")
    return (
        f'<div style="background:{INK_900};border:1px solid {BORDER};'
        f'border-radius:10px;padding:18px 20px 10px;">{"".join(parts)}</div>'
    )


def _coverage_chart(trace: dict[str, Any]) -> str:
    """Onde houve leitura fiável ao longo da sessão. Uma faixa, nada mais."""
    times = trace.get("times") or []
    ok = trace.get("ok") or []
    if len(times) < 4 or len(ok) != len(times):
        return ""

    width, height = 860.0, 74.0
    left, right, top = 46.0, 18.0, 12.0
    band_h = 26.0
    plot_w = width - left - right
    t0, t1 = times[0], times[-1]
    if t1 <= t0:
        return ""

    parts = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" role="img" '
        f'aria-label="Leitura fiavel ao longo da sessao" '
        f'style="display:block;overflow:visible;">',
        f'<rect x="{left:.1f}" y="{top:.1f}" width="{plot_w:.1f}" '
        f'height="{band_h:.1f}" fill="rgba(169,191,197,0.07)" rx="3"/>',
    ]
    step = plot_w / len(times)
    for i, good in enumerate(ok):
        if not good:
            continue
        parts.append(
            f'<rect x="{left + i * step:.2f}" y="{top:.1f}" '
            f'width="{step + 0.6:.2f}" height="{band_h:.1f}" fill="#5BBC99" '
            f'opacity="0.85"/>'
        )

    minute = 0
    while t0 + minute * 60 <= t1:
        x = left + (minute * 60 - t0) / (t1 - t0) * plot_w
        if x >= left:
            parts.append(
                f'<text x="{x:.1f}" y="{top + band_h + 16:.1f}" '
                f'text-anchor="middle" fill="{TEXT_3}" font-family="{MONO}" '
                f'font-size="10">{minute}\u2032</text>'
            )
        minute += 1
    parts.append("</svg>")
    return (
        f'<div style="background:{INK_900};border:1px solid {BORDER};'
        f'border-radius:10px;padding:14px 20px 8px;">{"".join(parts)}'
        f'<div style="font-family:{MONO};font-size:9.5px;letter-spacing:0.14em;'
        f'text-transform:uppercase;color:{TEXT_3};padding-bottom:6px;">'
        f'verde = leitura fiavel</div></div>'
    )


def _bars(metrics: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Todos os marcadores em percentagem, ordenados por magnitude.

    Devolve também os que ficaram de fora: os que vivem perto de zero, onde
    uma percentagem é uma razão com um denominador minúsculo e sai um número
    enorme que não quer dizer nada. Esses aparecem nomeados por baixo, com a
    diferença absoluta, em vez de desaparecerem em silêncio.
    """
    rows = [m for m in metrics if m.get("pct") is not None]
    excluded = [m["name"] for m in metrics if m.get("pct") is None]
    if not rows:
        return "", excluded
    rows.sort(key=lambda m: abs(m["pct"]), reverse=True)
    limit = max(max(abs(m["pct"]) for m in rows), 5.0)

    out = []
    for m in rows:
        value = m["pct"]
        fraction = abs(value) / limit / 2 * 100
        colour = m["deltaColor"]
        side = (
            f"left:50%;width:{fraction:.2f}%;"
            if value >= 0
            else f"right:50%;width:{fraction:.2f}%;"
        )
        out.append(
            f'<div style="display:grid;grid-template-columns:150px 1fr 62px;'
            f'align-items:center;gap:12px;">'
            f'<span style="font-family:{SANS};font-size:12.5px;color:{TEXT_2};'
            f'text-align:right;">{_esc(m["name"])}</span>'
            f'<span style="position:relative;display:block;height:15px;'
            f'background:{INK_900};border-radius:3px;">'
            f'<span style="position:absolute;left:50%;top:0;bottom:0;width:1px;'
            f'background:rgba(169,191,197,0.28);"></span>'
            f'<span style="position:absolute;top:2px;bottom:2px;{side}'
            f'background:{colour};border-radius:2px;opacity:'
            f'{"1" if m["reliable"] else "0.42"};"></span>'
            f"</span>"
            f'<span style="font-family:{MONO};font-size:11px;color:{colour};'
            f'font-variant-numeric:tabular-nums;text-align:right;">'
            f'{_esc(m["pctText"])}</span>'
            f"</div>"
        )

    header = (
        f'<div style="display:grid;grid-template-columns:150px 1fr 62px;'
        f'align-items:center;gap:12px;padding-bottom:4px;">'
        f"<span></span>"
        f'<span style="font-family:{MONO};font-size:9px;letter-spacing:0.18em;'
        f'text-transform:uppercase;color:{TEXT_3};text-align:center;">'
        f"menos ← repouso → mais</span>"
        f'<span style="font-family:{MONO};font-size:9px;letter-spacing:0.18em;'
        f'text-transform:uppercase;color:{TEXT_3};text-align:right;">'
        f"variação %</span>"
        f"</div>"
    )
    return (
        f'<div style="background:{INK_600};border:1px solid {BORDER};'
        f'border-radius:10px;padding:20px;display:flex;flex-direction:column;'
        f'gap:9px;overflow-x:auto;">{header}{"".join(out)}</div>',
        excluded,
    )


# --------------------------------------------------------------------------- #
# Secções
# --------------------------------------------------------------------------- #
def _eyebrow(number: str, label: str) -> str:
    return (
        f'<div style="display:flex;align-items:baseline;gap:14px;'
        f'margin-bottom:18px;">'
        f'<span style="font-family:{MONO};font-size:10px;font-weight:500;'
        f'letter-spacing:0.22em;text-transform:uppercase;color:{TEXT_3};'
        f'white-space:nowrap;">{_esc(number)} — {_esc(label)}</span>'
        f'<span style="flex:1;height:1px;background:{DIVIDER};"></span>'
        f"</div>"
    )


def _meta_cell(label: str, value: str, mono: bool = False) -> str:
    family = MONO if mono else SANS
    size = "13px" if mono else "14px"
    return (
        f'<div style="background:{INK_700};padding:14px 16px;display:flex;'
        f'flex-direction:column;gap:6px;">'
        f'<span style="font-family:{MONO};font-size:9px;font-weight:500;'
        f'letter-spacing:0.22em;text-transform:uppercase;color:{TEXT_3};">'
        f"{_esc(label)}</span>"
        f'<span style="font-family:{family};font-size:{size};color:{TEXT_1};'
        f'letter-spacing:0.03em;font-variant-numeric:tabular-nums;">'
        f"{_esc(value)}</span>"
        f"</div>"
    )


def _headline_card(item: dict[str, Any]) -> str:
    """Nome, percentagem face ao repouso, e se foi medido ou está estável.

    Sem unidade e sem valor absoluto: "fração de 1–45 Hz" não diz nada a quem
    acabou de tirar o headset da cabeça, e ocupava o espaço do único número
    que a pessoa quer ver aqui.
    """
    note = "medido" if item["reliable"] else "dentro do normal"
    return (
        f'<div style="background:{INK_600};border:1px solid {BORDER};'
        f'border-radius:10px;padding:20px 20px 18px;display:flex;'
        f'flex-direction:column;gap:14px;break-inside:avoid;">'
        f'<span style="font-family:{MONO};font-size:9px;font-weight:500;'
        f'letter-spacing:0.22em;text-transform:uppercase;color:{TEXT_3};">'
        f'{_esc(item["label"])}</span>'
        f'<span style="font-family:{SANS};font-size:38px;font-weight:600;'
        f'letter-spacing:-0.03em;line-height:1;color:{item["color"]};'
        f'font-variant-numeric:tabular-nums;">{_esc(item["change"])}</span>'
        f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">'
        f'<span style="width:6px;height:6px;border-radius:50%;display:block;'
        f'background:{item["color"]};flex:none;"></span>'
        f'<span style="font-family:{MONO};font-size:10px;letter-spacing:0.1em;'
        f'text-transform:uppercase;color:{TEXT_3};">vs repouso · {note}</span>'
        f"</div></div>"
    )


def _metric_card(m: dict[str, Any], index: int) -> str:
    coverage = m.get("coverage")
    coverage_text = f"{coverage * 100:.0f}% boas" if coverage is not None else "—"
    spark = _sparkline(m["series"], m.get("mantraAt"), m["deltaColor"], index)
    return (
        f'<div style="background:{INK_600};border:1px solid {BORDER};'
        f'border-radius:10px;padding:18px;display:flex;flex-direction:column;'
        f'gap:14px;break-inside:avoid;">'
        f'<div style="display:flex;align-items:flex-start;'
        f'justify-content:space-between;gap:12px;">'
        f'<div style="display:flex;flex-direction:column;gap:4px;min-width:0;">'
        f'<span style="font-family:{SANS};font-size:15px;font-weight:600;'
        f'letter-spacing:-0.01em;color:{TEXT_1};text-wrap:pretty;">'
        f'{_esc(m["name"])}</span>'
        f'<span style="font-family:{MONO};font-size:9px;letter-spacing:0.12em;'
        f'color:{TEXT_3};">{_esc(m["basis"])}</span></div>'
        f'<span style="font-family:{MONO};font-size:9.5px;letter-spacing:0.08em;'
        f'color:{TEXT_3};flex:none;padding-top:3px;white-space:nowrap;">'
        f"{_esc(coverage_text)}</span></div>"
        f'<div style="background:{INK_900};border-radius:6px;padding:8px;">'
        f"{spark}</div>"
        f'<div style="display:flex;flex-direction:column;gap:9px;">'
        f'<div style="display:flex;align-items:baseline;gap:7px;'
        f'flex-wrap:wrap;">'
        f'<span style="font-family:{MONO};font-size:11px;color:{TEXT_3};'
        f'font-variant-numeric:tabular-nums;">{_esc(m["baseline"])}</span>'
        f'<span style="font-family:{MONO};font-size:11px;color:{TEXT_3};">→</span>'
        f'<span style="font-family:{SANS};font-size:22px;font-weight:600;'
        f'letter-spacing:-0.02em;color:{TEXT_1};line-height:1;'
        f'font-variant-numeric:tabular-nums;">{_esc(m["value"])}</span>'
        f'<span style="font-family:{MONO};font-size:9px;letter-spacing:0.1em;'
        f'color:{TEXT_3};">{_esc(m["unit"])}</span></div>'
        f'<div style="display:flex;align-items:center;gap:8px;">'
        f'<span style="font-family:{MONO};font-size:11px;font-weight:500;'
        f'letter-spacing:0.06em;padding:3px 9px;border-radius:999px;'
        f'font-variant-numeric:tabular-nums;color:{m["deltaColor"]};'
        f'background:{m["deltaBg"]};white-space:nowrap;">'
        f'{_esc(m["pctText"])}</span>'
        f'<span style="font-family:{MONO};font-size:9px;letter-spacing:0.12em;'
        f'text-transform:uppercase;color:{TEXT_3};">vs repouso</span>'
        f"</div></div>"
        f'<p style="font-family:{SANS};font-size:12.5px;line-height:1.55;'
        f'color:{TEXT_2};margin:0;padding-top:12px;border-top:1px solid '
        f'{DIVIDER};text-wrap:pretty;">{_esc(m["interpretation"])}</p>'
        f"</div>"
    )


def render(data: dict[str, Any], fragment: bool = False) -> str:
    """A página. ``fragment=True`` omite o esqueleto, para publicação."""
    meta, session, copy = data["meta"], data["session"], data["copy"]
    logo = meta.get("logo") or ""

    header_logo = (
        f'<img src="{logo}" alt="neroes" style="height:46px;width:auto;'
        f'display:block;">'
        if logo
        else f'<span style="font-family:{SANS};font-size:22px;font-weight:600;'
        f'letter-spacing:-0.02em;color:{TEXT_1};">neroes</span>'
    )

    headline = "".join(_headline_card(h) for h in data["headline"])
    featured_note = (
        "Os biomarcadores que mais se destacaram nesta sessão em particular — "
        "primeiro os que se mediram com confiança. A escolha é feita por "
        "sessão, porque o que se move numa pessoa não é o que se move noutra."
    )

    # Só os marcadores. Os `extra` existem para verificar os outros e não têm
    # direção estabelecida na literatura: são material de investigação, e não
    # de um relatório que a pessoa leva para casa.
    # Os que mais mudaram, e mais nada. O resto fica em report.json: o corte
    # e de apresentacao, e uma pagina com oito cartoes dos quais seis dizem
    # "manteve-se estavel" cansa antes de informar.
    limit = int(data.get("maxMetrics", 6))
    metrics_markers = [m for m in data["metrics"] if m["role"] == "marker"][:limit]
    cards = "".join(_metric_card(m, i) for i, m in enumerate(metrics_markers))
    bars, _ = _bars(metrics_markers)

    reading = "".join(
        f'<p style="font-family:{SANS};font-size:15px;line-height:1.68;'
        f'color:{TEXT_2};margin:0;text-wrap:pretty;">{_esc(p)}</p>'
        for p in data["reading"]
    )

    glossary = "".join(
        f'<div style="display:flex;flex-direction:column;gap:5px;'
        f'break-inside:avoid;">'
        f'<span style="font-family:{MONO};font-size:10px;font-weight:500;'
        f'letter-spacing:0.14em;color:{TEXT_1};text-transform:uppercase;">'
        f'{_esc(g["term"])}</span>'
        f'<span style="font-family:{SANS};font-size:13px;line-height:1.55;'
        f'color:{TEXT_3};">{_esc(g["def"])}</span></div>'
        for g in data["glossary"]
    )

    spectrum_chart = _spectrum_chart(data.get("spectrum") or {})
    coverage_chart = _coverage_chart(data.get("coverageTrace") or {})
    tier = str(data.get("tier", "full"))
    # A numeracao segue o que a pagina mostra: com dois blocos em vez
    # de quatro, saltar de 02 para 05 parecia falta de seccoes.
    reading_number = {"full": "05", "descriptive": "03"}.get(tier, "02")
    if tier == "full":
        sections = f"""
  <div class="om-pad" style="padding-top:40px;">
    {_eyebrow("01", "Em destaque")}
    <p style="font-family:{SANS};font-size:14px;line-height:1.6;color:{TEXT_2};
       margin:-6px 0 18px;max-width:60ch;text-wrap:pretty;">{featured_note}</p>
    <div class="om-cards-3">{headline}</div>
  </div>

  <div class="om-pad" style="padding-top:36px;">
    {_eyebrow("02", "A sessão minuto a minuto")}
    {_line_chart(data["chart"], session["activeLabel"])}
  </div>

  <div class="om-pad" style="padding-top:36px;">
    {_eyebrow("03", "Variação de todos os biomarcadores")}
    <p style="font-family:{SANS};font-size:14px;line-height:1.6;color:{TEXT_2};
       margin:-6px 0 18px;max-width:60ch;text-wrap:pretty;">
      Quanto cada um mudou face ao teu repouso, em percentagem.</p>
    {bars}
  </div>

  <div class="om-pad" style="padding-top:36px;">
    {_eyebrow("04", "Biomarcador a biomarcador")}
    <div class="om-cards-2">{cards}</div>
  </div>

"""
    elif tier == "descriptive":
        sections = f"""
  <div class="om-pad" style="padding-top:40px;">
    {_eyebrow("01", "O teu espectro")}
    <p style="font-family:{SANS};font-size:14px;line-height:1.6;color:{TEXT_2};margin:-6px 0 18px;max-width:62ch;text-wrap:pretty;">
      Em que ritmos a tua atividade se concentrou durante a sessão. Não
      é uma comparação com nada — é o teu sinal, tal como foi gravado.</p>
    {spectrum_chart}
  </div>

  <div class="om-pad" style="padding-top:36px;">
    {_eyebrow("02", "Onde houve leitura")}
    {coverage_chart}
  </div>
"""
    else:
        sections = f"""
  <div class="om-pad" style="padding-top:40px;">
    {_eyebrow("01", "O que conseguimos gravar")}
    {coverage_chart}
  </div>
"""

    body = f"""
<div class="om-page">
<div class="om-sheet">

  <div style="background:{INK_HEAD};padding:22px 40px;display:flex;
       align-items:center;justify-content:space-between;gap:24px;flex-wrap:wrap;">
    {header_logo}
    <div style="text-align:right;display:flex;flex-direction:column;gap:5px;">
      <span style="font-family:{MONO};font-size:10px;font-weight:500;
            letter-spacing:0.22em;text-transform:uppercase;color:{TEXT_3};">
        Relatório de sessão</span>
      <span style="font-family:{MONO};font-size:12px;font-weight:500;
            letter-spacing:0.14em;color:{TEXT_2};">{_esc(meta["code"])}</span>
    </div>
  </div>

  <div class="om-pad" style="padding-top:44px;padding-bottom:36px;
       border-bottom:1px solid {DIVIDER};">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;">
      <span style="width:6px;height:6px;border-radius:50%;background:#2E9296;
            display:block;flex:none;"></span>
      <span style="font-family:{MONO};font-size:10px;font-weight:500;
            letter-spacing:0.22em;text-transform:uppercase;color:#2E9296;">
        {_esc(meta["festivalLabel"])}</span>
    </div>
    <h1 style="font-family:{SANS};font-size:clamp(28px,6vw,42px);font-weight:600;
        letter-spacing:-0.025em;line-height:1.12;color:{TEXT_1};
        margin:0 0 16px;text-wrap:balance;">{_esc(copy["title"])}</h1>
    <p style="font-family:{SANS};font-size:15px;line-height:1.6;color:{TEXT_2};
       margin:0;max-width:58ch;text-wrap:pretty;">{_esc(copy["intro"])}</p>

    <div class="om-meta">
      {_meta_cell("Participante", data["participant"]["name"])}
      {_meta_cell("Data · hora", session["datetime"], mono=True)}
      {_meta_cell("Duração", session["duration"], mono=True)}
      {_meta_cell("Áudio", session["mantra"])}
    </div>
  </div>

{sections}

  <div class="om-pad" style="padding-top:36px;">
    {_eyebrow(reading_number, "Leitura da sessão")}
    <div style="display:flex;flex-direction:column;gap:16px;max-width:66ch;">
      {reading}
    </div>
  </div>

  <div class="om-pad" style="padding-top:36px;padding-bottom:40px;">
    <div style="border-top:1px solid {BORDER};border-bottom:1px solid {BORDER};
         padding:32px 0;display:flex;gap:16px;align-items:flex-start;">
      <span style="width:7px;height:7px;border-radius:50%;background:#B8873C;
            display:block;margin-top:12px;flex:none;"></span>
      <p style="font-family:{SERIF};font-style:italic;
         font-size:clamp(19px,4.4vw,24px);line-height:1.42;color:{TEXT_1};
         margin:0;max-width:38ch;text-wrap:pretty;">A saúde mental devia ser
         tão treinável quanto a saúde física.</p>
    </div>
  </div>

  <div class="om-pad" style="padding-bottom:36px;">
    {_eyebrow("Anexo", "Glossário")}
    <div class="om-glossary">{glossary}</div>
  </div>

  <div class="om-pad" style="padding-top:32px;padding-bottom:40px;
       background:{INK_HEAD};border-top:1px solid {DIVIDER};display:flex;
       flex-direction:column;gap:18px;">
    <span style="font-family:{MONO};font-size:10px;font-weight:500;
          letter-spacing:0.22em;text-transform:uppercase;color:{TEXT_3};">
      Lisboa · Mental Training Platform</span>
    <p style="font-family:{SANS};font-size:12px;line-height:1.6;color:{TEXT_3};
       margin:0;max-width:70ch;">Não é um dispositivo médico; não diagnostica,
       cura nem trata qualquer condição. Os valores deste relatório descrevem
       uma única sessão de curta duração em ambiente de festival e não
       substituem avaliação profissional.</p>
    <span style="font-family:{MONO};font-size:10px;letter-spacing:0.14em;
          text-transform:uppercase;color:{TEXT_3};">
      {_esc(meta["footerCode"])}</span>
  </div>

</div>
</div>
"""

    style = f"""
<title>{_esc(copy["title"])}</title>
<link rel="stylesheet" href="{FONTS}">
<style>
  :root {{ color-scheme: dark; }}
  html {{ background: {INK_900}; }}
  body {{ margin: 0; background: {INK_900}; color: {TEXT_2};
         font-family: {SANS}; -webkit-font-smoothing: antialiased; }}
  * {{ box-sizing: border-box; }}
  .om-page {{ background: {INK_900}; padding: 32px 16px 64px;
              display: flex; justify-content: center; }}
  .om-sheet {{ width: 100%; max-width: 880px; background: {INK_700};
               border: 1px solid {BORDER}; border-radius: 10px;
               overflow: hidden; }}
  /* Uma só regra para a goteira lateral: cada secção herda-a e só mexe no
     eixo vertical. Sem isto, catorze secções com padding-shorthand acabavam
     por discordar entre si no telemóvel. */
  .om-pad {{ padding-left: 40px; padding-right: 40px; }}
  .om-meta {{ display: grid;
              grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
              gap: 1px; background: {DIVIDER}; border: 1px solid {BORDER};
              border-radius: 6px; overflow: hidden; margin-top: 32px; }}
  .om-cards-3 {{ display: grid;
                 grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
                 gap: 12px; }}
  .om-cards-2 {{ display: grid;
                 grid-template-columns: repeat(auto-fit, minmax(258px, 1fr));
                 gap: 12px; }}
  .om-glossary {{ display: grid;
                  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
                  gap: 18px 28px; }}
  @media (max-width: 640px) {{
    .om-page {{ padding: 12px 8px 32px; }}
    .om-pad {{ padding-left: 20px; padding-right: 20px; }}
  }}
  @media print {{
    .om-page {{ padding: 0; }}
    .om-sheet {{ border: 0; max-width: none; }}
  }}
</style>
"""

    if fragment:
        return style + body
    return (
        "<!doctype html>\n<html lang=\"pt-PT\">\n<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"{style}</head>\n<body>{body}</body>\n</html>\n"
    )
