"""A página da coorte: o que aconteceu, em média, em todas as sessões.

Feita para uma plateia de conferência que não é de neurociência. A regra de
composição é a mesma do relatório individual — cada gráfico mostra uma coisa,
cada número traz consigo quantas sessões o sustentam — com uma diferença: aqui
o resultado principal pode muito bem ser **"ainda não sabemos"**, e a página
tem de conseguir dizer isso sem parecer um fracasso.

O gráfico central não é uma média com uma barra de erro. É um ponto por
sessão, sobre a mediana e o intervalo: quem está na plateia vê ao mesmo tempo
a tendência e a dispersão, e percebe sozinho porque é que seis sessões não
chegam para afirmar nada. Uma barra sozinha esconderia exatamente isso.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import numpy as np

from ..cohort import CohortResult, MarkerAggregate
from ..config import Config
from .html import (
    BORDER,
    DIVIDER,
    FONTS,
    INK_600,
    INK_900,
    INK_HEAD,
    INK_700,
    MONO,
    SANS,
    SERIF,
    TEXT_1,
    TEXT_2,
    TEXT_3,
    _esc,
    _eyebrow,
)

UP = "#5BBC99"
DOWN = "#E8944D"
NEUTRAL = "#6E8B93"


def _logo(cfg: Config) -> str:
    try:
        raw = base64.b64encode(Path(cfg.paths.logo_lockup).read_bytes()).decode("ascii")
    except OSError:
        return ""
    return f"data:image/png;base64,{raw}"


def _forest(result: CohortResult) -> str:
    """Um ponto por sessão, a mediana, e o intervalo. Nada de barras sozinhas."""
    rows = [a for a in result.aggregates if a.n_sessions >= 3]
    if not rows:
        return ""

    row_h = 46.0
    width = 860.0
    left, right, top = 210.0, 30.0, 26.0
    height = top + len(rows) * row_h + 46.0
    plot_w = width - left - right

    values = [v for a in rows for v in a.values_std]
    limit = max(max(abs(v) for v in values), 1.0) * 1.08

    def x_of(v: float) -> float:
        return left + (v + limit) / (2 * limit) * plot_w

    parts = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" role="img" '
        f'aria-label="Variação por biomarcador, uma sessão por ponto" '
        f'style="display:block;overflow:visible;">'
    ]

    for tick in (-2, -1, 0, 1, 2):
        if abs(tick) > limit:
            continue
        x = x_of(tick)
        zero = tick == 0
        parts.append(
            f'<line x1="{x:.1f}" y1="{top - 8:.1f}" x2="{x:.1f}" '
            f'y2="{top + len(rows) * row_h:.1f}" stroke="rgba(169,191,197,'
            f'{0.34 if zero else 0.10})" stroke-width="{2 if zero else 1}"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{height - 26:.1f}" text-anchor="middle" '
            f'fill="{TEXT_3}" font-family="{MONO}" font-size="10">'
            f'{"sem mudança" if zero else f"{tick:+d}"}</text>'
        )

    for i, aggregate in enumerate(rows):
        y = top + i * row_h + row_h / 2
        colour = UP if aggregate.median_std > 0 else DOWN
        if not aggregate.consistent:
            colour = NEUTRAL

        parts.append(
            f'<text x="{left - 16:.1f}" y="{y + 4:.1f}" text-anchor="end" '
            f'fill="{TEXT_1}" font-family="{SANS}" font-size="13.5">'
            f"{_esc(aggregate.marker.friendly)}</text>"
        )

        if np.isfinite(aggregate.ci_low):
            parts.append(
                f'<line x1="{x_of(max(aggregate.ci_low, -limit)):.1f}" y1="{y:.1f}" '
                f'x2="{x_of(min(aggregate.ci_high, limit)):.1f}" y2="{y:.1f}" '
                f'stroke="{colour}" stroke-width="1.5" opacity="0.35"/>'
            )

        # Um ponto por sessão, espalhados na vertical para não se taparem.
        for j, value in enumerate(aggregate.values_std):
            offset = (j - (len(aggregate.values_std) - 1) / 2) * 5.0
            parts.append(
                f'<circle cx="{x_of(max(-limit, min(limit, value))):.1f}" '
                f'cy="{y + offset:.1f}" r="3.2" fill="{colour}" opacity="0.5"/>'
            )

        parts.append(
            f'<circle cx="{x_of(aggregate.median_std):.1f}" cy="{y:.1f}" r="6" '
            f'fill="{colour}" stroke="{INK_900}" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{width - right + 4:.1f}" y="{y + 4:.1f}" '
            f'fill="{TEXT_3}" font-family="{MONO}" font-size="10">'
            f"{aggregate.n_sessions}</text>"
        )

    parts.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 6:.1f}" '
        f'text-anchor="middle" fill="{TEXT_3}" font-family="{MONO}" '
        f'font-size="9.5" letter-spacing="0.14em">'
        f"MUDANÇA, EM MÚLTIPLOS DA OSCILAÇÃO NORMAL DE CADA PESSOA</text>"
    )
    parts.append("</svg>")

    legend = (
        f'<div style="display:flex;flex-wrap:wrap;gap:20px;margin-top:12px;'
        f'padding-top:14px;border-top:1px solid {DIVIDER};font-family:{MONO};'
        f'font-size:10.5px;letter-spacing:0.1em;text-transform:uppercase;'
        f'color:{TEXT_2};">'
        f'<span>• ponto pequeno = uma sessão</span>'
        f'<span>● ponto grande = o valor típico</span>'
        f'<span>— traço = margem de incerteza</span>'
        f"</div>"
    )
    return (
        f'<div style="background:{INK_900};border:1px solid {BORDER};'
        f'border-radius:10px;padding:20px 22px 14px;overflow-x:auto;">'
        f'{"".join(parts)}{legend}</div>'
    )


def _table(result: CohortResult) -> str:
    head = (
        f'<div style="display:grid;grid-template-columns:1.6fr 0.7fr 0.8fr 1.1fr;'
        f'gap:10px;padding:0 16px 8px;font-family:{MONO};font-size:9px;'
        f'letter-spacing:0.2em;text-transform:uppercase;color:{TEXT_3};">'
        f"<span>Biomarcador</span><span>Sessões</span><span>Variação</span>"
        f"<span>Mesmo sentido</span></div>"
    )
    rows = []
    for a in result.aggregates:
        percent = (
            f"{a.median_percent:+.0f}%".replace("-", "−")
            if np.isfinite(a.median_percent)
            else f"{a.median_std:+.2f}".replace("-", "−")
        )
        colour = UP if a.median_std > 0 else DOWN
        if not a.consistent:
            colour = TEXT_2
        agreement = f"{max(a.n_up, a.n_down)} de {a.n_sessions}"
        rows.append(
            f'<div style="display:grid;'
            f'grid-template-columns:1.6fr 0.7fr 0.8fr 1.1fr;gap:10px;'
            f'align-items:center;background:{INK_600};padding:13px 16px;">'
            f'<span style="font-family:{SANS};font-size:13.5px;color:{TEXT_1};">'
            f"{_esc(a.marker.friendly)}</span>"
            f'<span style="font-family:{MONO};font-size:12px;color:{TEXT_3};">'
            f"{a.n_sessions}</span>"
            f'<span style="font-family:{MONO};font-size:13px;color:{colour};'
            f'font-variant-numeric:tabular-nums;">{percent}</span>'
            f'<span style="font-family:{MONO};font-size:12px;color:{TEXT_2};">'
            f"{agreement}</span></div>"
        )
    return (
        head
        + f'<div style="display:flex;flex-direction:column;gap:1px;'
        f'background:{DIVIDER};border:1px solid {BORDER};border-radius:10px;'
        f'overflow:hidden;">{"".join(rows)}</div>'
    )


def _quality(result: CohortResult) -> str:
    cells = []
    for session in sorted(result.sessions, key=lambda s: s.name):
        included = session in result.included
        colour = UP if included else "#C4685A"
        cells.append(
            f'<div style="background:{INK_600};border:1px solid {BORDER};'
            f'border-radius:8px;padding:12px 14px;display:flex;'
            f'flex-direction:column;gap:5px;">'
            f'<span style="font-family:{MONO};font-size:9.5px;'
            f'letter-spacing:0.14em;color:{TEXT_3};">'
            f'{_esc(session.name.split("_")[0])} · '
            f'{session.duration_s / 60:.0f} MIN</span>'
            f'<span style="font-family:{SANS};font-size:19px;font-weight:600;'
            f'color:{colour};font-variant-numeric:tabular-nums;">'
            f"{session.coverage * 100:.0f}%</span>"
            f'<span style="font-family:{MONO};font-size:9px;'
            f'letter-spacing:0.12em;text-transform:uppercase;color:{TEXT_3};">'
            f'{"incluída" if included else "excluída"}</span></div>'
        )
    return (
        f'<div style="display:grid;'
        f'grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;">'
        f'{"".join(cells)}</div>'
    )


def _headline(result: CohortResult) -> tuple[str, str]:
    """A frase de topo, construída a partir do que os dados sustentam."""
    consistent = [a for a in result.aggregates if a.consistent]
    n = len(result.included)
    if not consistent:
        return (
            "Ainda não sabemos",
            f"Em {n} sessões com sinal utilizável, nenhum dos biomarcadores se "
            f"moveu sempre no mesmo sentido. Não é um resultado nulo — é um "
            f"conjunto de dados pequeno demais para distinguir um efeito real "
            f"da variação normal de quem está sentado e quieto.",
        )
    names = ", ".join(a.marker.friendly for a in consistent)
    return (
        "O que se repetiu",
        f"Em {n} sessões, {names} moveu-se sempre no mesmo sentido. Continua a "
        f"faltar um braço de comparação em silêncio para se saber se é o "
        f"mantra ou apenas o descanso.",
    )


def render(result: CohortResult, cfg: Config, title: str = "") -> str:
    """A página completa, autónoma."""
    heading, lead = _headline(result)
    logo = _logo(cfg)
    sessions = len(result.included)
    total = len(result.sessions)
    mantras = {
        "sri_vitthala": "Sri Vitthala Giridhari Parabhramane Namah",
        "om_namo": "Om Namo Narayanaya",
    }
    used = ", ".join(mantras.get(m, m) for m in result.mantras) or "—"
    minutes = sum(s.duration_s for s in result.included) / 60

    degenerate = ""
    if result.n_degenerate:
        degenerate = (
            f" Outras {result.n_degenerate} leituras isoladas saíram por a "
            f"referência daquela medida ter ficado indistinguível de zero: "
            f"dividir por ela dá um número enorme que não é uma mudança, é "
            f"uma divisão por quase-nada."
        )

    stat = (
        lambda label, value: f'<div style="background:{INK_700};padding:16px 18px;'
        f'display:flex;flex-direction:column;gap:6px;">'
        f'<span style="font-family:{MONO};font-size:9px;font-weight:500;'
        f'letter-spacing:0.22em;text-transform:uppercase;color:{TEXT_3};">'
        f"{_esc(label)}</span>"
        f'<span style="font-family:{SANS};font-size:15px;color:{TEXT_1};">'
        f"{_esc(value)}</span></div>"
    )

    body = f"""
<div class="om-page">
<div class="om-sheet">

  <div style="background:{INK_HEAD};padding:22px 40px;display:flex;
       align-items:center;justify-content:space-between;gap:24px;flex-wrap:wrap;">
    {f'<img src="{logo}" alt="neroes" style="height:46px;width:auto;display:block;">' if logo else ''}
    <span style="font-family:{MONO};font-size:10px;font-weight:500;
          letter-spacing:0.22em;text-transform:uppercase;color:{TEXT_3};">
      Análise de coorte</span>
  </div>

  <div class="om-pad" style="padding-top:44px;padding-bottom:36px;
       border-bottom:1px solid {DIVIDER};">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;">
      <span style="width:6px;height:6px;border-radius:50%;background:#2E9296;
            display:block;flex:none;"></span>
      <span style="font-family:{MONO};font-size:10px;font-weight:500;
            letter-spacing:0.22em;text-transform:uppercase;color:#2E9296;">
        {_esc(title or "O cérebro a ouvir mantras")}</span>
    </div>
    <h1 style="font-family:{SANS};font-size:clamp(30px,6vw,46px);font-weight:600;
        letter-spacing:-0.025em;line-height:1.1;color:{TEXT_1};margin:0 0 18px;
        text-wrap:balance;">{_esc(heading)}</h1>
    <p style="font-family:{SANS};font-size:16px;line-height:1.6;color:{TEXT_2};
       margin:0;max-width:60ch;text-wrap:pretty;">{_esc(lead)}</p>

    <div class="om-meta">
      {stat("Sessões analisadas", f"{total}")}
      {stat("Com sinal utilizável", f"{sessions}")}
      {stat("Minutos de gravação", f"{minutes:.0f}")}
      {stat("Mantras", used)}
    </div>
  </div>

  <div class="om-pad" style="padding-top:40px;">
    {_eyebrow("01", "O que mudou, e quanto")}
    <p style="font-family:{SANS};font-size:15px;line-height:1.6;color:{TEXT_2};
       margin:-6px 0 20px;max-width:64ch;text-wrap:pretty;">
      Cada linha é um biomarcador. Cada ponto pequeno é uma sessão. O ponto
      grande é o valor típico. À direita da linha central houve subida, à
      esquerda descida — e a distância mede-se em múltiplos daquilo que o
      sinal daquela pessoa já oscilava sozinho, em repouso. É o que permite
      pôr medidas de unidades diferentes lado a lado.</p>
    {_forest(result)}
    <p style="font-family:{SANS};font-size:14px;line-height:1.6;color:{TEXT_3};
       margin:18px 0 0;max-width:64ch;text-wrap:pretty;">
      Repare na dispersão dos pontos: em quase todas as linhas há sessões dos
      dois lados do zero. É essa dispersão, e não a posição do ponto grande,
      que diz o que este conjunto de dados consegue sustentar.</p>
  </div>

  <div class="om-pad" style="padding-top:36px;">
    {_eyebrow("02", "Os números")}
    {_table(result)}
  </div>

  <div class="om-pad" style="padding-top:36px;">
    {_eyebrow("03", "A qualidade do sinal, sessão a sessão")}
    <p style="font-family:{SANS};font-size:15px;line-height:1.6;color:{TEXT_2};
       margin:-6px 0 20px;max-width:64ch;text-wrap:pretty;">
      A percentagem é o tempo em que os sensores deram leitura fiável.
      Entraram na análise as sessões acima de {result.min_coverage:.0%}. As
      restantes {result.n_excluded} não foram corrigidas nem suavizadas —
      foram postas de lado, porque uma gravação de elétrodos soltos não
      acrescenta ruído neutro, acrescenta variação inventada.{degenerate}</p>
    {_quality(result)}
  </div>

  <div class="om-pad" style="padding-top:36px;">
    {_eyebrow("04", "O que isto não mostra")}
    <div style="display:flex;flex-direction:column;gap:16px;max-width:66ch;">
      <p style="font-family:{SANS};font-size:15px;line-height:1.68;
         color:{TEXT_2};margin:0;text-wrap:pretty;">
        <b style="color:{TEXT_1};">Falta o braço de silêncio.</b> Todas as
        sessões incluídas tiveram mantra. Sem as mesmas pessoas a passar os
        mesmos minutos sentadas e caladas, qualquer variação encontrada pode
        ser o efeito de descansar de olhos fechados — que sozinho já muda o
        sinal de forma substancial.</p>
      <p style="font-family:{SANS};font-size:15px;line-height:1.68;
         color:{TEXT_2};margin:0;text-wrap:pretty;">
        <b style="color:{TEXT_1};">Quem se senta na banca escolhe-se a si
        próprio.</b> São {sessions} pessoas diferentes, uma sessão cada — o
        que faz de cada uma uma observação independente, e é o que permite
        contar sentidos como se conta. Mas são pessoas que passaram por um
        festival de bem-estar e decidiram experimentar um sensor de EEG. O
        que se encontrar aqui descreve esse grupo, e não a população.</p>
      <p style="font-family:{SANS};font-size:15px;line-height:1.68;
         color:{TEXT_2};margin:0;text-wrap:pretty;">
        <b style="color:{TEXT_1};">Seis sessões não chegam.</b> Para um
        sentido se repetir mais do que o acaso explica, com este número de
        sessões, teria de acontecer em todas elas. É um critério exigente de
        propósito: é o único que este conjunto de dados sustenta.</p>
    </div>
  </div>

  <div class="om-pad" style="padding-top:36px;padding-bottom:40px;">
    <div style="border-top:1px solid {BORDER};border-bottom:1px solid {BORDER};
         padding:32px 0;display:flex;gap:16px;align-items:flex-start;">
      <span style="width:7px;height:7px;border-radius:50%;background:#B8873C;
            display:block;margin-top:12px;flex:none;"></span>
      <p style="font-family:{SERIF};font-style:italic;
         font-size:clamp(19px,4.4vw,24px);line-height:1.42;color:{TEXT_1};
         margin:0;max-width:40ch;text-wrap:pretty;">Medir com honestidade é
         mais lento do que medir com entusiasmo, e é a única forma de o
         resultado significar alguma coisa.</p>
    </div>
  </div>

  <div class="om-pad" style="padding-top:32px;padding-bottom:40px;
       background:{INK_HEAD};border-top:1px solid {DIVIDER};display:flex;
       flex-direction:column;gap:18px;">
    <span style="font-family:{MONO};font-size:10px;font-weight:500;
          letter-spacing:0.22em;text-transform:uppercase;color:{TEXT_3};">
      Lisboa · Mental Training Platform</span>
    <p style="font-family:{SANS};font-size:12px;line-height:1.6;color:{TEXT_3};
       margin:0;max-width:72ch;">Demonstração científica em ambiente de
       festival, com elétrodos secos e sem sala blindada. Não é um dispositivo
       médico e não diagnostica, cura nem trata qualquer condição. Os dados
       descrevem as sessões recolhidas e não se generalizam para além
       delas.</p>
  </div>

</div>
</div>
"""

    style = f"""
<title>O cérebro a ouvir mantras</title>
<link rel="stylesheet" href="{FONTS}">
<style>
  :root {{ color-scheme: dark; }}
  html {{ background: {INK_900}; }}
  body {{ margin: 0; background: {INK_900}; color: {TEXT_2};
         font-family: {SANS}; -webkit-font-smoothing: antialiased; }}
  * {{ box-sizing: border-box; }}
  .om-page {{ background: {INK_900}; padding: 32px 16px 64px;
              display: flex; justify-content: center; }}
  .om-sheet {{ width: 100%; max-width: 920px; background: {INK_700};
               border: 1px solid {BORDER}; border-radius: 10px;
               overflow: hidden; }}
  .om-pad {{ padding-left: 40px; padding-right: 40px; }}
  .om-meta {{ display: grid;
              grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
              gap: 1px; background: {DIVIDER}; border: 1px solid {BORDER};
              border-radius: 6px; overflow: hidden; margin-top: 32px; }}
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
    return (
        '<!doctype html>\n<html lang="pt-PT">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"{style}</head>\n<body>{body}</body>\n</html>\n"
    )
