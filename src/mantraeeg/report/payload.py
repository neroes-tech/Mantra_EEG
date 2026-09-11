"""Os dados do relatório do participante, numa só estrutura.

Este módulo faz a ponte entre o que a análise produz (``AnalysisResult``, em
unidades cruas, com buracos e ressalvas) e o que o relatório mostra. **Não
desenha nada** — devolve um dicionário, que serve tanto o HTML como o corpo do
email, e é o que se guarda em ``report.json`` para a sessão ser reproduzível
sem voltar a correr a análise.

Três regras que governam o que sai daqui, todas por o output ser lido por
pessoas que vão tirar conclusões sobre si próprias:

- **Um marcador cujo IC cruza zero é apresentado como estável**, com o valor,
  e nunca com linguagem de mudança. É a maioria dos marcadores numa sessão de
  poucos minutos, e é a leitura correta.
- **Quantidades com sinal não levam percentagem.** O ALAY passar de −0,030 a
  −0,048 sai "+57 %" e lê-se como melhoria quando é a direção oposta.
- **O que foi interpolado viaja com o número.** A cobertura de cada marcador
  está no payload e é impressa.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..spectral import welch_psd

from ..analysis import (
    PHASE_ACTIVE,
    PHASE_BASELINE,
    AnalysisResult,
    MarkerSummary,
    baseline_spread as _baseline_spread,
    percent_change,
    rank_by_variation,
)
from ..config import Config
from .audience import MAD_TO_SD, SMOOTH_EPOCHS, _smooth, prepare

#: Paleta do design system da Neroes, por papel. O verde e o latão são as duas
#: cores semânticas; o teal é o acento e serve o que está estável.
TEAL = "#43BEC3"
GREEN = "#5BBC99"
BRASS = "#D4A455"
BLUE = "#5B8AD4"
MUTED = "#6E8B93"

#: Tinta de fundo das pastilhas de variação, sobre o cartão ``--ink-600``.
TINT = {
    GREEN: "rgba(91, 188, 153, 0.14)",
    BRASS: "rgba(212, 164, 85, 0.14)",
    TEAL: "rgba(67, 190, 195, 0.12)",
    MUTED: "rgba(169, 191, 197, 0.08)",
}


@dataclass(frozen=True)
class ReportContext:
    """O que não vem dos dados: quem, quando, e o que se ouviu."""

    participant_name: str
    session_started: datetime
    duration_s: float
    #: Nome do mantra, ou ``None`` quando a sessão correu em silêncio. Não há
    #: default: uma sessão de controlo não pode sair do relatório com o nome
    #: de um mantra que não tocou.
    mantra_label: str | None
    session_code: str
    festival_label: str


def _data_uri(path: Path) -> str:
    """Uma imagem embebida. Vazio se o ficheiro não existir — sem exceção."""
    try:
        raw = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    except OSError:
        return ""
    suffix = Path(path).suffix.lower().lstrip(".") or "png"
    return f"data:image/{'jpeg' if suffix in ('jpg', 'jpeg') else suffix};base64,{raw}"


def _fmt(value: float, decimals: int) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{value:.{decimals}f}".replace(".", ",")


def _j(value: float) -> float | None:
    """JSON não tem NaN. Ausência é ``null``, e o renderer omite."""
    return float(value) if np.isfinite(value) else None


def _decimals(marker_id: str, value: float) -> int:
    if not np.isfinite(value):
        return 2
    magnitude = abs(value)
    if magnitude >= 100:
        return 0
    if magnitude >= 10:
        return 1
    return 3 if magnitude < 0.1 else 2


def _delta_colour(summary: MarkerSummary) -> str:
    """Verde só quando há efeito **e** vai no sentido esperado.

    Um marcador estável fica em cinzento, não em verde pálido: a cor é
    informação, e pintar de verde o que não se mediu é a forma mais rápida de
    o relatório dizer uma coisa falsa sem escrever uma única palavra errada.
    """
    if not summary.reliable:
        return MUTED
    if summary.direction_ok is None:
        return TEAL
    return GREEN if summary.direction_ok else BRASS


def _sparkline_values(
    result: AnalysisResult, marker_id: str, cfg: Config
) -> tuple[list[float], float | None]:
    """A trajetória de um marcador ao longo da sessão, sem buracos.

    Cada cartão mostra a **sua** série completa, e não a janela comum de
    ``audience.prepare`` — aqui não há eixo partilhado a defender, e cortar a
    série pelo pior marcador do conjunto esconderia dados bons.
    """
    series = result.series_of(marker_id)
    if series is None:
        return [], None
    values = np.asarray(series.values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 3:
        return [], None
    filled = np.interp(series.times_s, series.times_s[finite], values[finite])
    filled = _smooth(filled, SMOOTH_EPOCHS)

    active = np.asarray(series.phases) == PHASE_ACTIVE
    fraction = float(np.argmax(active)) / len(filled) if active.any() else None
    return [float(v) for v in filled], fraction


def _spectrum(result: AnalysisResult, cfg: Config) -> dict[str, Any]:
    """O espectro médio do sinal da pessoa, nos canais de métrica.

    **Descritivo, não comparativo.** Não depende de uma linha de base nem de
    haver diferença entre fases, portanto continua a ser verdade quando a
    comparação deixa de se sustentar. É o que se mostra a alguém cuja
    gravação apanhou o palco ao lado: isto é o teu sinal, e é mesmo teu.

    Só entram as épocas que passaram no gate. Com poucas, a curva é ruidosa —
    e é suposto parecer ruidosa.
    """
    pre = result.preprocessed
    if pre is None:
        return {}
    quality = result.quality
    rows = [
        quality.name_to_row[n]
        for n in cfg.montage.metric_channels
        if n in quality.name_to_row
    ]
    if not rows:
        return {}

    good = quality.channel_ok[rows].all(axis=0)
    starts = quality.starts[good]
    if starts.size == 0:
        return {}

    win = quality.win_samples
    segments = [
        pre.clean[rows, start : start + win]
        for start in starts
        if start + win <= pre.clean.shape[1]
    ]
    if not segments:
        return {}

    freqs, psd = welch_psd(
        np.concatenate(segments, axis=1), pre.sfreq, 4.0, 0.5
    )
    band = (1.0, 40.0)
    keep = (freqs >= band[0]) & (freqs <= band[1])
    mean_psd = np.nanmean(psd[:, keep], axis=0)
    if not np.any(np.isfinite(mean_psd)) or np.nanmax(mean_psd) <= 0:
        return {}

    return {
        "freqs": [round(float(f), 2) for f in freqs[keep]],
        # Em dB: o espectro cai como 1/f e numa escala linear as bandas
        # rápidas ficavam coladas ao eixo.
        "db": [round(float(10 * np.log10(max(v, 1e-12))), 2) for v in mean_psd],
        "bands": {
            "theta": list(cfg.bands["theta"]),
            "alpha": list(cfg.bands["alpha"]),
            "beta": list(cfg.bands["beta"]),
        },
        "nEpochs": int(starts.size),
    }


def _coverage_trace(result: AnalysisResult, cfg: Config) -> dict[str, Any]:
    """Onde, ao longo da sessão, houve leitura fiável.

    O que se mostra quando não há mais nada honesto para mostrar: isto foi o
    que conseguimos gravar, e estes foram os bocados aproveitáveis.
    """
    quality = result.quality
    rows = [
        quality.name_to_row[n]
        for n in cfg.montage.metric_channels
        if n in quality.name_to_row
    ]
    if not rows or quality.times_s.size == 0:
        return {}
    good = quality.channel_ok[rows].all(axis=0)
    return {
        "times": [round(float(t), 1) for t in quality.times_s],
        "ok": [bool(v) for v in good],
    }


def _interpretation(summary: MarkerSummary, cfg: Config) -> str:
    """O que a medida capta, e — só quando houve — o que mudou.

    A ressalva de "manteve-se dentro do normal" saiu daqui. Aparecia em oito
    de oito cartões, dizia a mesma coisa oito vezes, e a leitura da sessão já
    a diz uma vez em prosa. O mesmo para o aviso de tensão muscular: repetido
    em cada cartão vira ruído e deixa de se ler.
    """
    what = summary.marker.explanation
    if not np.isfinite(summary.baseline):
        return f"{what} Não houve leitura suficiente nesta sessão para o medir."
    if not summary.reliable:
        return what

    direction = "subiu" if summary.active > summary.baseline else "desceu"
    verdict = (
        f"Nesta sessão {direction}, e a diferença é maior do que a variação "
        f"normal do teu próprio sinal."
    )
    if summary.direction_ok is False:
        verdict += " O sentido é o oposto do que a literatura descreve."
    return f"{what} {verdict}"


GLOSSARY = [
    (
        "Calibração",
        "Os primeiros minutos da sessão, sentado e quieto. Servem para medir "
        "como é o teu sinal quando não está a acontecer nada.",
    ),
    (
        "Repouso",
        "O estado medido durante a calibração. É a referência com que tudo o "
        "resto é comparado — a tua, e não a média de outras pessoas.",
    ),
    (
        "Alfa",
        "O ritmo de 8 a 12 oscilações por segundo. Domina quando o cérebro "
        "está desperto mas em repouso, sem se esforçar por resolver nada.",
    ),
    (
        "Beta",
        "O ritmo de 15 a 30 oscilações por segundo. Acompanha o pensamento "
        "ativo: resolver, planear, antecipar.",
    ),
    (
        "Teta",
        "O ritmo de 4 a 7 oscilações por segundo. Aparece com a atenção "
        "virada para dentro, e também com a sonolência.",
    ),
    (
        "Sincronia",
        "O grau em que dois pontos do escalpe oscilam em conjunto na mesma "
        "frequência.",
    ),
]


def _reading_low_signal(
    context: ReportContext, coverage: float, tier: str
) -> list[str]:
    """A leitura quando a comparação não se sustenta.

    Não é a leitura normal com uma ressalva colada: é outra leitura. A frase
    "nenhum biomarcador mudou" está errada aqui — não é que nada mudou, é que
    **nada pôde ser medido**, e as duas coisas não se dizem da mesma maneira a
    quem esteve oito minutos sentado.
    """
    percent = f"{coverage * 100:.0f}%"
    ambient = (
        "Numa banca de festival isto acontece: palcos, colunas e "
        "instrumentação elétrica a poucos metros injetam no sensor um sinal "
        "muito maior do que o do cérebro, e basta um elétrodo a perder "
        "contacto com o couro cabeludo para a leitura desse ponto se perder."
    )

    if tier == "descriptive":
        return [
            f"O sensor deu leitura fiável em {percent} do tempo. É pouco para "
            f"comparar o antes com o durante — uma comparação precisa de uma "
            f"referência sólida dos dois lados, e aqui não a há. O que se "
            f"segue não são variações: é o teu sinal, tal como foi gravado.",
            ambient,
            "O que está nos gráficos é real e é teu: o espectro mostra em que "
            "ritmos a tua atividade se concentrou, e a linha mostra onde "
            "houve leitura ao longo da sessão. O que não fazemos é dizer-te "
            "que subiu ou desceu, porque com esta quantidade de sinal isso "
            "seria uma frase inventada.",
            "Se quiseres o relatório completo, vale a pena repetir a sessão "
            "com os sensores bem molhados e afastados da fonte de ruído. São "
            "os mesmos minutos, e a diferença na leitura é enorme.",
        ]

    return [
        f"Não foi possível medir a tua atividade cerebral nesta sessão: o "
        f"sensor só deu leitura fiável em {percent} do tempo.",
        ambient,
        "Por isso não há números neste relatório. Podíamos apresentar-te "
        "alguns — mas com o sensor sem contacto os valores seriam ruído "
        "ambiente, não a tua atividade, e duas pessoas diferentes receberiam "
        "resultados diferentes gerados pela mesma ausência de sinal. "
        "Preferimos dizer-te isto.",
        "O gráfico em baixo mostra o que conseguimos gravar. Se puderes "
        "voltar à banca, repetimos a sessão — leva os mesmos minutos e "
        "normalmente basta molhar bem os sensores.",
    ]


def _join(names: list[str]) -> str:
    """«a, b e c» — a vírgula serial não existe em português."""
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " e " + names[-1]


def _reading(
    result: AnalysisResult,
    context: ReportContext,
    coverage: float,
    cfg: Config,
    metrics: list[dict[str, Any]],
    tier: str = "full",
) -> list[str]:
    """A leitura da sessão, em prosa, construída a partir dos números.

    **Isto não é escrito por um modelo de linguagem.** É uma função de regras
    sobre os resultados: que marcadores foram fiáveis, em que sentido se
    moveram, quanto, quais foram sinalizados como muscular, e que cobertura
    teve o sinal. Corre dentro do executável, offline, sem chamar nada.

    A personalização vem de os números serem os da pessoa — cada frase é
    escolhida por um teste sobre esses números, e não por geração livre. É
    também o que garante que o relatório de dois participantes com o mesmo
    resultado diz a mesma coisa, que num contexto onde alguém tira conclusões
    sobre si próprio importa mais do que a variedade de fraseado.

    A ressalva do n=1 não é uma nota de rodapé: é o parágrafo que impede a
    pessoa de sair da banca com uma conclusão que os dados não sustentam.
    """
    # Só o que o relatório mostra. Os `extra` não aparecem em lado nenhum na
    # página, e nomeá-los aqui mandava a pessoa procurar um cartão que não
    # existe.
    shown = {m["id"] for m in metrics if m["role"] == "marker"}
    if tier != "full":
        return _reading_low_signal(context, coverage, tier)

    by_id = {m["id"]: m for m in metrics}
    reliable = [
        s for s in result.summaries if s.reliable and s.marker.id in shown
    ]
    total = len(shown)

    def friendly(summary: MarkerSummary) -> str:
        return summary.marker.friendly or summary.marker.label

    def with_change(summary: MarkerSummary) -> str:
        entry = by_id.get(summary.marker.id)
        return (
            f"{friendly(summary)} ({entry['pctText']})"
            if entry
            else friendly(summary)
        )

    # 1 — o que se mediu.
    if reliable:
        up = [s for s in reliable if s.active > s.baseline]
        down = [s for s in reliable if s.active <= s.baseline]
        pieces = []
        if up:
            pieces.append(f"subiram {_join([with_change(s) for s in up])}")
        if down:
            pieces.append(f"desceram {_join([with_change(s) for s in down])}")
        first = (
            f"De {total} biomarcadores medidos, {len(reliable)} mudaram mais do "
            f"que a tua própria oscilação em repouso: {'; '.join(pieces)}. Os "
            f"restantes {total - len(reliable)} mantiveram-se estáveis — o "
            f"resultado mais comum numa sessão de poucos minutos, e que não "
            f"significa que não tenha acontecido nada: significa que a "
            f"diferença foi menor do que o ruído do próprio sinal."
        )
    else:
        first = (
            f"Nenhum dos {total} biomarcadores mudou mais do que a tua própria "
            f"oscilação em repouso. Numa sessão de poucos minutos isto é o "
            f"resultado mais comum, e é a leitura honesta: o sinal não mostra "
            f"diferença suficiente para se afirmar seja o que for."
        )

    # 2 — o aviso muscular, quando existe. Vem antes da explicação do desenho
    # porque é sobre a validade dos números que se acabaram de ler.
    flagged = [
        s for s in reliable if abs(s.r_emg) > cfg.diagnostics.r_emg_red_threshold
    ]
    if flagged:
        first += (
            f" Uma ressalva sobre {_join([friendly(s) for s in flagged])}: a "
            f"variação acompanhou de perto a tensão dos músculos da mandíbula. "
            f"Esses músculos produzem no elétrodo um sinal muito maior do que "
            f"o do cérebro, por isso essa leitura pode não ser cerebral."
        )

    # 3 — contra o que foi comparado.
    if context.mantra_label is None:
        second = (
            "Esta foi uma sessão de controlo: silêncio do princípio ao fim, "
            "sem mantra. Serve para saber o que o sinal faz sozinho, que é a "
            "única forma de mais tarde se poder dizer que o mantra fez alguma "
            "coisa."
        )
    else:
        second = (
            f"A comparação é entre os primeiros minutos em repouso e os "
            f"minutos com «{context.mantra_label}». As duas partes foram "
            f"gravadas de olhos fechados de propósito: abrir e fechar os olhos "
            f"multiplica sozinho o alfa por duas a cinco vezes, e se as "
            f"condições fossem diferentes seria isso que o relatório estaria a "
            f"medir, e não o mantra."
        )

    # 4 — o que uma sessão não pode dizer.
    third = (
        "Uma sessão, uma pessoa, sem grupo de comparação. O que está aqui "
        "descreve o que o teu sinal fez nestes minutos — não permite atribuir "
        "a causa ao que estavas a ouvir, nem prever o que aconteceria noutro "
        "dia. Para isso são precisas várias sessões."
    )

    # 5 — a qualidade do registo.
    if coverage >= 0.97:
        quality = (
            f"O sensor deu leitura fiável em {coverage * 100:.0f}% do tempo, "
            f"o que é um registo limpo."
        )
    elif coverage >= 0.8:
        quality = (
            f"O sensor deu leitura fiável em {coverage * 100:.0f}% do tempo. "
            f"O resto foi descartado, não preenchido."
        )
    else:
        quality = (
            f"O sensor só deu leitura fiável em {coverage * 100:.0f}% do "
            f"tempo — contacto irregular, movimento ou tensão muscular. Os "
            f"números que se seguem assentam em menos dados do que o normal e "
            f"devem ser lidos com essa reserva."
        )
    fourth = (
        f"{quality} Nas linhas do relatório os pontos estão unidos para se ver "
        f"a tendência, mas nenhum número assenta em sinal inventado."
    )
    return [first, second, third, fourth]


def build(
    result: AnalysisResult,
    cfg: Config,
    context: ReportContext,
    featured: Iterable[str],
    phases: list[tuple[str, float, float]] | None = None,
) -> dict[str, Any]:
    """O payload do relatório. Sem I/O, sem desenho."""
    featured_ids = list(featured)
    started = context.session_started
    minutes, seconds = divmod(int(round(context.duration_s)), 60)

    # A ordem do relatório é a da magnitude da mudança — a mesma de
    # ``rank_by_variation``, para o destaque, as barras e os cartões contarem
    # a mesma história. ``maxMetrics`` corta a cauda: oito cartões, seis dos
    # quais a dizer "manteve-se estável", cansam antes de informar.
    order = rank_by_variation(result, cfg, limit=len(result.summaries))
    position = {marker_id: i for i, marker_id in enumerate(order)}
    ordered = sorted(
        result.summaries,
        key=lambda s: (
            s.marker.role != "marker",
            position.get(s.marker.id, len(order)),
        ),
    )

    metrics: list[dict[str, Any]] = []
    for summary in ordered:
        marker = summary.marker
        decimals = _decimals(marker.id, summary.baseline)
        spread = _baseline_spread(result, marker.id)
        percent = percent_change(result, summary)
        standardised = (
            (summary.active - summary.baseline) / spread
            if np.isfinite(spread)
            else float("nan")
        )
        values, mantra_at = _sparkline_values(result, marker.id, cfg)
        colour = _delta_colour(summary)
        metrics.append(
            {
                "id": marker.id,
                "name": marker.friendly or marker.label,
                "basis": marker.label,
                "unit": marker.unit,
                "baseline": _fmt(summary.baseline, decimals),
                "value": _fmt(summary.active, decimals),
                "delta": summary.change_text(),
                #: Percentagem face ao repouso, ou ``None`` quando a linha de
                #: base está perto demais de zero para uma razão significar
                #: alguma coisa. O relatório mostra a diferença absoluta nesses.
                "pct": _j(percent),
                "pctText": (
                    f"{percent:+.0f}%".replace("-", "−")
                    if np.isfinite(percent)
                    else summary.change_text()
                ),
                #: Percentagem desproporcionada face ao que a sustenta. Sai
                #: mais pequena no cartao e nao entra nas barras — ver
                #: report.large_pct na configuracao.
                "extreme": bool(
                    np.isfinite(percent) and abs(percent) > cfg.report.large_pct
                ),
                "stdChange": _j(standardised),
                "deltaColor": colour,
                "deltaBg": TINT[colour],
                "reliable": bool(summary.reliable),
                "coverage": _j(summary.coverage),
                "n": f"leitura fiável em {summary.coverage * 100:.0f}% do tempo",
                "series": values,
                "mantraAt": mantra_at,
                "featured": marker.id in featured_ids,
                "rank": position.get(marker.id, 999),
                "role": marker.role,
                "interpretation": _interpretation(summary, cfg),
            }
        )

    by_id = {m["id"]: m for m in metrics}
    # No destaque não entra a unidade nem o valor absoluto: "fração de 1–45 Hz"
    # não diz nada a quem acabou de tirar o headset da cabeça. O que a pessoa
    # quer ver é quanto mudou face ao seu próprio repouso.
    headline = [
        {
            "label": by_id[i]["name"],
            "change": by_id[i]["pctText"],
            "extreme": by_id[i]["extreme"],
            "color": by_id[i]["deltaColor"],
            "reliable": by_id[i]["reliable"],
        }
        for i in featured_ids
        if i in by_id
    ]

    reliable = [s for s in result.summaries if s.reliable]
    coverage = float(np.nanmean([s.coverage for s in result.summaries]))
    if coverage >= cfg.report.tier_full:
        tier = "full"
    elif coverage >= cfg.report.tier_descriptive:
        tier = "descriptive"
    else:
        tier = "insufficient"

    # Gráfico da sessão: eixo comum, sem buracos, em unidades da variabilidade
    # da própria calibração. É a vista do participante — a de investigação,
    # com os buracos, fica em report/figures.py e não entra aqui.
    chart: dict[str, Any] = {"series": [], "t_start": 0.0, "t_end": 0.0}
    if phases:
        audience = prepare(result, featured_ids, phases)
        chart = {
            "t_start": audience.t_start,
            "t_end": audience.t_end,
            "mantraAt": audience.mantra_start_s,
            "coverage": audience.coverage,
            "series": [
                {
                    "id": s.marker_id,
                    "label": s.label,
                    "times": [float(t) for t in s.times_s],
                    "deviation": [float(v) for v in s.deviation],
                }
                for s in audience.series
            ],
        }

    return {
        "meta": {
            "code": context.session_code,
            # Embebido aqui e não procurado pelo renderer: dentro do
            # executável não há pasta tpl/, e um logótipo em falta é um
            # cabeçalho partido no relatório de um participante.
            "logo": _data_uri(cfg.paths.logo_lockup),
            "festivalLabel": context.festival_label,
            "footerCode": (
                f"Relatório {context.session_code} · gerado "
                f"{datetime.now():%d/%m/%Y %H:%M}"
            ),
        },
        "participant": {"name": context.participant_name},
        "session": {
            "datetime": f"{started:%d/%m/%Y · %H:%M}",
            "duration": f"{minutes:02d}:{seconds:02d} min",
            "mantra": context.mantra_label or "Nenhum — sessão em silêncio",
            "silent": context.mantra_label is None,
            "coverage": coverage,
            "activeLabel": "silêncio" if context.mantra_label is None else "mantra",
        },
        "copy": {
            "lowSignal": tier != "full",
            "title": (
                "O que o teu sinal fez em silêncio"
                if context.mantra_label is None
                else "O impacto do mantra no teu cérebro"
            ),
            "intro": (
                "Este relatório reúne o que o headset registou enquanto "
                + (
                    "estiveste sentado em silêncio, de olhos fechados. "
                    if context.mantra_label is None
                    else f"ouviste «{context.mantra_label}», de olhos fechados. "
                )
                + f"{sum(1 for s in result.summaries if s.marker.role == 'marker')} "
                "biomarcadores cerebrais, cada um "
                "comparado com o teu próprio estado de repouso — e não com uma "
                "média de outras pessoas."
            ),
        },
        #: "full" | "descriptive" | "insufficient". Decide o que a página
        #: mostra — ver report/html.py.
        "tier": tier,
        "spectrum": _spectrum(result, cfg) if tier != "insufficient" else {},
        "coverageTrace": _coverage_trace(result, cfg),
        "headline": headline if tier == "full" else [],
        #: Quantos cartões e barras o relatório mostra. Os restantes ficam em
        #: report.json, para investigação — o corte é de apresentação.
        "maxMetrics": int(cfg.report.max_metrics),
        "chart": chart,
        "metrics": metrics,
        "reading": _reading(result, context, coverage, cfg, metrics, tier),
        "glossary": [{"term": t, "def": d} for t, d in GLOSSARY],
        "counts": {
            "total": len(result.summaries),
            "reliable": len(reliable),
            "markers": sum(1 for s in result.summaries if s.marker.role == "marker"),
        },
        "notes": list(result.notes),
    }
