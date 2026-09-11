"""Gate de qualidade por época, e os índices com que os marcadores se comparam.

Um marcador só é calculado se **todos** os canais de que depende estiverem bons
nessa época; caso contrário ``nan``. Nunca se interpola sobre épocas rejeitadas
(SPEC 3.3).

Os índices de EMG e de rede são medidos sobre a cópia **só detrended**, nunca
sobre o sinal filtrado a 1–45 Hz. Depois do passa-banda a banda da rede está
atenuada dezenas de dB e o índice nunca dispararia — foi um bug real, apanhado
na revisão da spec.

``emg_index`` e ``motion_index`` **não são marcadores**: são as séries com que
cada marcador é correlacionado na tabela de diagnóstico. Um marcador cuja série
correlaciona com o EMG está a medir músculo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats as sp_stats

from .config import Config
from .preprocess import Preprocessed
from .spectral import relative_power, welch_psd


@dataclass
class EpochQuality:
    """Máscara de qualidade por amostra e por canal, mais os índices."""

    #: ``(n_ch, n_epochs)`` booleano: canal utilizável naquela época.
    channel_ok: np.ndarray
    #: Início de cada época de potência, em amostras.
    starts: np.ndarray
    win_samples: int
    #: Índices por época, para as correlações de diagnóstico.
    emg: np.ndarray
    emg_wide: np.ndarray
    line: np.ndarray
    motion: np.ndarray
    names: tuple[str, ...]
    name_to_row: dict[str, int]
    #: Centro de cada época de potência, em segundos. Permite alinhar os
    #: índices com marcadores que vivam noutra grelha (a conectividade corre a
    #: 10 s / 5 s, contra 4 s / 2 s da potência).
    times_s: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def window_ok(self, start: int, stop: int, channels: tuple[str, ...]) -> bool:
        """Todos os canais pedidos estão bons em todas as épocas da janela?

        Uma janela de conectividade cobre várias épocas de potência; exige-se
        que **todas** estejam boas, que é a regra mais restritiva de B10.
        """
        covered = (self.starts >= start - self.win_samples) & (self.starts < stop)
        if not covered.any():
            return False
        rows = [self.name_to_row[c] for c in channels if c in self.name_to_row]
        if len(rows) != len(channels):
            return False
        return bool(self.channel_ok[np.ix_(rows, covered)].all())

    def correlate(
        self, values: np.ndarray, index: str, times_s: np.ndarray | None = None
    ) -> float:
        """Spearman entre a série de um marcador e um índice de artefacto.

        É a coluna mais importante da tabela de diagnóstico: um marcador que
        correlaciona com o EMG está a medir músculo, por muito bonito que o
        gráfico seja.

        Quando o marcador vive noutra grelha — a conectividade corre a 10 s com
        passo de 5 s, contra 4 s e 2 s da potência — o índice é **interpolado**
        para os instantes do marcador. Sem isto os marcadores de conectividade
        ficavam sem ``r_emg``, que é precisamente onde a contaminação muscular
        seria mais difícil de detetar a olho.
        """
        reference = {
            "emg": self.emg,
            "emg_wide": self.emg_wide,
            "motion": self.motion,
            "line": self.line,
        }[index]

        if times_s is not None and values.size != reference.size:
            if self.times_s.size == 0:
                return float("nan")
            finite = np.isfinite(reference)
            if finite.sum() < 2:
                return float("nan")
            reference = np.interp(
                times_s, self.times_s[finite], reference[finite],
                left=np.nan, right=np.nan,
            )
        if values.size != reference.size:
            return float("nan")

        mask = np.isfinite(values) & np.isfinite(reference)
        if mask.sum() < 5:
            return float("nan")
        return float(sp_stats.spearmanr(values[mask], reference[mask]).statistic)


def motion_index(accel: np.ndarray, gyro: np.ndarray) -> float:
    """Desvio da aceleração face a 1 g, mais a magnitude do giroscópio."""
    magnitude = np.linalg.norm(np.asarray(accel, dtype=np.float64), axis=0)
    linear = float(np.mean(np.abs(magnitude - 1.0)))
    rotational = float(np.mean(np.abs(np.asarray(gyro, dtype=np.float64))))
    return linear + rotational / 100.0


def epoch_quality(
    pre: Preprocessed,
    cfg: Config,
    accel: np.ndarray | None = None,
    gyro: np.ndarray | None = None,
) -> EpochQuality:
    """Avalia cada época de potência, canal a canal."""
    sfreq = pre.sfreq
    grid = cfg.epochs.power
    win = int(round(grid.win_s * sfreq))
    hop = int(round(grid.hop_s * sfreq))
    starts = np.arange(0, max(pre.n_samples - win + 1, 0), hop)

    montage = cfg.montage
    names = tuple(montage.channels[i] for i in sorted(montage.channels))
    name_to_row = {name: i for i, name in enumerate(names)}
    q = cfg.quality

    n_ch = len(names)
    channel_ok = np.zeros((n_ch, len(starts)), dtype=bool)
    emg = np.full(len(starts), np.nan)
    emg_wide = np.full(len(starts), np.nan)
    line = np.full(len(starts), np.nan)
    motion = np.full(len(starts), np.nan)

    for i, start in enumerate(starts):
        stop = start + win
        clean = pre.clean[:n_ch, start:stop]
        # Índices sobre a cópia com NOTCH mas sem passa-banda.
        #
        # Sem passa-banda porque o corte a 45 Hz atenuaria o topo da banda de
        # EMG (30-45). Com notch porque o que interessa ao gate é o resíduo de
        # rede que sobrevive à filtragem, não quanta rede o local tem: medido
        # antes do notch, o índice chega a 21 000 e rejeitava 100 % das épocas.
        rough = pre.notched[:n_ch, start:stop]
        freqs, psd_rough = welch_psd(rough, sfreq, grid.welch_seg_s, grid.welch_overlap)

        emg_ch = relative_power(psd_rough, freqs, cfg.bands["emg"], cfg.bands.total)
        wide_ch = relative_power(
            psd_rough, freqs, cfg.bands["emg_wide"], cfg.bands.total
        )
        line_ch = relative_power(psd_rough, freqs, cfg.bands["line"], cfg.bands.total)
        emg[i] = float(np.nanmedian(emg_ch))
        emg_wide[i] = float(np.nanmedian(wide_ch))
        line[i] = float(np.nanmedian(line_ch))

        moved = False
        if accel is not None and gyro is not None and stop <= accel.shape[1]:
            offset = pre.edge_samples
            a = accel[:, offset + start : offset + stop]
            g = gyro[:, offset + start : offset + stop]
            if a.size and g.size:
                magnitude = np.linalg.norm(a, axis=0)
                moved = bool(
                    np.any(np.abs(magnitude - 1.0) > q.motion.accel_dev_g)
                    or np.any(np.abs(g) > q.motion.gyro_dps)
                )
                motion[i] = motion_index(a, g)

        if moved and q.motion.invalidates_all_channels:
            continue

        amplitude = np.abs(clean).max(axis=1)
        step = np.abs(np.diff(clean, axis=1)).max(axis=1)
        deviation = clean.std(axis=1)
        channel_ok[:, i] = (
            (amplitude <= q.max_abs_uv)
            & (step <= q.max_step_uv)
            & (deviation >= q.min_std_uv)
            & (np.nan_to_num(emg_ch, nan=1.0) <= q.max_emg_rel)
            & (np.nan_to_num(line_ch, nan=1.0) <= q.max_line_rel)
        )

    return EpochQuality(
        channel_ok=channel_ok,
        starts=starts,
        win_samples=win,
        emg=emg,
        emg_wide=emg_wide,
        line=line,
        motion=motion,
        names=names,
        name_to_row=name_to_row,
        times_s=pre.t0_s + (starts + win / 2.0) / sfreq,
    )
