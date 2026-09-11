"""Carregamento e validação da configuração.

Tudo o que é limiar, banda, duração, peso ou parâmetro de janela vive em
``config/default.yaml``. Este módulo transforma-o em dataclasses imutáveis e
**valida na carga**, não em uso: um YAML incoerente rebenta ao arrancar a
aplicação, não a meio de uma sessão com uma pessoa sentada na cadeira.

Contém também a aritmética de grelha (:func:`n_epochs`,
:meth:`Config.reference_adequacy`) partilhada pelo menu da UI e por
``normalize.py``, para que as duas contas não possam divergir.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

from .apppaths import resolve as resolve_path

from .bands import Bands

RefQuality = Literal["ok", "thin", "insufficient"]
SourceName = Literal["brainflow", "unicorn_dll", "lsl", "replay", "synthetic"]
EyeCondition = Literal["open", "closed"]

_VALID_SOURCES = ("brainflow", "unicorn_dll", "lsl", "replay", "synthetic")


class ConfigError(ValueError):
    """A configuração é inválida. Levantada na carga."""


# --------------------------------------------------------------------------- #
# Aritmética de grelha
# --------------------------------------------------------------------------- #
def n_epochs(duration_s: float, win_s: float, hop_s: float) -> int:
    """Número de épocas completas que cabem em ``duration_s``.

    Uma janela deslizante de ``win_s`` com passo ``hop_s`` produz
    ``floor((duration - win) / hop) + 1`` épocas, ou zero se nem uma janela
    inteira couber. É esta a conta que decide se uma calibração curta consegue
    sustentar uma referência — ver :meth:`Config.reference_adequacy`.
    """
    if duration_s < win_s:
        return 0
    return int((duration_s - win_s) // hop_s) + 1


# --------------------------------------------------------------------------- #
# Secções
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DeviceCfg:
    source: SourceName
    sfreq_nominal: float
    n_device_channels: int
    line_freq_hz: float
    settle_discard_s: float
    stale_after_s: float
    ring_buffer_s: float
    read_chunk_s: float
    brainflow: Mapping[str, Any]
    unicorn_dll: Mapping[str, Any]
    lsl: Mapping[str, Any]
    replay: Mapping[str, Any]

    def resolved_dll_paths(self) -> list[Path]:
        """Candidatos a ``Unicorn.dll``, com ``$ENV:VAR`` já resolvido (SPEC 8.2)."""
        out: list[Path] = []
        for raw in self.unicorn_dll.get("dll_paths", []):
            text = str(raw)
            if text.startswith("$ENV:"):
                value = os.environ.get(text[len("$ENV:") :], "")
                if not value:
                    continue
                text = value
            out.append(Path(text))
        # O BrainFlow traz a sua própria Unicorn.dll. Vale como último
        # candidato: numa máquina do evento sem o Unicorn Suite instalado, é
        # a única que existe, e é o que permite listar o número de série
        # (que dá o nome à pasta da sessão) sem instalar nada da g.tec.
        try:
            import brainflow

            bundled = Path(brainflow.__file__).parent / "lib" / "Unicorn.dll"
            if bundled not in out:
                out.append(bundled)
        except Exception:
            pass
        return out


@dataclass(frozen=True)
class MontageValidationCfg:
    enabled: bool
    tap_test_seconds_per_channel: float
    tap_test_min_ratio: float
    auto_check_window_s: float
    blink_band_hz: tuple[float, float]
    frontopolar_dominance_min_ratio: float
    no_contact_min_interchannel_r: float
    no_contact_max_pair_diff_ratio: float


@dataclass(frozen=True)
class MontageCfg:
    """Montagem personalizada (SPEC 1.2).

    Nunca ler nomes de canal do dispositivo. O BrainFlow devolve a montagem
    *default* do Unicorn (``Fz, C3, Cz, C4, Pz, PO7, Oz, PO8``), que não é esta;
    usá-la trocaria F3 com F4 sem qualquer aviso.
    """

    channels: Mapping[int, str]
    metric_channels: tuple[str, ...]
    eog_channels: tuple[str, ...]
    unused_channels: tuple[str, ...]
    reference: str
    apply_car: bool
    min_good_channels_to_start: int
    validation: MontageValidationCfg

    @property
    def name_to_index(self) -> dict[str, int]:
        """Mapa posição -> índice no buffer do dispositivo."""
        return {name: idx for idx, name in self.channels.items()}

    @property
    def n_eeg(self) -> int:
        return len(self.channels)

    def index_of(self, name: str) -> int:
        try:
            return self.name_to_index[name]
        except KeyError:
            raise KeyError(
                f"posição {name!r} não está na montagem; disponíveis: "
                f"{sorted(self.name_to_index)}"
            ) from None


@dataclass(frozen=True)
class CuesCfg:
    """Deixas de início e fim de fase.

    De olhos fechados a pessoa não vê o ecrã: as deixas têm de ser áudio.
    """

    mode: str
    start_sound: Path
    end_sound: Path


@dataclass(frozen=True)
class ProtocolCfg:
    calibration_s: float
    calib_use_last_s: float
    mantra_s: float
    settle_s: float
    settle_as_baseline_min_s: float
    n_mantra_thirds: int
    eye_calibration: EyeCondition
    eye_mantra: EyeCondition
    eye_settle: EyeCondition
    cues: CuesCfg

    @property
    def settle_is_baseline(self) -> bool:
        """O segmento pós-mantra é longo o suficiente para servir de 2ª linha de base."""
        return self.settle_s >= self.settle_as_baseline_min_s

    @property
    def eye_condition(self) -> EyeCondition:
        """A condição ocular da sessão.

        Só é bem definida porque a carga da config garante que a calibração e a
        meditação coincidem — ver :func:`load_config`. Comparar fases de
        condições diferentes é proibido pela SPEC 2.
        """
        return self.eye_calibration

    def eye_for(self, phase: str) -> EyeCondition:
        try:
            return {
                "calibration": self.eye_calibration,
                "mantra": self.eye_mantra,
                "settle": self.eye_settle,
            }[phase]
        except KeyError:
            raise KeyError(f"fase desconhecida: {phase!r}") from None


@dataclass(frozen=True)
class EogRegressionCfg:
    enabled: bool
    regressor_band_hz: tuple[float, float]
    use_mean_diff_regressors: bool
    fit_on_blink_segments_only: bool
    blink_detect_uv: float
    blink_pad_s: float
    max_band_variance_removed: float
    run_control_pass: bool


@dataclass(frozen=True)
class PreprocessCfg:
    detrend: str
    notch_freqs_hz: tuple[float, ...]
    notch_q: float
    notch_passes: int
    bandpass_hz: tuple[float, float]
    bandpass_order: int
    filter_edge_s: float
    notch_redundant_atten_db: float
    eog_regression: EogRegressionCfg


@dataclass(frozen=True)
class EpochGridCfg:
    """Uma grelha de epocagem. Potência e conectividade têm cadências distintas."""

    win_s: float
    hop_s: float
    welch_seg_s: float
    welch_overlap: float
    window: str = "hann"
    min_segments: int = 0

    def n_epochs(self, duration_s: float) -> int:
        return n_epochs(duration_s, self.win_s, self.hop_s)

    def n_welch_segments(self) -> int:
        """Sub-segmentos de Welch dentro de uma janela desta grelha."""
        step = self.welch_seg_s * (1.0 - self.welch_overlap)
        if step <= 0:
            raise ConfigError("welch_overlap tem de ser < 1")
        return n_epochs(self.win_s, self.welch_seg_s, step)


@dataclass(frozen=True)
class EpochsCfg:
    power: EpochGridCfg
    connectivity: EpochGridCfg


@dataclass(frozen=True)
class MotionCfg:
    accel_dev_g: float
    gyro_dps: float
    invalidates_all_channels: bool


@dataclass(frozen=True)
class QualityCfg:
    max_abs_uv: float
    max_step_uv: float
    min_std_uv: float
    max_emg_rel: float
    max_line_rel: float
    residual_blink_uv: float
    indices_from_unfiltered: bool
    min_calibration_coverage: float
    motion: MotionCfg


@dataclass(frozen=True)
class ContactLevelCfg:
    max_std_uv: float
    max_p2p_uv: float


@dataclass(frozen=True)
class ContactMonitorCfg:
    update_period_s: float
    window_s: float
    detrend_window: bool
    trust_device_validation_flag: bool
    grade_on_filtered: bool
    min_std_uv: float
    max_drift_uv_per_s: float
    line_warn_ratio: float
    slope_fit_range: tuple[float, float]
    slope_welch_seg_s: float
    plausible_slope_range: tuple[float, float]
    green: ContactLevelCfg
    yellow: ContactLevelCfg


@dataclass(frozen=True)
class NormalizeCfg:
    statistic: str
    mad_consistency_constant: float
    mad_fallback_iqr_constant: float
    min_epochs_ok: int
    min_epochs_thin: int
    min_windows_ok: int
    min_windows_thin: int


@dataclass(frozen=True)
class BlockBootstrapCfg:
    method: str
    length_selection: str
    fixed_length_s: float
    min_length_s: float
    max_length_s: float
    acf_threshold: float
    min_effective_samples: float


@dataclass(frozen=True)
class NullCheckCfg:
    enabled: bool
    max_false_positive_rate: float


@dataclass(frozen=True)
class SurrogatesCfg:
    enabled: bool
    n_surrogates: int
    method: str


@dataclass(frozen=True)
class StatsCfg:
    inference_on: str
    bootstrap_n: int
    ci_level: float
    resample_calibration: bool
    berger_min_log2: float
    berger_min_reactivity: float
    berger_min_channel_agreement: float
    berger_flanks: tuple[tuple[float, float], ...]
    theil_sen: bool
    stability_odd_even: bool
    stability_halves: bool
    block_bootstrap: BlockBootstrapCfg
    null_check: NullCheckCfg
    connectivity_surrogates: SurrogatesCfg


@dataclass(frozen=True)
class CompositeCfg:
    weights_eyes_open: Mapping[str, float]
    weights_eyes_closed: Mapping[str, float]
    renormalize_on_nan: bool

    def weights_for(self, eye: EyeCondition) -> Mapping[str, float]:
        return self.weights_eyes_open if eye == "open" else self.weights_eyes_closed


@dataclass(frozen=True)
class MarkersCfg:
    ap_exponent: Mapping[str, Any]
    lziv: Mapping[str, Any]
    spectral_entropy: Mapping[str, Any]


@dataclass(frozen=True)
class DiagnosticsCfg:
    r_emg_red_threshold: float
    r_motion_red_threshold: float
    r_luminance_red_threshold: float
    correlation_method: str


@dataclass(frozen=True)
class AggregateCfg:
    min_sessions: int
    consistency_binomial_ci: bool


@dataclass(frozen=True)
class ReportCfg:
    profile: str
    featured_markers: tuple[str, ...]
    featured_mode: str
    featured_count: int
    max_metrics: int
    language: str
    dpi: int
    figsize_in: tuple[float, float]
    participant_footer: str


@dataclass(frozen=True)
class UiCfg:
    inspector_trace_s: float
    inspector_trace_period_s: float
    single_screen: bool
    title: str
    subtitle: str
    shortcuts: Mapping[str, str]
    marker_keys: Mapping[str, str]
    validation_pair: tuple[str, str]
    participant_screen: Mapping[str, Any]
    operator_screen: Mapping[str, Any]
    email: Mapping[str, Any]


@dataclass(frozen=True)
class DriveCfg:
    enabled: bool
    client_id: str
    client_secret: str
    folder_id: str
    scope: str
    token_file: Path
    delete_after_upload: bool
    max_attempts: int
    base_backoff_s: float

    @property
    def configured(self) -> bool:
        """Há o suficiente para tentar: cliente OAuth e pasta de destino."""
        return bool(self.enabled and self.client_id and self.folder_id)


@dataclass(frozen=True)
class PublishCfg:
    """Onde a página do relatório é publicada para o link abrir."""

    owner: str
    repo: str
    branch: str
    folder: str
    #: Resolvido na carga, da variável de ambiente ou do ficheiro. Nunca vem
    #: escrito no YAML — o YAML viaja nos pacotes que se copiam.
    token: str
    base_url: str

    @property
    def configured(self) -> bool:
        return bool(self.owner and self.repo and self.token)


@dataclass(frozen=True)
class MantraCfg:
    """Um mantra disponível para escolha no painel do investigador."""

    id: str
    label: str
    file: Path


@dataclass(frozen=True)
class PathsCfg:
    drive_folder_name: str
    sessions_dir: Path
    aggregate_dir: Path
    assets_dir: Path
    #: Os mantras oferecidos, pela ordem em que aparecem no seletor.
    mantras: tuple[MantraCfg, ...]
    #: O ``id`` do que está pré-selecionado.
    default_mantra: str
    mantra_audio: Path | None
    mantra_start_offset_s: float
    logo: Path
    logo_lockup: Path

    def mantra(self, mantra_id: str | None) -> MantraCfg:
        """O mantra por id, com o default como recurso.

        Nunca levanta: se o id guardado numa sessão antiga já não existir no
        registo, a banca abre com o default em vez de não abrir.
        """
        for entry in self.mantras:
            if entry.id == mantra_id:
                return entry
        return next(
            (m for m in self.mantras if m.id == self.default_mantra),
            self.mantras[0],
        )


@dataclass(frozen=True)
class ReferenceAdequacy:
    """O que uma dada duração de calibração compra, por grelha.

    Produzido por :meth:`Config.reference_adequacy` e mostrado ao operador no
    menu **antes** de gravar, para que o compromisso entre tempo de fila e
    qualidade da referência seja feito com números à frente.
    """

    calib_total_s: float
    calib_used_s: float
    n_power_epochs: int
    n_conn_windows: int
    power_quality: RefQuality
    conn_quality: RefQuality

    def quality_for(self, kind: str) -> RefQuality:
        """``kind`` é ``"pair"`` para marcadores de conectividade, o resto é potência."""
        return self.conn_quality if kind == "pair" else self.power_quality

    def describe(self) -> str:
        return (
            f"Calibração {self.calib_total_s:.0f} s "
            f"-> últimos {self.calib_used_s:.0f} s úteis\n"
            f"  potência:  {self.n_power_epochs:3d} épocas   [{self.power_quality}]\n"
            f"  coerência: {self.n_conn_windows:3d} janelas  [{self.conn_quality}]"
        )


# --------------------------------------------------------------------------- #
# Config de topo
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    version: int
    device: DeviceCfg
    montage: MontageCfg
    bands: Bands
    protocol: ProtocolCfg
    preprocess: PreprocessCfg
    epochs: EpochsCfg
    quality: QualityCfg
    contact_monitor: ContactMonitorCfg
    normalize: NormalizeCfg
    stats: StatsCfg
    composite: CompositeCfg
    markers: MarkersCfg
    diagnostics: DiagnosticsCfg
    aggregate: AggregateCfg
    report: ReportCfg
    ui: UiCfg
    drive: DriveCfg
    paths: PathsCfg
    publish: PublishCfg
    source_path: Path | None = field(default=None, compare=False)

    # -- aritmética de grelha ------------------------------------------------ #
    def reference_adequacy(
        self,
        calibration_s: float | None = None,
        calib_use_last_s: float | None = None,
    ) -> ReferenceAdequacy:
        """Quantas épocas e janelas uma calibração compra, e se chegam.

        As durações são editáveis na interface, portanto nada pode assumir um
        número de épocas: mede-se o que a duração escolhida comprou. Um marcador
        cuja grelha saia ``insufficient`` devolve ``nan`` para a sessão inteira,
        em vez de um número fabricado a partir de meia dúzia de janelas.
        """
        total = self.protocol.calibration_s if calibration_s is None else calibration_s
        use_last = (
            self.protocol.calib_use_last_s
            if calib_use_last_s is None
            else calib_use_last_s
        )
        used = min(total, use_last)

        n_pow = self.epochs.power.n_epochs(used)
        n_con = self.epochs.connectivity.n_epochs(used)
        nz = self.normalize

        def grade(n: int, ok: int, thin: int) -> RefQuality:
            if n >= ok:
                return "ok"
            if n >= thin:
                return "thin"
            return "insufficient"

        return ReferenceAdequacy(
            calib_total_s=total,
            calib_used_s=used,
            n_power_epochs=n_pow,
            n_conn_windows=n_con,
            power_quality=grade(n_pow, nz.min_epochs_ok, nz.min_epochs_thin),
            conn_quality=grade(n_con, nz.min_windows_ok, nz.min_windows_thin),
        )


# --------------------------------------------------------------------------- #
# Carga
# --------------------------------------------------------------------------- #
def _pair(value: Any, what: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(f"{what}: esperado par [a, b], obtido {value!r}")
    return float(value[0]), float(value[1])


def _reject_redundant_notches(pp: "PreprocessCfg", sfreq: float) -> None:
    """Rejeita notches que o passa-banda já elimina sozinho.

    Avalia a resposta real do Butterworth desenhado à frequência do notch e
    compara com ``notch_redundant_atten_db``. Um notch que sobreviva a esta
    verificação está a fazer trabalho; um que não sobreviva só custa tempo.
    """
    import numpy as np
    from scipy import signal

    nyq = sfreq / 2.0
    lo, hi = pp.bandpass_hz
    sos = signal.butter(
        pp.bandpass_order, [lo / nyq, hi / nyq], btype="bandpass", output="sos"
    )
    for f in pp.notch_freqs_hz:
        if f >= nyq:
            raise ConfigError(
                f"notch a {f} Hz está em ou acima de Nyquist ({nyq} Hz)"
            )
        w, h = signal.sosfreqz(sos, worN=[f], fs=sfreq)
        # sosfiltfilt aplica o filtro duas vezes: a atenuação em potência dobra em dB.
        atten_db = -2.0 * 20.0 * np.log10(max(abs(h[0]), 1e-12))
        if atten_db > pp.notch_redundant_atten_db:
            raise ConfigError(
                f"notch a {f} Hz é redundante: o passa-banda "
                f"({lo}-{hi} Hz, ordem {pp.bandpass_order}, fase zero) já atenua "
                f"{atten_db:.0f} dB, acima do limite de "
                f"{pp.notch_redundant_atten_db:.0f} dB; remova-o de "
                f"preprocess.notch_freqs_hz"
            )


def _grid(raw: Mapping[str, Any], what: str) -> EpochGridCfg:
    grid = EpochGridCfg(
        win_s=float(raw["win_s"]),
        hop_s=float(raw["hop_s"]),
        welch_seg_s=float(raw["welch_seg_s"]),
        welch_overlap=float(raw["welch_overlap"]),
        window=str(raw.get("window", "hann")),
        min_segments=int(raw.get("min_segments", 0)),
    )
    if grid.hop_s > grid.win_s:
        raise ConfigError(f"epochs.{what}: hop_s ({grid.hop_s}) > win_s ({grid.win_s})")
    if grid.welch_seg_s > grid.win_s:
        raise ConfigError(
            f"epochs.{what}: welch_seg_s ({grid.welch_seg_s}) > win_s ({grid.win_s})"
        )
    return grid


def _drive_secret(dr: Mapping[str, Any]) -> str:
    """O segredo do cliente OAuth, de fora do YAML.

    A Google diz que para aplicações instaladas isto não é realmente um
    segredo — vai embutido em cada cópia distribuída. Mas o repositório é
    público, e o GitHub bloqueia um `GOCSPX-` num push antes de olhar para a
    nuance. Fica ao lado dos outros segredos, em ``data/``, que está no
    ``.gitignore``.

    Ordem: variável de ambiente, ficheiro, e por fim o que estiver no YAML —
    para uma instalação antiga com o valor inline continuar a arrancar.
    """
    env = str(dr.get("client_secret_env", "")).strip()
    if env:
        value = os.environ.get(env, "").strip()
        if value:
            return value
    file_name = str(dr.get("client_secret_file", "")).strip()
    if file_name:
        path = resolve_path(file_name)
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    return str(dr.get("client_secret", "") or "")


def load_config(path: str | Path = "config/default.yaml") -> Config:
    """Carrega e valida o YAML. Levanta :class:`ConfigError` se for incoerente.

    Os caminhos relativos que estão dentro do ficheiro — ``assets/``,
    ``data/sessions``, os mantras — são resolvidos contra a raiz da aplicação
    (:func:`apppaths.app_root`) e não contra o diretório de trabalho. É o que
    faz o executável encontrar os seus ficheiros quando alguém o abre com
    duplo clique a partir do ambiente de trabalho.
    """
    path = resolve_path(path)
    if not path.exists():
        raise ConfigError(f"config não encontrada: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"config não é um mapa YAML: {path}")

    d = raw["device"]
    source = str(d["source"])
    if source not in _VALID_SOURCES:
        raise ConfigError(f"device.source {source!r} inválido; use um de {_VALID_SOURCES}")
    device = DeviceCfg(
        source=source,  # type: ignore[arg-type]
        sfreq_nominal=float(d["sfreq_nominal"]),
        n_device_channels=int(d["n_device_channels"]),
        line_freq_hz=float(d["line_freq_hz"]),
        settle_discard_s=float(d["settle_discard_s"]),
        stale_after_s=float(d["stale_after_s"]),
        ring_buffer_s=float(d["ring_buffer_s"]),
        read_chunk_s=float(d["read_chunk_s"]),
        brainflow=dict(d["brainflow"]),
        unicorn_dll=dict(d["unicorn_dll"]),
        lsl=dict(d["lsl"]),
        replay=dict(d["replay"]),
    )

    m = raw["montage"]
    channels = {int(k): str(v) for k, v in m["channels"].items()}
    if len(set(channels.values())) != len(channels):
        raise ConfigError(f"montage.channels tem posições repetidas: {channels}")
    expected_idx = set(range(len(channels)))
    if set(channels) != expected_idx:
        raise ConfigError(
            f"montage.channels tem de cobrir os índices {sorted(expected_idx)}, "
            f"tem {sorted(channels)}"
        )
    v = m["validation"]
    montage = MontageCfg(
        channels=channels,
        metric_channels=tuple(m["metric_channels"]),
        eog_channels=tuple(m["eog_channels"]),
        unused_channels=tuple(m["unused_channels"]),
        reference=str(m["reference"]),
        apply_car=bool(m["apply_car"]),
        min_good_channels_to_start=int(m["min_good_channels_to_start"]),
        validation=MontageValidationCfg(
            enabled=bool(v["enabled"]),
            tap_test_seconds_per_channel=float(v["tap_test_seconds_per_channel"]),
            tap_test_min_ratio=float(v["tap_test_min_ratio"]),
            auto_check_window_s=float(v["auto_check_window_s"]),
            blink_band_hz=_pair(v["blink_band_hz"], "montage.validation.blink_band_hz"),
            frontopolar_dominance_min_ratio=float(v["frontopolar_dominance_min_ratio"]),
            no_contact_min_interchannel_r=float(v["no_contact_min_interchannel_r"]),
            no_contact_max_pair_diff_ratio=float(v["no_contact_max_pair_diff_ratio"]),
        ),
    )
    known = set(channels.values())
    for group in ("metric_channels", "eog_channels", "unused_channels"):
        unknown = set(getattr(montage, group)) - known
        if unknown:
            raise ConfigError(
                f"montage.{group} refere posições que não estão em channels: "
                f"{sorted(unknown)}"
            )
    if montage.apply_car:
        raise ConfigError(
            "montage.apply_car está ligado; a SPEC 1.2 proíbe CAR sobre estes "
            "canais anteriores agrupados (distorce assimetria e coerência)"
        )

    bands = Bands.from_config(raw["bands"])

    p = raw["protocol"]
    eyes: dict[str, str] = {}
    for phase in ("calibration", "mantra", "settle"):
        value = str(p[f"eye_{phase}"])
        if value not in ("open", "closed"):
            raise ConfigError(
                f"protocol.eye_{phase} {value!r} inválido ('open'|'closed')"
            )
        eyes[phase] = value
    # A invariante ocular da SPEC 2, verificada em vez de suposta. Fechar as
    # pálpebras multiplica o alfa por 2 a 5x; comparar uma calibração de olhos
    # abertos com uma meditação de olhos fechados produz uma "melhoria" que é
    # inteiramente efeito de Berger.
    if eyes["calibration"] != eyes["mantra"]:
        raise ConfigError(
            f"invariante ocular violada: a calibração é de olhos "
            f"{eyes['calibration']} e a meditação de olhos {eyes['mantra']}. "
            f"Não são comparáveis — o efeito de Berger multiplica o alfa por 2 a "
            f"5x só por fechar as pálpebras, muito acima de qualquer efeito de "
            f"meditação. Ver SPEC 2 e BIOMARKERS.md."
        )
    cues_raw = p["cues"]
    cues_mode = str(cues_raw["mode"])
    if cues_mode not in ("audio", "visual", "both"):
        raise ConfigError(f"protocol.cues.mode {cues_mode!r} inválido")
    if eyes["calibration"] == "closed" and cues_mode == "visual":
        raise ConfigError(
            "protocol.cues.mode é 'visual' mas a calibração é de olhos fechados; "
            "a pessoa não veria a deixa. Use 'audio' ou 'both'."
        )
    protocol = ProtocolCfg(
        calibration_s=float(p["calibration_s"]),
        calib_use_last_s=float(p["calib_use_last_s"]),
        mantra_s=float(p["mantra_s"]),
        settle_s=float(p["settle_s"]),
        settle_as_baseline_min_s=float(p["settle_as_baseline_min_s"]),
        n_mantra_thirds=int(p["n_mantra_thirds"]),
        eye_calibration=eyes["calibration"],  # type: ignore[arg-type]
        eye_mantra=eyes["mantra"],  # type: ignore[arg-type]
        eye_settle=eyes["settle"],  # type: ignore[arg-type]
        cues=CuesCfg(
            mode=cues_mode,
            start_sound=Path(cues_raw["start_sound"]),
            end_sound=Path(cues_raw["end_sound"]),
        ),
    )
    if protocol.calib_use_last_s > protocol.calibration_s:
        raise ConfigError(
            f"protocol.calib_use_last_s ({protocol.calib_use_last_s}) > "
            f"calibration_s ({protocol.calibration_s})"
        )

    pp = raw["preprocess"]
    eog = pp["eog_regression"]
    preprocess = PreprocessCfg(
        detrend=str(pp["detrend"]),
        notch_freqs_hz=tuple(float(x) for x in pp["notch_freqs_hz"]),
        notch_q=float(pp["notch_q"]),
        notch_passes=int(pp["notch_passes"]),
        bandpass_hz=_pair(pp["bandpass_hz"], "preprocess.bandpass_hz"),
        bandpass_order=int(pp["bandpass_order"]),
        filter_edge_s=float(pp["filter_edge_s"]),
        notch_redundant_atten_db=float(pp["notch_redundant_atten_db"]),
        eog_regression=EogRegressionCfg(
            enabled=bool(eog["enabled"]),
            regressor_band_hz=_pair(
                eog["regressor_band_hz"], "preprocess.eog_regression.regressor_band_hz"
            ),
            use_mean_diff_regressors=bool(eog["use_mean_diff_regressors"]),
            fit_on_blink_segments_only=bool(eog["fit_on_blink_segments_only"]),
            blink_detect_uv=float(eog["blink_detect_uv"]),
            blink_pad_s=float(eog["blink_pad_s"]),
            max_band_variance_removed=float(eog["max_band_variance_removed"]),
            run_control_pass=bool(eog["run_control_pass"]),
        ),
    )
    bp_lo, bp_hi = preprocess.bandpass_hz
    nyq = device.sfreq_nominal / 2.0
    if bp_hi >= nyq:
        raise ConfigError(
            f"preprocess.bandpass_hz superior ({bp_hi}) >= Nyquist ({nyq})"
        )
    # Um notch acima do corte superior não é automaticamente inútil: a 50 Hz um
    # Butterworth de ordem 4 com corte a 45 Hz atenua só ~5 dB (~11 dB com
    # filtfilt), e a rede pode ser duas ordens de grandeza acima do EEG. Só é
    # redundante quando o passa-banda já o mata — o que se mede, não se adivinha.
    _reject_redundant_notches(preprocess, device.sfreq_nominal)
    total_lo, total_hi = bands.total
    if total_lo < bp_lo or total_hi > bp_hi:
        raise ConfigError(
            f"banda total ({total_lo}-{total_hi}) excede o passa-banda "
            f"({bp_lo}-{bp_hi}); a potência relativa ficaria mal normalizada"
        )

    e = raw["epochs"]
    epochs = EpochsCfg(
        power=_grid(e["power"], "power"),
        connectivity=_grid(e["connectivity"], "connectivity"),
    )
    n_seg = epochs.connectivity.n_welch_segments()
    if n_seg < epochs.connectivity.min_segments:
        raise ConfigError(
            f"a janela de conectividade ({epochs.connectivity.win_s} s com "
            f"sub-segmentos de {epochs.connectivity.welch_seg_s} s) dá {n_seg} "
            f"segmentos, abaixo do mínimo de {epochs.connectivity.min_segments} "
            f"exigido por BIOMARKERS.md B10"
        )

    q = raw["quality"]
    mo = q["motion"]
    quality = QualityCfg(
        max_abs_uv=float(q["max_abs_uv"]),
        max_step_uv=float(q["max_step_uv"]),
        min_std_uv=float(q["min_std_uv"]),
        max_emg_rel=float(q["max_emg_rel"]),
        max_line_rel=float(q["max_line_rel"]),
        residual_blink_uv=float(q["residual_blink_uv"]),
        indices_from_unfiltered=bool(q["indices_from_unfiltered"]),
        min_calibration_coverage=float(q["min_calibration_coverage"]),
        motion=MotionCfg(
            accel_dev_g=float(mo["accel_dev_g"]),
            gyro_dps=float(mo["gyro_dps"]),
            invalidates_all_channels=bool(mo["invalidates_all_channels"]),
        ),
    )
    line_lo, line_hi = bands["line"]
    if quality.indices_from_unfiltered is False and line_hi > bp_hi:
        raise ConfigError(
            f"quality.indices_from_unfiltered está desligado mas a banda de rede "
            f"({line_lo}-{line_hi}) cai fora do passa-banda ({bp_lo}-{bp_hi}); o "
            f"line_index seria sempre ~0 e nunca dispararia"
        )

    c = raw["contact_monitor"]
    contact_monitor = ContactMonitorCfg(
        update_period_s=float(c["update_period_s"]),
        window_s=float(c["window_s"]),
        detrend_window=bool(c["detrend_window"]),
        trust_device_validation_flag=bool(c["trust_device_validation_flag"]),
        grade_on_filtered=bool(c["grade_on_filtered"]),
        min_std_uv=float(c["min_std_uv"]),
        max_drift_uv_per_s=float(c["max_drift_uv_per_s"]),
        line_warn_ratio=float(c["line_warn_ratio"]),
        slope_fit_range=_pair(c["slope_fit_range"], "contact_monitor.slope_fit_range"),
        slope_welch_seg_s=float(c["slope_welch_seg_s"]),
        plausible_slope_range=_pair(
            c["plausible_slope_range"], "contact_monitor.plausible_slope_range"
        ),
        green=ContactLevelCfg(**{k: float(x) for k, x in c["green"].items()}),
        yellow=ContactLevelCfg(**{k: float(x) for k, x in c["yellow"].items()}),
    )
    if not contact_monitor.detrend_window:
        raise ConfigError(
            "contact_monitor.detrend_window está desligado; o Unicorn entrega "
            "offsets de eletrodo de centenas de mV e sem detrend o sd/p2p mede "
            "deriva DC, não contacto — todos os canais ficariam vermelhos"
        )

    nz = raw["normalize"]
    normalize = NormalizeCfg(
        statistic=str(nz["statistic"]),
        mad_consistency_constant=float(nz["mad_consistency_constant"]),
        mad_fallback_iqr_constant=float(nz["mad_fallback_iqr_constant"]),
        min_epochs_ok=int(nz["min_epochs_ok"]),
        min_epochs_thin=int(nz["min_epochs_thin"]),
        min_windows_ok=int(nz["min_windows_ok"]),
        min_windows_thin=int(nz["min_windows_thin"]),
    )
    if normalize.min_epochs_thin > normalize.min_epochs_ok:
        raise ConfigError("normalize.min_epochs_thin > min_epochs_ok")
    if normalize.min_windows_thin > normalize.min_windows_ok:
        raise ConfigError("normalize.min_windows_thin > min_windows_ok")

    s = raw["stats"]
    bb, nc, su = s["block_bootstrap"], s["null_check"], s["connectivity_surrogates"]
    stats = StatsCfg(
        inference_on=str(s["inference_on"]),
        bootstrap_n=int(s["bootstrap_n"]),
        ci_level=float(s["ci_level"]),
        resample_calibration=bool(s["resample_calibration"]),
        berger_min_log2=float(s["berger_min_log2"]),
        berger_min_reactivity=float(s["berger_min_reactivity"]),
        berger_min_channel_agreement=float(s["berger_min_channel_agreement"]),
        berger_flanks=tuple(
            (float(a), float(b)) for a, b in s["berger_flanks"]
        ),
        theil_sen=bool(s["theil_sen"]),
        stability_odd_even=bool(s["stability_odd_even"]),
        stability_halves=bool(s["stability_halves"]),
        block_bootstrap=BlockBootstrapCfg(
            method=str(bb["method"]),
            length_selection=str(bb["length_selection"]),
            fixed_length_s=float(bb["fixed_length_s"]),
            min_length_s=float(bb["min_length_s"]),
            max_length_s=float(bb["max_length_s"]),
            acf_threshold=float(bb["acf_threshold"]),
            min_effective_samples=float(bb["min_effective_samples"]),
        ),
        null_check=NullCheckCfg(
            enabled=bool(nc["enabled"]),
            max_false_positive_rate=float(nc["max_false_positive_rate"]),
        ),
        connectivity_surrogates=SurrogatesCfg(
            enabled=bool(su["enabled"]),
            n_surrogates=int(su["n_surrogates"]),
            method=str(su["method"]),
        ),
    )
    if not 0.0 < stats.ci_level < 1.0:
        raise ConfigError(f"stats.ci_level ({stats.ci_level}) tem de estar em (0, 1)")
    if stats.block_bootstrap.min_length_s > stats.block_bootstrap.max_length_s:
        raise ConfigError("stats.block_bootstrap.min_length_s > max_length_s")

    co = raw["composite"]
    composite = CompositeCfg(
        weights_eyes_open={k: float(x) for k, x in co["weights_eyes_open"].items()},
        weights_eyes_closed={k: float(x) for k, x in co["weights_eyes_closed"].items()},
        renormalize_on_nan=bool(co["renormalize_on_nan"]),
    )
    for label, weights in (
        ("weights_eyes_open", composite.weights_eyes_open),
        ("weights_eyes_closed", composite.weights_eyes_closed),
    ):
        if sum(abs(w) for w in weights.values()) <= 0.0:
            raise ConfigError(f"composite.{label}: soma dos pesos absolutos é zero")

    mk = raw["markers"]
    markers = MarkersCfg(
        ap_exponent=dict(mk["ap_exponent"]),
        lziv=dict(mk["lziv"]),
        spectral_entropy=dict(mk["spectral_entropy"]),
    )

    dg = raw["diagnostics"]
    diagnostics = DiagnosticsCfg(
        r_emg_red_threshold=float(dg["r_emg_red_threshold"]),
        r_motion_red_threshold=float(dg["r_motion_red_threshold"]),
        r_luminance_red_threshold=float(dg["r_luminance_red_threshold"]),
        correlation_method=str(dg["correlation_method"]),
    )

    ag = raw["aggregate"]
    aggregate = AggregateCfg(
        min_sessions=int(ag["min_sessions"]),
        consistency_binomial_ci=bool(ag["consistency_binomial_ci"]),
    )

    rp = raw["report"]
    profile = str(rp["profile"])
    if profile not in ("researcher", "participant"):
        raise ConfigError(f"report.profile {profile!r} inválido")
    featured = tuple(rp["featured_markers"])
    if profile == "participant" and not featured:
        raise ConfigError(
            "report.profile é 'participant' mas featured_markers está vazio; "
            "os três marcadores só se escolhem depois da agregação (SPEC 5.3)"
        )
    report = ReportCfg(
        profile=profile,
        featured_markers=featured,
        featured_mode=str(rp["featured_mode"]),
        featured_count=int(rp["featured_count"]),
        max_metrics=int(rp.get("max_metrics", 6)),
        language=str(rp["language"]),
        dpi=int(rp["dpi"]),
        figsize_in=_pair(rp["figsize_in"], "report.figsize_in"),
        participant_footer=str(rp["participant_footer"]).strip(),
    )

    u = raw["ui"]
    ui = UiCfg(
        inspector_trace_s=float(u["inspector_trace_s"]),
        inspector_trace_period_s=float(u["inspector_trace_period_s"]),
        single_screen=bool(u["single_screen"]),
        title=str(u["title"]),
        subtitle=str(u["subtitle"]),
        shortcuts={str(k): str(v) for k, v in u["shortcuts"].items()},
        marker_keys={str(k): str(v) for k, v in u["marker_keys"].items()},
        validation_pair=(
            str(u["validation_pair"][0]),
            str(u["validation_pair"][1]),
        ),
        participant_screen=dict(u["participant_screen"]),
        operator_screen=dict(u["operator_screen"]),
        email=dict(u["email"]),
    )

    dr = raw["drive"]
    drive = DriveCfg(
        enabled=bool(dr["enabled"]),
        client_id=str(dr["client_id"]),
        client_secret=_drive_secret(dr),
        folder_id=str(dr["folder_id"]),
        scope=str(dr["scope"]),
        token_file=resolve_path(dr["token_file"]),
        delete_after_upload=bool(dr["delete_after_upload"]),
        max_attempts=int(dr["max_attempts"]),
        base_backoff_s=float(dr["base_backoff_s"]),
    )
    if drive.enabled and not drive.folder_id:
        raise ConfigError(
            "drive.enabled está ligado mas drive.folder_id está vazio. O id é a "
            "parte final do link de partilha: .../folders/<ID>."
        )

    pub = (raw.get("publish") or {}).get("github") or {}
    token = os.environ.get(str(pub.get("token_env", "")).strip(), "").strip()
    if not token and str(pub.get("token_file", "")).strip():
        token_path = resolve_path(str(pub["token_file"]))
        if token_path.exists():
            token = token_path.read_text(encoding="utf-8").strip()
    publish = PublishCfg(
        owner=str(pub.get("owner", "")).strip(),
        repo=str(pub.get("repo", "")).strip(),
        branch=str(pub.get("branch", "main")).strip() or "main",
        folder=str(pub.get("folder", "")).strip(),
        token=token,
        base_url=str(pub.get("base_url", "")).strip(),
    )

    pa = raw["paths"]
    entries = list(pa.get("mantras") or [])
    if not entries:
        raise ConfigError(
            "paths.mantras está vazio: sem mantra não há fase de mantra. "
            "Cada entrada precisa de id, label e file."
        )
    mantras = tuple(
        MantraCfg(
            id=str(m["id"]),
            label=str(m["label"]),
            file=resolve_path(m["file"]),
        )
        for m in entries
    )
    ids = [m.id for m in mantras]
    if len(set(ids)) != len(ids):
        raise ConfigError(f"paths.mantras tem ids repetidos: {ids}")
    defaults = [str(m["id"]) for m in entries if m.get("default")]
    if len(defaults) != 1:
        raise ConfigError(
            f"paths.mantras precisa de exatamente um `default: true`, "
            f"encontrei {len(defaults)}: {defaults}"
        )
    default_mantra = defaults[0]
    # Um ficheiro em falta so se descobriria quando a fase do mantra comeca,
    # com o participante ja de olhos fechados. Mas so e fatal se faltarem
    # TODOS: com um dos dois no sitio a banca continua a funcionar, e o
    # --selftest de scripts/run_booth.py diz qual e que falta.
    if not any(entry.file.exists() for entry in mantras):
        missing = ", ".join(str(m.file) for m in mantras)
        raise ConfigError(
            f"nenhum ficheiro de mantra existe. Verifique a pasta assets/: {missing}"
        )

    paths = PathsCfg(
        drive_folder_name=str(pa["drive_folder_name"]),
        sessions_dir=resolve_path(pa["sessions_dir"]),
        aggregate_dir=resolve_path(pa["aggregate_dir"]),
        assets_dir=resolve_path(pa["assets_dir"]),
        mantras=mantras,
        default_mantra=default_mantra,
        mantra_audio=(
            resolve_path(pa["mantra_audio"])
            if str(pa["mantra_audio"]).strip()
            else None
        ),
        mantra_start_offset_s=float(pa["mantra_start_offset_s"]),
        logo=resolve_path(pa["logo"]),
        logo_lockup=resolve_path(pa.get("logo_lockup", pa["logo"])),
    )

    return Config(
        version=int(raw["version"]),
        device=device,
        montage=montage,
        bands=bands,
        protocol=protocol,
        preprocess=preprocess,
        epochs=epochs,
        quality=quality,
        contact_monitor=contact_monitor,
        normalize=normalize,
        stats=stats,
        composite=composite,
        markers=markers,
        diagnostics=diagnostics,
        aggregate=aggregate,
        report=report,
        ui=ui,
        drive=drive,
        paths=paths,
        publish=publish,
        source_path=path,
    )
