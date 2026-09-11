"""Smoke test de hardware. O primeiro entregável do projeto (SPEC 6.1).

Sem UI e sem análise. Valida montagem, contacto, taxa real e caminho de
aquisição antes de existir mais alguma coisa.

    python scripts/check_unicorn.py
    python scripts/check_unicorn.py --validate-montage
    python scripts/check_unicorn.py --source replay --file gravacao.csv

Sem ``--source``, tenta a cascata da SPEC 1.3 — BrainFlow, depois ctypes sobre
``Unicorn.dll``, depois LSL — com diagnóstico explícito em cada falha.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mantraeeg.acquisition import Acquisition  # noqa: E402
from mantraeeg.config import Config, load_config  # noqa: E402
from mantraeeg.montage import (  # noqa: E402
    auto_checks,
    identify_tapped_channel,
)
from mantraeeg.sources import make_source  # noqa: E402
from mantraeeg.sources.base import (  # noqa: E402
    ACCEL_SLICE,
    COUNTER_ROW,
    EEG_SLICE,
    GYRO_SLICE,
    BATTERY_ROW,
    VALIDATION_ROW,
    EegSource,
    SourceError,
)

CASCADE = ("brainflow", "unicorn_dll", "lsl")
OK, WARN, BAD = "[ ok ]", "[aviso]", "[FALHA]"


def _setup_console() -> None:
    """Windows: consola em UTF-8, senão os µ e os acentos rebentam."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def head(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# --------------------------------------------------------------------------- #
# 1. Abrir a fonte
# --------------------------------------------------------------------------- #
def open_source(cfg: Config, requested: str | None, file: str | None):
    """Abre a fonte pedida, ou percorre a cascata da SPEC 1.3."""
    head("1. Caminho de aquisição")
    order = [requested] if requested else list(CASCADE)
    failures: list[str] = []

    for name in order:
        print(f"\n-> a tentar {name} ...")
        try:
            source = make_source(cfg, override=name, path=file)
            info = source.open()
        except SourceError as exc:
            print(f"   {BAD} {exc}")
            failures.append(f"{name}: {exc}")
            continue
        except Exception as exc:
            print(f"   {BAD} erro inesperado: {exc!r}")
            failures.append(f"{name}: {exc!r}")
            continue

        print(f"   {OK} aberto via {info.source_name}")
        print(f"        dispositivo : {info.device_id}")
        print(f"        sfreq       : {info.sfreq_nominal} Hz (lida do dispositivo)")
        print(f"        canais      : {info.n_channels} (layout canónico SPEC 1.1)")
        if info.backend_channel_names:
            print(
                f"        nomes do backend: {list(info.backend_channel_names)}\n"
                f"        {WARN} estes são a montagem DEFAULT do Unicorn e NÃO são "
                f"usados;\n"
                f"               a montagem vem da config (SPEC 1.2)."
            )
        for key, value in info.extra.items():
            if key not in ("brainflow_rows",):
                print(f"        {key}: {value}")
        return source, info

    head("Nenhum caminho de aquisição funcionou")
    for f in failures:
        print(f"  {BAD} {f}")
    print(
        "\nVerificar:\n"
        "  - o headset está ligado (LED) e emparelhado por Bluetooth?\n"
        "  - o dongle da g.tec está espetado?\n"
        "  - alguma outra aplicação tem o dispositivo aberto?\n"
        "  - para o caminho ctypes, o Unicorn Suite está instalado?"
    )
    raise SystemExit(2)


# --------------------------------------------------------------------------- #
# 2. Adquirir
# --------------------------------------------------------------------------- #
def acquire(cfg: Config, source: EegSource, seconds: float) -> tuple[np.ndarray, object]:
    head(f"2. Aquisição de {seconds:.0f} s")
    acq = Acquisition(cfg, source)
    print(
        f"     a descartar {cfg.device.settle_discard_s:.0f} s de assentamento do "
        f"amplificador antes de contar"
    )
    acq.start()
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < seconds + cfg.device.settle_discard_s:
            time.sleep(0.5)
            n = acq.stats.n_samples
            # Replay não-realtime esgota-se de imediato: não faz sentido esperar
            # pelo relógio de parede quando já não há mais dados a chegar.
            if getattr(source, "exhausted", False):
                print()
                print(f"     gravação esgotada: {n} amostras úteis")
                break
            print(
                f"\r     {n / cfg.device.sfreq_nominal:6.1f} s úteis  "
                f"({n} amostras)",
                end="",
                flush=True,
            )
    finally:
        print()
        stats = acq.stop()

    data = acq.buffer.peek_last(acq.buffer.capacity)
    if data.shape[1] == 0:
        print(f"  {BAD} não chegou uma única amostra.")
        raise SystemExit(3)
    return data, stats


# --------------------------------------------------------------------------- #
# 3. Estatísticas por canal
# --------------------------------------------------------------------------- #
def report_channels(cfg: Config, data: np.ndarray) -> None:
    head("3. Estatísticas por canal")
    monitor = Acquisition(cfg, source=_NullSource()).monitor
    report = monitor.evaluate(data)

    print(
        "  As medidas são feitas sobre a janela DETRENDED. O Unicorn entrega µV\n"
        "  com offset de eletrodo de centenas de mV; o sd do sinal bruto seria\n"
        "  deriva DC, não contacto.\n"
    )
    print(
        f"  {'idx':>3}  {'posição':<6} {'sd (µV)':>10} {'p2p (µV)':>11} "
        f"{'48-52/1-45':>11}  estado"
    )
    print(f"  {'-' * 68}")
    for c in report.channels:
        mark = {"green": "verde", "yellow": "AMARELO", "red": "VERMELHO"}[c.level]
        note = "" if c.reason == "ok" else f"  ({c.reason})"
        print(
            f"  {c.index:>3}  {c.name:<6} {c.std_uv:>10.2f} {c.p2p_uv:>11.2f} "
            f"{c.line_rel:>11.3f}  {mark}{note}"
        )

    n_good = report.n_good
    needed = cfg.montage.min_good_channels_to_start
    print(f"\n  canais não-vermelhos: {n_good}")
    if report.global_warning:
        print(f"\n  {BAD} {report.global_warning}")
    elif n_good < needed:
        print(f"  {WARN} abaixo dos {needed} canais úteis exigidos pela SPEC 2.")
    else:
        print(f"  {OK} contacto suficiente para arrancar.")


class _NullSource(EegSource):
    """Fonte inerte, só para construir um ContactMonitor sem abrir hardware."""

    def open(self):  # pragma: no cover
        raise SourceError("fonte inerte")

    def start(self) -> None:
        pass

    def _read_raw(self) -> np.ndarray:
        return np.zeros((17, 0), dtype=np.float32)

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# 4. Acelerómetro, giroscópio, bateria
# --------------------------------------------------------------------------- #
def report_imu(data: np.ndarray) -> None:
    head("4. Acelerómetro, giroscópio e bateria")
    accel, gyro = data[ACCEL_SLICE], data[GYRO_SLICE]
    mag = np.linalg.norm(accel, axis=0)
    print(f"  aceleração  média por eixo (g) : {accel.mean(axis=1).round(3)}")
    print(
        f"  magnitude   média {mag.mean():.3f} g   "
        f"|desvio de 1 g| máximo {np.abs(mag - 1.0).max():.3f} g"
    )
    print(f"  giroscópio  |máximo| por eixo (°/s): {np.abs(gyro).max(axis=1).round(2)}")
    print(f"  bateria     {data[BATTERY_ROW].mean():.0f} %")
    validation = np.unique(data[VALIDATION_ROW])
    print(f"  indicador de validação do dispositivo: {validation}")
    print(
        f"  {WARN} este indicador leu 1 com os elétrodos completamente soltos.\n"
        f"         Não é usado como verificação de contacto."
    )


# --------------------------------------------------------------------------- #
# 5. Continuidade, taxa real e jitter
# --------------------------------------------------------------------------- #
def report_timing(cfg: Config, data: np.ndarray, stats: object, source: EegSource) -> None:
    head("5. Continuidade, taxa real e jitter")
    counter = np.asarray(data[COUNTER_ROW], dtype=np.float64)
    steps = np.diff(counter)
    gaps = int(np.sum((np.abs(steps - 1.0) > 1) & (steps > 0)))
    wraps = int(np.sum(steps < 0))
    print(f"  amostras analisadas      : {data.shape[1]}")
    print(f"  falhas do contador       : {gaps}   (wraps: {wraps})")
    if gaps:
        print(
            f"  {BAD} houve perda de amostras. A base temporal da análise tem de\n"
            f"         ser reconstruída a partir do contador."
        )
    else:
        print(f"  {OK} contador contínuo, sem amostras perdidas.")

    measured = getattr(stats, "measured_sfreq", float("nan"))
    nominal = cfg.device.sfreq_nominal
    print(f"  taxa nominal             : {nominal:.3f} Hz")
    is_fast_replay = source.info.source_name == "replay" and not source.info.extra.get(
        "realtime", False
    )
    if is_fast_replay:
        print(
            "  taxa real medida         : n/a (replay acelerado entrega tudo de "
            "imediato)"
        )
    else:
        print(f"  taxa real medida         : {measured:.3f} Hz")
    if np.isfinite(measured) and not is_fast_replay:
        drift = (measured - nominal) / nominal
        total = drift * (cfg.protocol.calibration_s + cfg.protocol.mantra_s)
        print(
            f"  desvio                   : {drift * 100:+.2f} %  "
            f"({total:+.2f} s ao longo de uma sessão)"
        )

    ts = getattr(source, "last_timestamps", np.zeros(0))
    if np.size(ts) > 2:
        dt = np.diff(np.asarray(ts, dtype=np.float64))
        print(
            f"  jitter inter-amostra     : mediana {np.median(dt) * 1e3:.2f} ms, "
            f"p99 {np.percentile(dt, 99) * 1e3:.1f} ms, máx {dt.max() * 1e3:.1f} ms"
        )
        print(
            f"  {WARN} as amostras chegam em rajadas de pacotes Bluetooth. Os\n"
            f"         timestamps do backend NÃO servem de base temporal — a base\n"
            f"         é o contador do dispositivo mais a taxa medida."
        )

    overruns = getattr(stats, "overruns", 0)
    discarded = getattr(stats, "discarded_settle_samples", 0)
    print(f"  overruns do ring buffer  : {overruns}")
    print(f"  descartado no assentamento: {discarded} amostras")
    for warning in getattr(stats, "plausibility_warnings", []):
        print(f"  {WARN} amplitude implausível: {warning}")


# --------------------------------------------------------------------------- #
# 6. Montagem
# --------------------------------------------------------------------------- #
def report_montage_auto(cfg: Config, data: np.ndarray) -> None:
    head("6. Validação automática de montagem")
    n = int(cfg.montage.validation.auto_check_window_s * cfg.device.sfreq_nominal)
    eeg = data[EEG_SLICE][:, -n:] if data.shape[1] > n else data[EEG_SLICE]
    for result in auto_checks(eeg, cfg):
        mark = {True: OK, False: BAD, None: WARN}[result.passed]
        print(f"  {mark} {result.name}: {result.detail}")


def run_tap_test(cfg: Config, source: EegSource) -> None:
    head("7. Teste de toque (validação definitiva da montagem)")
    print(
        "  Vou pedir-lhe que toque num elétrodo de cada vez. O teste diz qual o\n"
        "  ÍNDICE que reagiu — é assim que se apanha um F3/F4 trocado.\n"
    )
    acq = Acquisition(cfg, source)
    acq.start()
    seconds = cfg.montage.validation.tap_test_seconds_per_channel
    errors: list[str] = []
    try:
        time.sleep(cfg.device.settle_discard_s)
        for expected_idx in sorted(cfg.montage.channels):
            name = cfg.montage.channels[expected_idx]
            input(f"  Toque em {name} e prima Enter para começar a medir...")
            acq.buffer.read()  # esvazia o que ficou do prompt
            print(f"    a medir {seconds:.0f} s — toque agora, repetidamente ...")
            time.sleep(seconds)
            window = acq.buffer.peek_last(int(seconds * cfg.device.sfreq_nominal))
            found, ratio = identify_tapped_channel(window[EEG_SLICE], cfg)
            if found is None:
                print(f"    {WARN} nenhum canal se destacou (razão {ratio:.1f}x).")
                errors.append(f"{name}: sem resposta clara")
            elif found == expected_idx:
                print(f"    {OK} índice {found} reagiu — corresponde a {name} ({ratio:.1f}x).")
            else:
                got = cfg.montage.channels.get(found, "?")
                print(
                    f"    {BAD} tocou em {name} (índice esperado {expected_idx}) mas "
                    f"reagiu o índice {found}, que a config chama {got!r} ({ratio:.1f}x)."
                )
                errors.append(f"{name}: esperado {expected_idx}, reagiu {found}")
    finally:
        acq.stop()

    if errors:
        print(f"\n  {BAD} MONTAGEM INCORRETA. Corrigir montage.channels na config:")
        for e in errors:
            print(f"        - {e}")
    else:
        print(f"\n  {OK} montagem confirmada elétrodo a elétrodo.")


# --------------------------------------------------------------------------- #
def main() -> int:
    _setup_console()
    ap = argparse.ArgumentParser(description="Smoke test do Unicorn (SPEC 6.1)")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--source", choices=[*CASCADE, "replay"], default=None)
    ap.add_argument("--file", default=None, help="gravação, para --source replay")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--validate-montage", action="store_true", help="teste de toque")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"config: {args.config}   (versão {cfg.version})")
    print(f"montagem: {dict(sorted(cfg.montage.channels.items()))}")

    source, info = open_source(cfg, args.source, args.file)
    try:
        data, stats = acquire(cfg, source, args.seconds)
        report_channels(cfg, data)
        report_imu(data)
        report_timing(cfg, data, stats, source)
        report_montage_auto(cfg, data)
        if args.validate_montage:
            run_tap_test(cfg, source)
    finally:
        head("Desligar")
        source.stop()
        source.close()
        print(f"  {OK} dispositivo libertado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
