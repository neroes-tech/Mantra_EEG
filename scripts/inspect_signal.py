"""Inspetor de sinal ao vivo, autónomo.

    python scripts/inspect_signal.py
    python scripts/inspect_signal.py --source replay --file tests/fixtures/unicorn_sample.csv

Abre uma janela com os traçados dos oito canais e o semáforo por elétrodo. É a
ferramenta para colocar o equipamento na cabeça e ajustar até o sinal estar bom,
com feedback imediato em vez de um script em lote de cada vez.

É o mesmo painel que o M4 abre com CTRL+I durante qualquer fase da sessão.
Fechar a janela desliga o dispositivo de forma limpa.
"""

from __future__ import annotations

import argparse
import signal as os_signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PyQt6 import QtCore, QtWidgets  # noqa: E402

from mantraeeg.acquisition import Acquisition  # noqa: E402
from mantraeeg.montage import auto_checks, summarize  # noqa: E402
from mantraeeg.recorder import Recorder  # noqa: E402
from mantraeeg.config import load_config  # noqa: E402
from mantraeeg.sources import make_source  # noqa: E402
from mantraeeg.sources.base import SourceError  # noqa: E402
from mantraeeg.ui.inspector import SignalInspector  # noqa: E402

CASCADE = ("brainflow", "unicorn_dll", "lsl")


def open_source(cfg, requested: str | None, file: str | None):
    """Abre a fonte pedida, ou percorre a cascata da SPEC 1.3."""
    failures: list[str] = []
    for name in [requested] if requested else list(CASCADE):
        try:
            source = make_source(cfg, override=name, path=file)
            info = source.open()
        except Exception as exc:
            failures.append(f"  {name}: {exc}")
            continue
        print(
            f"ligado via {info.source_name} ({info.device_id}), "
            f"{info.sfreq_nominal} Hz"
        )
        return source
    raise SourceError("nenhuma fonte abriu:\n" + "\n".join(failures))


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspetor de sinal ao vivo")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--source", choices=[*CASCADE, "replay"], default=None)
    ap.add_argument("--file", default=None, help="gravação, para --source replay")
    ap.add_argument(
        "--record",
        action="store_true",
        help="gravar tudo o que passar pela janela para data/sessions/",
    )
    ap.add_argument("--note", default="", help="nota livre gravada no events.json")
    ap.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="fechar sozinho ao fim de N s (garante que a gravação é fechada)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.source == "replay":
        # Em replay o inspetor só é útil em tempo real; a alternativa entrega
        # tudo de imediato e a janela ficaria parada no fim da gravação.
        cfg.device.replay["realtime"] = True

    try:
        source = open_source(cfg, args.source, args.file)
    except SourceError as exc:
        print(exc, file=sys.stderr)
        print(
            "\nVerificar:\n"
            "  - o headset está ligado e emparelhado por Bluetooth?\n"
            "  - o dongle da g.tec está espetado?\n"
            "  - outra aplicação tem o dispositivo aberto?",
            file=sys.stderr,
        )
        return 2

    app = QtWidgets.QApplication(sys.argv)
    os_signal.signal(os_signal.SIGINT, os_signal.SIG_DFL)  # Ctrl+C fecha a janela

    recorder = Recorder(cfg) if args.record else None
    if recorder is not None:
        print(f"a gravar para {recorder.dir}")
        if args.note:
            recorder.add_event("operator_note", t_s=0.0, detail=args.note)

    acq = Acquisition(cfg, source, on_chunk=recorder.write if recorder else None)
    acq.start()

    window = SignalInspector(cfg, acq, recorder=recorder)
    window.resize(1500, 900)
    window.show()

    # Deixa o Python correr entre eventos Qt, para o Ctrl+C funcionar.
    keepalive = QtCore.QTimer()
    keepalive.start(200)
    keepalive.timeout.connect(lambda: None)

    def shutdown() -> None:
        window.stop()
        stats = acq.stop()
        if recorder is not None:
            # As verificações automáticas de montagem viajam com os dados.
            eeg = acq.buffer.peek_last(
                int(
                    cfg.montage.validation.auto_check_window_s
                    * cfg.device.sfreq_nominal
                )
            )
            validation = (
                summarize(auto_checks(eeg[0:8], cfg)) if eeg.shape[1] > 2 else {}
            )
            out = recorder.finalize(
                source.info,
                measured_sfreq=stats.measured_sfreq,
                acquisition_stats=stats,
                montage_validation=validation,
            )
            print(
                f"gravado: {out}  ({recorder.n_samples} amostras, "
                f"{recorder.n_samples / cfg.device.sfreq_nominal:.0f} s)"
            )
            if window.marks:
                print(f"marcas: {len(window.marks)} segmentos")
                for t, label in window.marks:
                    print(f"  {t:7.1f} s  {label}")
                print()
                print("validar o pipeline contra estas marcas:")
                print(f"  python scripts/compare_segments.py --session {out}")
            else:
                print(
                    f"{'!' * 3} sem marcas: use as teclas durante a gravacao para "
                    f"marcar segmentos"
                )
        source.close()
        print("dispositivo libertado.")

    app.aboutToQuit.connect(shutdown)

    if args.seconds > 0:
        print(f"a fechar automaticamente ao fim de {args.seconds:.0f} s")
        QtCore.QTimer.singleShot(int(args.seconds * 1000), app.quit)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
