"""Constroi o executavel da banca e monta a pasta de distribuicao.

    python scripts/build_exe.py
    python scripts/build_exe.py --clean          # apaga build/ e dist/ antes
    python scripts/build_exe.py --no-mantras     # sem os videos (dist leve)

Depois de correr, `dist/MantraEEG/` e copiavel tal e qual para os portateis do
evento. A unica coisa que a maquina de destino precisa e do Unicorn emparelhado
por Bluetooth.

macOS: o PyInstaller nao faz compilacao cruzada, e mais importante do que isso,
a g.tec nao distribui driver do Unicorn para macOS — o BrainFlow so suporta
esta placa em Windows e Linux. Um executavel para Mac nao encontraria o
dispositivo, por isso nao ha aqui um alvo para Mac. O codigo em si nao tem nada
de especifico do Windows: se algum dia houver driver, correr o PyInstaller num
Mac com este mesmo spec chega.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: O que e copiado para junto do executavel depois da construcao. Fica fora do
#: pacote de proposito: e o que o operador pode ver, trocar e editar na banca.
ALONGSIDE = [
    ("config", "*.yaml"),
    ("assets", "*"),
]

#: Sem isto o primeiro mantra sozinho leva 1,1 GB para a pasta de distribuicao.
MEDIA_SUFFIXES = {".mp4", ".m4v", ".mov", ".mkv"}


def setup_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def copy_alongside(dist: Path, include_media: bool) -> None:
    total = 0
    for folder, pattern in ALONGSIDE:
        source = ROOT / folder
        if not source.exists():
            continue
        target = dist / folder
        target.mkdir(parents=True, exist_ok=True)
        for item in sorted(source.glob(pattern)):
            if not item.is_file():
                continue
            if not include_media and item.suffix.lower() in MEDIA_SUFFIXES:
                print(f"  (saltado) {folder}/{item.name}")
                continue
            shutil.copy2(item, target / item.name)
            total += item.stat().st_size
            print(f"  {folder}/{item.name}  {item.stat().st_size / 1e6:.1f} MB")
    print(f"  --- {total / 1e6:.0f} MB ao lado do executavel")


#: Credenciais que a banca precisa para gravar no Drive, enviar o email e
#: publicar o relatorio. Vao no pacote para ele sair pronto a correr, por
#: decisao expressa. NUNCA vai `data/sessions` nem `data/reports`: esses sao
#: dados de participantes, e nao sao nossos para distribuir.
CREDENTIALS = (
    "drive_token.json",
    "github_token.txt",
    "google_client_secret.txt",
)


def copy_credentials(dist: Path, include: bool) -> None:
    target = dist / "data"
    target.mkdir(exist_ok=True)
    if not include:
        print("  (sem credenciais: quem receber tem de as por a mao)")
        return
    found = False
    for name in CREDENTIALS:
        source = ROOT / "data" / name
        if source.exists():
            shutil.copy2(source, target / name)
            print(f"  data/{name}")
            found = True
        else:
            print(f"  (em falta) data/{name}")
    if found:
        print(
            "  --- o pacote leva credenciais: quem o tiver envia email pela\n"
            "      conta autenticada e escreve no repositorio dos relatorios."
        )


def make_zip(dist: Path) -> Path:
    """O zip pronto a partilhar, com o que tem de ir e nada mais."""
    out = ROOT / "dist" / "MantraEEG.zip"
    out.unlink(missing_ok=True)
    import zipfile

    print("\na criar o zip…")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for item in sorted(dist.rglob("*")):
            if not item.is_file():
                continue
            # Os videos ja estao comprimidos: deflate neles gasta minutos por
            # nada.
            stored = item.suffix.lower() in MEDIA_SUFFIXES
            archive.write(
                item,
                item.relative_to(dist.parent).as_posix(),
                compress_type=zipfile.ZIP_STORED if stored else None,
            )
    print(f"  {out}  ({out.stat().st_size / 1e6:.0f} MB)")
    return out


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description="Construcao do executavel")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument(
        "--no-mantras",
        action="store_true",
        help="nao copiar os videos; copia-os a mao depois",
    )
    ap.add_argument(
        "--sem-credenciais",
        action="store_true",
        help="nao copiar os tokens; para distribuir fora da equipa",
    )
    ap.add_argument("--zip", action="store_true", help="criar dist/MantraEEG.zip")
    args = ap.parse_args()

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("falta o PyInstaller:  pip install pyinstaller")
        return 1

    if args.clean:
        for folder in ("build", "dist"):
            shutil.rmtree(ROOT / folder, ignore_errors=True)
            print(f"apagado {folder}/")

    print("a construir…")
    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", "mantraeeg.spec", "--noconfirm"],
        cwd=ROOT,
    )
    if result.returncode != 0:
        return result.returncode

    dist = ROOT / "dist" / "MantraEEG"
    if not dist.exists():
        print(f"o PyInstaller nao produziu {dist}")
        return 1

    print("\na copiar o conteudo para junto do executavel:")
    copy_alongside(dist, include_media=not args.no_mantras)
    copy_credentials(dist, include=not args.sem_credenciais)

    if args.zip:
        make_zip(dist)

    size = sum(f.stat().st_size for f in dist.rglob("*") if f.is_file())
    print(f"\npronto: {dist}   ({size / 1e6:.0f} MB)")
    print("\nNa maquina de destino:")
    print("  1. emparelhar o Unicorn por Bluetooth (dongle da g.tec, PIN 0000)")
    print("  2. copiar a pasta MantraEEG inteira")
    print("  3. duplo clique em MantraEEG.exe")
    if args.no_mantras:
        print("  4. copiar os .mp4 para MantraEEG/assets/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
