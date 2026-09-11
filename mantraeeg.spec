# -*- mode: python ; coding: utf-8 -*-
"""Empacotamento da banca: uma pasta, um executável, conteúdo ao lado.

    python -m PyInstaller mantraeeg.spec --noconfirm

Produz ``dist/MantraEEG/`` com esta forma:

    MantraEEG/
      MantraEEG.exe        <- duplo clique
      _internal/           <- Python, Qt, BrainFlow, o código
      config/default.yaml  <- editável na banca, sem reconstruir
      assets/              <- mantras, logótipos, deixas sonoras
      data/                <- criado ao correr; gravações e token

**Os mantras não vão para dentro do pacote.** São 1,1 GB só o primeiro: metê-lo
no executável dá um ficheiro impossível de copiar para os portáteis do evento,
e trocar de mantra passaria a exigir uma reconstrução. Ficam ao lado, e a
aplicação resolve os caminhos contra a pasta do executável
(``mantraeeg.apppaths``).

**BrainFlow traz as bibliotecas nativas dentro do pacote** — é por isso que o
computador da banca não precisa de instalar nada além do emparelhamento
Bluetooth do Unicorn. O Unicorn Suite continua a ser preciso se se usar o
caminho da DLL da g.tec (``device.source: unicorn_dll``), que é o recurso
alternativo; com ``brainflow`` não é.
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs

ROOT = Path(SPECPATH)

# BrainFlow carrega .dll/.so por caminho em tempo de execução; sem isto o
# pacote arranca e só falha ao ligar ao dispositivo, que é o pior sítio.
binaries = collect_dynamic_libs("brainflow")

datas = [
    (str(ROOT / "config" / "default.yaml"), "config"),
]

hiddenimports = [
    # Qt Multimedia não é importado por nome em lado nenhum — é carregado
    # dentro de _build_video, num try/except. O PyInstaller não o vê.
    "PyQt6.QtMultimedia",
    "PyQt6.QtMultimediaWidgets",
    "PyQt6.QtSvg",
    # scipy.signal traz extensões que a análise espectral usa por nome.
    "scipy.signal",
    "scipy.special._cdflib",
]

a = Analysis(
    [str(ROOT / "scripts" / "run_booth.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # matplotlib só serve as figuras de investigação; o relatório do
    # participante é SVG escrito à mão. Tirá-lo poupa ~40 MB — mas
    # report/figures.py importa-o, por isso fica.
    excludes=["tkinter", "pytest", "IPython", "notebook"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MantraEEG",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # Sem consola: é um quiosque. Os erros vão para o ecrã e para os
    # ficheiros da sessão, não para uma janela preta atrás da aplicação.
    console=False,
    disable_windowed_traceback=False,
    icon=str(ROOT / "assets" / "logo.png"),
)

# O mesmo código, com consola, para o dia do evento: `--selftest` verifica a
# instalação e imprime o que falta. O executável principal não tem consola (é
# um quiosque), e sem isto não haveria onde ler a resposta.
diagnostico = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Diagnostico",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    diagnostico,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="MantraEEG",
)
