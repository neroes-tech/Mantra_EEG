# mantra-eeg

Estação de demonstração de EEG para festival de bem-estar, com **análise
post-hoc**. Hardware: g.tec Unicorn Hybrid Black a 250 Hz. Entrega: executável
Windows.

Especificação em [`SPEC.md`](SPEC.md). Definição dos biomarcadores em
[`BIOMARKERS.md`](BIOMARKERS.md), **normativo** sobre a SPEC no que respeita a
fórmulas e nomes de métricas.

## Estado

| Marco | Conteúdo | Estado |
|---|---|---|
| **M0** | Hardware e gravação | **concluído** |
| M1 | Biblioteca de análise | `preprocess`, `spectral`, `stats` feitos; falta conectividade, registo de marcadores, normalização |
| M2 | Passagem de análise e estatística | por fazer |
| M3 | Relatório e agregação | por fazer |
| **M4** | UI e executável | **ecrãs e fluxo feitos**; falta o build PyInstaller |

## Pré-requisitos

- **Python 3.11 x64** (a SPEC 12 fixa esta versão; o PyInstaller e o PyQt6 não
  estão rodados em versões mais recentes).
- **Unicorn Suite** instalado — fornece `Unicorn.dll` em
  `C:\Program Files\gtec\Unicorn Suite\Hybrid Black\`. O BrainFlow traz a sua
  própria cópia da DLL, portanto o Suite não é estritamente necessário para o
  caminho de aquisição primário, mas é o que dá o `UnicornSuite.exe` e a
  documentação da C API.
- **Headset emparelhado** por Bluetooth através do dongle da g.tec.

A `UnicornPy` (API Python licenciada) está **fora de âmbito**.

## Instalação

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Correr

```powershell
# A banca completa: consola do operador + ecra do participante
.\.venv\Scripts\python.exe .\scripts\run_booth.py

# Inspetor de sinal isolado, para ajustar os eletrodos
.\.venv\Scripts\python.exe scripts\inspect_signal.py

# Gravar com marcacao de segmentos (teclas 1-6 durante a gravacao)
.\.venv\Scripts\python.exe scripts\inspect_signal.py --record --seconds 320

# Validar o pipeline pelo efeito de Berger
.\.venv\Scripts\python.exe scripts\compare_segments.py --figure
```

Atalhos durante a sessao: **Ctrl+I** inspecionar o sinal, **V** mostrar/esconder
o video (o audio nunca para), **Esc** abortar.

## Verificar

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\check_unicorn.py
.\.venv\Scripts\python.exe scripts\check_unicorn.py --validate-montage
```

Sem hardware, o smoke test corre sobre uma gravação Unicorn real:

```powershell
.\.venv\Scripts\python.exe scripts\check_unicorn.py `
    --source replay --file tests\fixtures\unicorn_sample.csv
```

## Notas de hardware apuradas empiricamente

Estas foram medidas contra o dispositivo e contra gravações reais, e estão
codificadas no comportamento do programa. Ficam aqui porque nenhuma é óbvia a
partir da documentação.

**O caminho de aquisição é o BrainFlow.** Abre o `UNICORN_BOARD` e auto-deteta o
dispositivo emparelhado sem serial. Traz a sua própria `Unicorn.dll`, distinta
da do Suite. Os caminhos `unicorn_dll` (ctypes) e `lsl` existem como fallback;
o segundo requer o UnicornLSL, que o Suite não instalou aqui.

**O BrainFlow apresenta 19 linhas, não as 17 do dispositivo** — acrescenta
`package_num`, `timestamp` e `marker`. A camada de fontes reduz ao layout
canónico da SPEC 1.1.

**`get_eeg_names()` devolve a montagem *default* do Unicorn**
(`Fz, C3, Cz, C4, Pz, PO7, Oz, PO8`), que **não** é a nossa. Usá-la trocaria F3
com F4 sem qualquer aviso. A montagem vem exclusivamente da config, e o
`raw_meta.json` regista explicitamente que os nomes do backend não foram usados.

**Os timestamps do BrainFlow não servem de base temporal.** As amostras chegam
em rajadas de pacotes Bluetooth: mediana de 2,5 ms entre amostras, p99 de 27 ms,
máximo de 64 ms, com uma taxa global de ~250,5 Hz. A base temporal é o contador
do dispositivo mais a taxa medida.

**O offset DC de eletrodo é de centenas de mV**, diferente por canal (medido:
185 a 711 mV). O offset constante não afeta o desvio padrão, mas com contacto
mau o sinal deriva — mediu-se o DC a subir 5 660 µV/s sem estabilizar — e essa
deriva domina o desvio padrão e sobretudo o pico-a-pico. O monitor de contacto
faz detrend da sua janela antes de medir, e a config proíbe desligá-lo.

**O indicador de validação do dispositivo (linha 16) não deteta falta de
contacto.** Leu `1` em todas as amostras com os elétrodos completamente soltos.
Não é usado.

**A rede de 50 Hz pode ser esmagadora.** Com os elétrodos numa mesa mediu-se a
PSD a 50 Hz em 1,3x10⁸ vezes a PSD a 10 Hz. O notch é aplicado **duas vezes**:
uma passagem deixa a razão 48-52/1-45 em 16,6, duas levam-na a 0,030, e o custo
na banda útil sobe apenas de -0,23 para -0,45 dB no pior ponto.

**O piso de ruído parece EEG.** Com os elétrodos soltos, depois de filtrar, os
oito canais dão sd de 4,5 µV e declive log-log de -1,25 — plausível por todas as
métricas de amplitude e de forma espectral. O que apanha isto é a verificação de
independência entre canais e a taxa de deriva, mais nada.

**Referência ou terra soltos produzem oito canais com o mesmo sinal.**
Correlação de 1,000000 entre todos os pares e uma diferença entre canais de
0,07 % da amplitude. É detetado explicitamente, com um critério **relativo** à
amplitude — um limiar em µV absolutos falha precisamente nas amplitudes
maiores, que são as piores.

## Estrutura

```
config/default.yaml     toda a configuração; zero números mágicos no código
src/mantraeeg/
  config.py             dataclasses validadas + aritmética de grelha
  bands.py              estrutura das bandas (valores vêm do YAML)
  ring_buffer.py        escritor único / leitor único, overruns contados
  sources/              base, brainflow_src, unicorn_dll, lsl, replay
  acquisition.py        thread + monitor de contacto
  recorder.py           streaming para disco + raw_meta.json
  montage.py            validação de montagem (o risco de F3/F4 trocados)
  preprocess.py         cadeia da SPEC 3.1 (notch em cascata, EOG por regressao)
  spectral.py           Welch, potencia de banda, declive log-log
  stats.py              Cliff, bootstrap por blocos, n_efectivo
  session.py            maquina de estados do protocolo (sem Qt, testavel)
  audio.py              deixas sintetizadas + reproducao do mantra
  ui/
    app.py              orquestra a sessao
    participant.py      ecra do participante (preto, temporizador, video)
    operator.py         consola do operador
    inspector.py        painel do CTRL+I
scripts/
  run_booth.py          a banca (--selftest verifica a instalacao)
  build_exe.py          constroi dist/MantraEEG/
  build_report.py       report.json + relatorio.html de uma sessao
  send_report_email.py  publica e envia (--dry-run escreve um .eml)
  check_unicorn.py      smoke test de hardware (SPEC 6.1)
  inspect_signal.py     inspetor isolado, com --record
  compare_segments.py   validacao do pipeline pelo efeito de Berger
tests/                  185 testes
```

`data/` não vai para o git.

---

## O relatório do participante

```powershell
python scripts/build_report.py --silent              # sessao de controlo
python scripts/build_report.py --mantra "Sri Vitthala"
python scripts/build_report.py                       # le o mantra da gravacao
```

Escreve `report.json` (os dados) e `relatorio.html` (a página) na pasta da
sessão. O JSON é a fonte do email também, para os dois nunca divergirem.

O mantra tem de estar determinado: ou vem de `raw_meta.json`, ou de
`--mantra`, ou é `--silent`. **Não há default** — uma sessão de controlo não
pode sair do relatório com o nome de um mantra que não tocou.

A prosa da secção "Leitura da sessão" é gerada por regras sobre os números
(`report/payload.py::_reading`), não por um modelo de linguagem: corre dentro
do executável, offline, e dois participantes com o mesmo resultado recebem a
mesma frase.

## O email

Vai pela conta Google que iniciou sessão no painel (Ctrl+I) — a mesma do
Drive, sem SMTP e sem palavra-passe guardada na banca. O corpo está em
`ui.email.body` na configuração, com `{nome}` e `{link}`.

```powershell
python scripts/send_report_email.py --to eu@exemplo.pt --name Ana --dry-run
python scripts/send_report_email.py --to eu@exemplo.pt --name Ana --sign-in
```

`--dry-run` escreve um `.eml` na pasta da sessão em vez de enviar. Abre com
duplo clique e mostra exatamente o que a pessoa recebe.

**O âmbito `gmail.send` faz parte de `drive.scope`.** Mudá-lo invalidou o
token guardado: o primeiro arranque a seguir pede login outra vez, uma vez por
computador.

## O link do relatório

**O Google Drive não serve HTML.** Medido: um ficheiro `text/html` partilhado
publicamente devolve

```
Content-Type: application/octet-stream
Content-Disposition: attachment; filename="FBE-2026-110628.html"
```

ou seja, o link é livre mas **descarrega** em vez de abrir. Para o
participante clicar e ver o relatório é preciso alojamento estático.

### GitHub Pages — gratuito, permanente, cinco minutos

O que é preciso fazer, uma vez:

1. **Conta no GitHub** (github.com/signup) — gratuita, sem cartão.
2. **Criar um repositório público**, por exemplo `relatorios`. Marcar
   *Public* e *Add a README file* (o repositório não pode estar vazio).
3. **Ligar o Pages**: no repositório, *Settings* → *Pages* → *Source:
   Deploy from a branch* → *Branch: `main`*, pasta `/ (root)* → *Save*.
   Ao fim de um minuto aparece lá o endereço
   `https://<utilizador>.github.io/relatorios/`.
4. **Criar um token**: *Settings* da conta (não do repositório) →
   *Developer settings* → *Personal access tokens* → *Fine-grained tokens* →
   *Generate new token*.
   - *Repository access*: **Only select repositories** → `relatorios`
   - *Permissions* → *Repository permissions* → **Contents: Read and write**
   - *Expiration*: o que der jeito; se expirar, o envio falha com uma
     mensagem que o diz
   - Copiar o token (`github_pat_…`) — só aparece uma vez.
5. **Pôr o token na máquina**, num ficheiro com o token e mais nada:
   `MantraEEG/data/github_token.txt`. Não vai para o git nem é copiado pelo
   `build_exe.py`. Em alternativa, a variável de ambiente
   `MANTRA_GITHUB_TOKEN`.
6. **Editar `config/default.yaml`**:

```yaml
  email:
    link_mode: github_pages

publish:
  github:
    owner: "<o-teu-utilizador>"
    repo: "relatorios"
```

7. **Confirmar**: `Diagnostico.exe --selftest` passa a verificar o
   repositório, o token e se está público.

O link fica `https://<utilizador>.github.io/relatorios/r/FBE-2026-110628-3f9a2c81.html`
e abre em qualquer browser, sem sessão iniciada, e reencaminha-se.

**Domínio próprio** (opcional): no repositório, *Settings* → *Pages* →
*Custom domain* → `relatorios.neroes.tech`, e um registo CNAME no DNS a
apontar para `<utilizador>.github.io`. Depois é só pôr
`base_url: "https://relatorios.neroes.tech"` na config.

### Duas notas que não são detalhe

**O endereço é público.** Qualquer pessoa com o link abre o relatório, que
tem o nome do participante. Por isso o nome do ficheiro leva oito dígitos
hexadecimais aleatórios (`…-3f9a2c81.html`): o código deriva da hora da
sessão e sozinho seria adivinhável por quem esteve na fila. Não é privado —
é não indexado e não adivinhável. Se isso não chegar para o enquadramento de
RGPD, as opções são tirar o nome do relatório ou alojar atrás de
autenticação.

**O Pages não publica no instante.** Constrói o site primeiro, o que leva de
dez segundos a um minuto (mais na primeira vez). A aplicação **espera até o
endereço responder** antes de mandar o email, até 90 s, para o primeiro
clique não dar 404. É por isso que `ui.email.timeout_s` está em 180.

### Os outros dois modos

`base_url` — se puserem os ficheiros num sítio que já servem (uma pasta no
site da Neroes, um bucket). A aplicação devolve `<public_base_url>/<ficheiro>`
mas **não publica lá**; alguém tem de copiar o ficheiro.

`drive` — o que existia. O link descarrega, e por isso neste modo o relatório
vai **também em anexo** (`attach_report: auto`). Com o Pages ligado o anexo
deixa de ser enviado.

## O executável

```powershell
pip install pyinstaller
python scripts/build_exe.py --clean
```

Produz `dist/MantraEEG/`, com esta forma:

```
MantraEEG/
  MantraEEG.exe        <- duplo clique, sem consola
  Diagnostico.exe      <- o mesmo com consola: `Diagnostico.exe --selftest`
  _internal/           <- Python, Qt, BrainFlow (inclui a Unicorn.dll)
  config/default.yaml  <- editavel na banca, sem reconstruir
  assets/              <- mantras, logotipos, deixas sonoras
  data/                <- criado ao correr
```

**Os mantras ficam ao lado, não dentro.** São 1,1 GB só o primeiro: metê-los
no pacote dava um ficheiro impossível de copiar e trocar de mantra passaria a
exigir uma reconstrução. `--no-mantras` salta a cópia, para iterar depressa.

Na máquina do evento a única responsabilidade é o **emparelhamento Bluetooth
do Unicorn** (dongle da g.tec). O Unicorn Suite **não** é preciso: o BrainFlow
traz a sua própria `Unicorn.dll` dentro do pacote, e `resolved_dll_paths()`
usa-a como último candidato. Antes de abrir a banca:

```powershell
.\Diagnostico.exe --selftest
```

Verifica configuração, mantras, logótipo, Qt Multimedia, headset emparelhado,
BrainFlow e permissão de escrita — e diz o que falta, em vez de se descobrir
com uma pessoa já sentada na cadeira.

### macOS

Não há alvo para Mac, e a razão não é o empacotamento: **a g.tec não
distribui driver do Unicorn para macOS**. O BrainFlow suporta esta placa em
Windows e Linux; num Mac o executável abria e nunca encontrava o dispositivo.
O código não tem nada de específico do Windows — se algum dia houver driver,
correr o PyInstaller num Mac com o mesmo `mantraeeg.spec` chega (o PyInstaller
não faz compilação cruzada, tem de ser num Mac).
