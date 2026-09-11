# mantra-eeg — Especificação técnica v4

Estação de demonstração de EEG para festival de bem-estar, com **análise post-hoc**.

A pessoa coloca o equipamento, o operador liga o dispositivo e verifica o contacto. Segue
uma calibração de 60 s a olhar para uma cruz de fixação, enquanto o investigador explica.
A pessoa clica para iniciar, o vídeo do mantra arranca e ela medita ~4 min com evocação
mental do mantra, sem vocalização. No fim, a aplicação processa a gravação completa em
poucos segundos e apresenta os gráficos.

Hardware: **g.tec Unicorn Hybrid Black**, 250 Hz. Entrega: **executável Windows**.
Alvo: v1 funcional em 24 h.

Documentos:
- Este ficheiro — arquitetura, protocolo, empacotamento, critérios de aceitação.
- `BIOMARKERS.md` — definição dos marcadores, assinaturas das funções, bibliografia.
  **Normativo sobre este documento no que respeita a fórmulas e nomes de métricas.**
- `PROMPT_CLAUDE_CODE.md` — instruções de arranque para os agentes.

Alterações v3→v4: **nada é calculado em tempo real** exceto qualidade de contacto. Todos os
biomarcadores são computados após a aquisição. Isto permite filtragem de fase zero, correção
de EOG por regressão, sobreposição de épocas menor e bootstrap defensável. O relatório da
v1 é um **relatório de investigador** com os onze marcadores e uma tabela de diagnóstico
para seleção dos três melhores.

---

## 1. Hardware e aquisição

### 1.1 Dispositivo

Unicorn Hybrid Black. Amostragem **fixa em 250 Hz**. Buffer de **17 canais**: 8 de EEG
(índices 0–7), 3 de acelerómetro (8–10), 3 de giroscópio (11–13), bateria (14), contador
(15), indicador de validação (16). Ler estes índices das constantes da API, não hardcoded.

### 1.2 Montagem

Montagem personalizada, **não** a default do Unicorn:

| Índice | Posição | Uso |
|---|---|---|
| 0 | F3 | métricas |
| 1 | F4 | métricas |
| 2 | C3 | métricas |
| 3 | C4 | métricas |
| 4 | Fp1 | regressor de EOG e deteção de pestanejo |
| 5 | Fp2 | regressor de EOG e deteção de pestanejo |
| 6 | Oz | gravado, não usado na v1 |
| 7 | Pz | gravado, não usado na v1 |

Mapeamento índice→posição na config. **Confirmar contra o hardware real no M0**: se F3 e F4
estiverem trocados, o ALAY sai com o sinal invertido e ninguém repara.

Referência: manter a do hardware. **Não aplicar CAR** sobre estes canais anteriores
agrupados, distorceria a assimetria e a coerência F3–F4. Opção de config, desligada.

Rede: **50 Hz**.

### 1.3 Aquisição sem API licenciada

A Unicorn Python API exige licença e está **fora de âmbito**. A C API (`Unicorn.dll`) vem
incluída no Suite. Ordem de tentativa, a decidir empiricamente nos primeiros 10 minutos:

1. **`BrainflowSource`** — `pip install brainflow`, `BoardIds.UNICORN_BOARD`. Consome a
   mesma DLL. Primeira escolha.
2. **`UnicornDllSource`** — binding em `ctypes` sobre `Unicorn.dll` (16 funções do
   `unicorn.h`). Segunda escolha.
3. **`LSLSource`** — via UnicornLSL. Fallback; o executável tem de lançar e vigiar o
   UnicornLSL como subprocesso.

`UnicornPy`, C# e bridges UDP estão excluídos.

Desenvolvimento: **`SyntheticSource`** (com guiões, ver 6.2) e **`ReplaySource`**. Trocar de
fonte é uma linha na config.

### 1.4 O que corre em tempo real

**Só duas coisas**, e nenhuma delas é um biomarcador:

1. **Escrita da gravação.** Thread dedicada de aquisição → ring buffer → ficheiro. A UI
   nunca lê do dispositivo.
2. **Monitor de contacto.** Por canal, a cada 1 s, sobre os últimos 2 s: desvio padrão,
   amplitude pico-a-pico, e razão de potência 48–52 Hz sobre 1–45 Hz. Semáforo verde /
   amarelo / vermelho no ecrã do operador. Isto é vigilância de elétrodos, não análise.

Nenhum biomarcador, nenhum índice, nenhuma normalização acontece durante a sessão.

### 1.5 Acelerómetro e giroscópio

Gravados sempre, na mesma base temporal do EEG. Usados na análise post-hoc para:
- rejeição de épocas por movimento (magnitude de aceleração menos 1 g > `0.06 g`, magnitude
  do giroscópio > `8 °/s`);
- **diagnóstico de marcadores**: correlação entre a série temporal de cada marcador e o
  índice de movimento (ver 5.3).

---

## 2. Protocolo

| Estado | Duração | Ocular | Ecrã do participante |
|---|---|---|---|
| `IDLE` | — | — | Splash Neroes |
| `CONNECT` | livre | — | Procurar e ligar dispositivo |
| `CONTACT` | livre | aberto | Instruções; semáforos no ecrã do operador |
| `MENU` | livre | aberto | Tempos de calibração e meditação, editáveis |
| `CALIBRATION` | 60 s | **aberto** | Cruz de fixação, fundo escuro |
| `WAIT_START` | livre | aberto | Cruz + botão para iniciar |
| `MANTRA` | 240 s | **aberto** | Vídeo do mantra, fullscreen |
| `SETTLE` | 15 s | aberto | Cruz, sem áudio |
| `ANALYSIS` | ~segundos | — | Ecrã de processamento com progresso |
| `REPORT` | — | aberto | Gráficos |

Durações editáveis **na interface**, não apenas no YAML. Defaults na config.

`CALIBRATION`: só os **últimos `calib_use_last_s`** segundos (default 30) entram na análise.
Os primeiros 30 estão contaminados pela explicação em voz alta e pelas respostas da pessoa.

**Invariante ocular.** A condição ocular da calibração tem de ser igual à da meditação. Cruz
com olhos abertos seguida de vídeo com olhos abertos é consistente. O fecho das pálpebras
multiplica o alfa por si só (efeito Berger), pelo que calibração de olhos abertos com
meditação de olhos fechados **não é comparável**. A condição ocular viaja com os dados e a
função de comparação levanta exceção se os segmentos diferirem.

Não avançar de `CONTACT` com menos de 4 dos 6 canais úteis bons. Se a cobertura de épocas
boas na calibração for < 60 %, avisar o operador no relatório e permitir repetir a sessão.

Modos `blind` (olhos fechados) e `bar` (feedback em tempo real): **v2**. Não implementar. Se
`blind` for implementado, a calibração passa obrigatoriamente a olhos fechados.

---

## 3. Análise post-hoc

Corre uma vez, sobre a gravação completa, na transição `SETTLE` → `ANALYSIS`.
Orçamento: **< 10 s** para uma sessão de ~5,5 min. Se exceder, o gargalo é o bootstrap;
reduzir reamostragens antes de reduzir marcadores.

### 3.1 Pré-processamento

Sobre o registo completo, na ordem:

1. Remoção de média e detrend linear por canal.
2. Notch IIR a 50 e 100 Hz (Q = 30).
3. Passa-banda 1–45 Hz, Butterworth ordem 4, **`sosfiltfilt`** (fase zero). Só é possível
   offline e é uma das razões para esta arquitetura.
4. **Correção de EOG por regressão.** Regredir Fp1 e Fp2 (filtrados a 1–5 Hz) de F3, F4, C3
   e C4 por mínimos quadrados, e subtrair a componente prevista. Recupera dados que de
   outro modo seriam rejeitados. Guardar os coeficientes no `offline.json` para auditoria.
   Ligável/desligável na config, **ligado por defeito**, e o relatório indica se foi usado.

Não usar ICA: com 6 canais úteis a decomposição é instável e não há tempo para inspeção
manual.

### 3.2 Epocagem

- Janela **4,0 s**, hop **2,0 s** (sobreposição 50 %).
- Welch com sub-segmentos de 2,0 s, Hann, sobreposição 50 % → resolução 0,5 Hz.
- Sem suavização temporal. Sem EMA. Os gráficos mostram as épocas tal como são.
- Para coerência e wPLI: janela **10,0 s**, hop **5,0 s**, sub-segmentos de 1,0 s (≥19
  segmentos por janela). Cadência própria, mais lenta. Ver `BIOMARKERS.md` B10.

A sobreposição de 50 % substitui a de 87,5 % da v3 e é o que torna o bootstrap por blocos
defensável.

### 3.3 Gate de qualidade

Por época e por canal, época marcada má se qualquer condição se verificar:

- amplitude absoluta máxima > 120 µV;
- diferença entre amostras consecutivas > 30 µV;
- desvio padrão < 0,5 µV (elétrodo morto);
- potência relativa 30–45 Hz > 0,35 (EMG);
- potência 48–52 Hz relativa a 1–45 Hz > 0,30 (linha);
- movimento por acelerómetro ou giroscópio (1.5) → invalida todos os canais.

Aplicado **depois** da correção de EOG, para não rejeitar épocas que a regressão salvou.
Pestanejo residual acima de 75 µV em 1–5 Hz em Fp1/Fp2 continua a invalidar F3/F4.

Limiares na config. Um marcador só é calculado se todos os canais de que depende estiverem
bons nessa época; caso contrário `NaN`. Épocas `NaN` aparecem como faixas cinzentas nos
gráficos. **Nunca interpolar.**

### 3.4 Normalização

Média e desvio padrão das épocas boas do segmento de **calibração** (últimos 30 s). Todos os
marcadores são apresentados como z-score e como % de variação face a essa referência.
Nunca limiares absolutos entre pessoas.

O desvio padrão de 30 s de calibração é uma estimativa pobre. Consequência obrigatória: usar
**mediana e desvio absoluto mediano (MAD)** em vez de média e desvio padrão, e reportar o
número de épocas boas usadas na estimativa em todos os painéis.

---

## 4. Biomarcadores

As definições, canais, direções esperadas, assinaturas de funções e bibliografia estão em
`BIOMARKERS.md`, que é normativo. Resumo dos onze:

| ID | Nome | Canais |
|---|---|---|
| `alpha_rel` | alfa relativo frontocentral | F3 F4 C3 C4 |
| `beta_rel` | beta relativo frontal | F3 F4 |
| `alpha_beta` | razão alfa/beta | F3 F4 C3 C4 |
| `smr_rel` | SMR relativo central | C3 C4 |
| `theta_rel` | teta relativo frontal | F3 F4 |
| `faa` | ALAY / assimetria alfa frontal | F3 F4 |
| `coh_alpha1` | coerência alfa1 interhemisférica | F3–F4, C3–C4 |
| `coh_theta2` | coerência teta2 interhemisférica | F3–F4 |
| `wpli_alpha1` | wPLI alfa1 (verificação de robustez) | F3–F4 |
| `ap_exponent` | expoente aperiódico (specparam) | F3 F4 C3 C4 |
| `lziv` | complexidade de Lempel-Ziv | F3 F4 C3 C4 |

Extras derivados, calculados e reportados mas não contados como marcadores autónomos:
frequência de pico do alfa, entropia espectral, potência de banda corrigida pelo aperiódico.

### 4.1 Registo de marcadores

Todos os marcadores são declarados num **registo único** (`markers.py`), não espalhados por
funções ad-hoc:

```python
@dataclass(frozen=True)
class Marker:
    id: str
    label: str                      # rótulo em português para o gráfico
    kind: Literal["psd", "pair", "timeseries"]
    channels: tuple[str, ...]       # ou pares, para kind="pair"
    fn: Callable                    # assinatura conforme BIOMARKERS.md
    expected_direction: Literal["up", "down", "unknown"]
    in_composite: bool
    notes: str                      # cautelas a imprimir no relatório
```

O relatório, a tabela de diagnóstico e a agregação entre sessões **iteram o registo**.
Acrescentar um marcador é acrescentar uma entrada, não editar cinco ficheiros.

`theta_rel`, `faa`, `ap_exponent`, `lziv`, `wpli_alpha1` têm `in_composite=False`, pelas
razões documentadas em `BIOMARKERS.md`. `expected_direction="unknown"` para `ap_exponent` e
`lziv`.

### 4.2 Índice composto

```
calm_index = w_a*z(alpha_rel) - w_b*z(beta_rel) + w_s*z(smr_rel)
```

Pesos (condição olhos abertos, o caso da v1): `w_a = 0.35`, `w_b = 0.40`, `w_s = 0.25`.
Olhos fechados, para v2: `0.50 / 0.30 / 0.20`. Se uma componente for `NaN`, renormalizar os
pesos das restantes.

---

## 5. Estatística e diagnóstico de marcadores

Este é o núcleo da v1. O objetivo declarado da v1 **não** é impressionar o participante, é
dar ao investigador dados para escolher os três marcadores a manter.

### 5.1 Efeito por sessão

Para cada marcador, comparar a calibração com cada terço da meditação (M1, M2, M3):

- **Variação percentual** da mediana.
- **Delta de Cliff** como dimensão de efeito robusta e não paramétrica. Não assumir
  normalidade nem homocedasticidade em séries de uma pessoa.
- **IC 95 % por bootstrap por blocos**, blocos de 8 s, 2000 reamostragens. Blocos porque as
  épocas se sobrepõem 50 %.
- **Declive Theil-Sen** ao longo da meditação, com IC bootstrap.

Não apresentar p-values de testes de permutação época-a-época. Com épocas sobrepostas é
anticonservador e não é honesto.

Se o IC cruzar zero, o texto diz **"estável"**, nunca "melhorou".

### 5.2 Tabela de diagnóstico por marcador

Uma linha por marcador, em `marker_report.csv` e num painel do relatório:

| Coluna | Definição |
|---|---|
| `marker` | ID |
| `direction_ok` | o efeito foi na `expected_direction`? |
| `pct_change` | variação % calibração → meditação |
| `cliffs_delta` | dimensão de efeito |
| `ci_low`, `ci_high` | IC bootstrap |
| `crosses_zero` | booleano |
| `coverage` | % de épocas boas para este marcador |
| `split_half` | o efeito replica na 1ª e na 2ª metade da meditação? sinal e magnitude de cada |
| `r_emg` | correlação de Spearman entre a série do marcador e o índice de EMG |
| `r_motion` | correlação de Spearman com o índice de movimento |
| `n_epochs` | épocas boas usadas |

`r_emg` e `r_motion` são as colunas mais importantes desta tabela. Um marcador com `|r_emg|`
alto está a medir músculo, não cérebro, por muito bonito que o gráfico seja. Marcar a
vermelho acima de `0.5` (limiar na config).

### 5.3 Agregação entre sessões

`scripts/aggregate_sessions.py` percorre `data/sessions/`, junta os `marker_report.csv` e
produz um ranking em `data/aggregate/ranking.csv` e `ranking.png`.

Métrica de seleção primária, e é esta que decide os três marcadores:

```
consistency = fração de sessões em que o efeito foi na direção esperada
              E o IC não cruzou zero
```

Colunas secundárias: mediana de `cliffs_delta`, mediana de `coverage`, mediana de `|r_emg|`,
número de sessões.

**Não escolher pela dimensão média do efeito.** Numa banca, um marcador com efeito grande em
metade das pessoas e invertido na outra metade é pior que um marcador modesto e consistente.
O script imprime um ranking sugerido mas **não** decide: a decisão é do investigador e fica
registada na config como `featured_markers: [id, id, id]`.

---

## 6. Estrutura do código

```
mantra-eeg/
  config/
    default.yaml
    scripts/                    # guiões da fonte sintética
  assets/
    neroes_logo.png
    mantra.mp4
  src/mantraeeg/
    config.py        bands.py        ring_buffer.py
    sources/
      base.py  brainflow_src.py  unicorn_dll.py  lsl.py
      replay.py  synthetic.py
    acquisition.py                # thread + gravação + monitor de contacto
    preprocess.py                 # filtros, regressão de EOG
    quality.py                    # gate de épocas, índices de EMG e movimento
    spectral.py                   # Welch, potência de banda, relativa
    connectivity.py               # coerência, wPLI
    markers.py                    # REGISTO + funções de cálculo
    normalize.py                  # mediana/MAD da calibração
    stats.py                      # Cliff, bootstrap por blocos, Theil-Sen, split-half
    analysis.py                   # orquestra a passagem post-hoc
    recorder.py
    report/
      figures.py  diagnostics.py  pdf.py
    ui/
      app.py  connect.py  contact.py  menu.py  session.py  report_view.py
  scripts/
    check_unicorn.py              # smoke test de hardware
    run_booth.py                  # entrypoint
    analyse_session.py            # reprocessa uma gravação
    aggregate_sessions.py         # ranking entre sessões
    simulate_session.py           # sessão completa headless sobre fonte sintética
  tests/
  data/sessions/  data/aggregate/
```

Nota: não há `session_pipeline` em tempo real. `acquisition.py` grava, `analysis.py` analisa.
A separação é deliberada e é o que torna esta versão simples.

### 6.1 `check_unicorn.py`

O primeiro entregável do projeto. Sem UI, sem análise:

1. Tenta BrainFlow; se falhar, `ctypes` sobre `Unicorn.dll`; se falhar, lista streams LSL.
   Diagnóstico explícito em cada falha.
2. Lista dispositivos, liga, imprime número de canais e frequência lidos do dispositivo.
3. Adquire 30 s e imprime por canal: desvio padrão, pico-a-pico, razão 48–52 / 1–45 Hz.
4. Imprime estatísticas de acelerómetro e giroscópio.
5. Verifica continuidade do contador (deteta amostras perdidas).
6. Desliga de forma limpa.

Valida montagem, contacto, taxa real e caminho de aquisição antes de existir mais nada.

### 6.2 Fonte sintética com guião

YAML com segmentos de parâmetros espectrais alvo, estado ocular por segmento, e artefactos
injetáveis. Para os marcadores de conectividade, o guião tem de poder injetar **coerência
controlada** entre canais (fonte comum com peso configurável), senão B10 fica sem teste.

```yaml
segments:
  - {name: calib,  duration_s: 60,  eye: open, alpha_h: 0.3, theta_h: 0.3,
     beta_h: 0.9, exponent: 1.4, coh_alpha: 0.2}
  - {name: mantra, duration_s: 240, eye: open, alpha_h: 0.6, theta_h: 0.5,
     beta_h: 0.4, exponent: 1.35, coh_alpha: 0.6, ramp: true}
artifacts:
  blinks_per_min: 18
  motion_events_per_min: 1.5
  jump_prob_per_epoch: 0.01
  emg_bursts_per_min: 2
```

Guiões obrigatórios: `stressed_then_calm`, `flat_no_change`, `heavy_artifact`,
`coherence_only` (só a coerência muda), `emg_confound` (o "efeito" é inteiramente EMG — o
diagnóstico tem de o apanhar em `r_emg`).

### 6.3 Saída por sessão

```
data/sessions/2026-09-09T14-32-07_p017/
  raw.npy              (17, n_samples) float32
  raw_meta.json        sfreq real, mapa de canais, referência, condição ocular,
                       durações usadas, versão do código, git hash
  epochs.parquet       uma linha por época: t, fase, todos os marcadores, flags de qualidade
  events.json          transições, timestamp de início do vídeo, abortos, notas do operador
  offline.json         specparam por fase, coeficientes da regressão de EOG, ICs bootstrap
  marker_report.csv    a tabela de 5.2
  report.png  report.pdf
```

Sem dados pessoais em nomes de ficheiro. Só ID sequencial.

---

## 7. Relatório

Dois perfis, escolhidos na config (`report_profile`).

### 7.1 `researcher` — default da v1

Multi-página. É a ferramenta para escolher os três marcadores.

1. **Página de diagnóstico.** A tabela de 5.2 renderizada, com as colunas `r_emg` e
   `r_motion` destacadas. Cobertura global. Se a regressão de EOG foi aplicada.
2. **Um painel por marcador**, gerado por iteração do registo. Cada painel: série temporal
   em z, marca da transição calibração→mantra, faixas cinzentas de qualidade, barras
   calibração / M1 / M2 / M3 com IC bootstrap, e o texto de `notes` do registo impresso em
   pequeno (as cautelas de `BIOMARKERS.md`).
3. **Painel de qualidade.** Índice de EMG, índice de movimento e cobertura por canal ao
   longo do tempo, no mesmo eixo temporal dos marcadores. Serve para ler os painéis
   anteriores com desconfiança informada.
4. **Espectros.** Log-log médio por fase, com o ajuste specparam sobreposto, por canal.

### 7.2 `participant` — para quando os três estiverem escolhidos

Uma página, só os `featured_markers` da config, mais o `calm_index`. Linguagem simples.
Frase de leitura automática por regras a partir dos ICs; **nunca inventa melhoria**; se nada
mudou de forma fiável diz "o teu sinal manteve-se estável". Rodapé obrigatório: demonstração
científica, não é diagnóstico nem avaliação clínica. Cabeçalho: condição ocular, durações,
cobertura de qualidade.

---

## 8. Interface e executável

### 8.1 Fluxo

Conforme a tabela da secção 2. Duas janelas: consola do operador e ecrã do participante.

**Consola do operador:** traçados crus dos 6 canais úteis (5 s), painel do acelerómetro,
semáforo por canal com o motivo, fase e tempo restante, botões (ligar, verificar contacto,
iniciar, abortar, nova sessão), notas de texto livre gravadas em `events.json`.
Sem métricas ao vivo, porque não existem.

**Ecrã do participante:** splash Neroes, instruções, cruz de fixação, botão de início, vídeo
do mantra fullscreen, ecrã de processamento com progresso, relatório.

Áudio: vem do próprio vídeo; o timestamp do início da reprodução vai para `events.json`.

Nova sessão sem reiniciar a aplicação nem reconectar o dispositivo.

### 8.2 Empacotamento

- PyInstaller **one-folder** (não one-file: arranque mais rápido e a `Unicorn.dll` resolve-se
  melhor).
- `config/`, `assets/` e o vídeo **externos ao executável**, alteráveis no evento sem
  rebuild.
- Hidden imports explícitos: `scipy.special`, `scipy.signal`, `specparam`, `pyqtgraph`,
  backends de `matplotlib`.
- `Unicorn.dll` localizada em runtime: variável de ambiente, depois caminho default do
  Suite, depois pasta da aplicação. Mensagem clara se não encontrar.
- Sem privilégios de administrador, sem instalador. Pasta que se copia para uma pen.
- Saída em `./data/` relativa ao executável.
- **Testar o build numa máquina que nunca teve Python instalado.** É a única forma de
  apanhar dependências que o ambiente de desenvolvimento estava a resolver.

---

## 9. Critérios de aceitação

1. `python scripts/check_unicorn.py` liga ao hardware real, imprime 250 Hz, 17 canais e
   estatísticas plausíveis por canal.
2. `python scripts/simulate_session.py --script stressed_then_calm --headless` corre uma
   sessão completa sem hardware e produz `report.pdf` e `marker_report.csv`.
3. Nessa sessão, `alpha_beta` e `calm_index` sobem com IC que não cruza zero.
4. `--script flat_no_change`: nenhum marcador reporta melhoria; o texto diz "estável".
5. `--script coherence_only`: `coh_alpha1` move-se, os marcadores de potência ficam estáveis.
   Prova que a conectividade está a medir conectividade.
6. `--script emg_confound`: o marcador afetado aparece com `r_emg` acima do limiar e marcado
   a vermelho na tabela de diagnóstico. **Este é o teste mais importante da v1.**
7. `--script heavy_artifact`: faixas cinzentas coerentes e cobertura reportada próxima do
   esperado.
8. Comparar segmentos com condições oculares diferentes levanta exceção.
9. `aggregate_sessions.py` sobre ≥3 sessões simuladas produz `ranking.csv` ordenado por
   `consistency`.
10. Acrescentar um marcador ao registo fá-lo aparecer automaticamente no relatório, na tabela
    de diagnóstico e na agregação, sem editar mais nenhum ficheiro.
11. Trocar a fonte na config é a única alteração para mudar de aquisição.
12. Alterar as durações na interface altera o protocolo; os valores usados ficam em
    `raw_meta.json`.
13. Análise post-hoc de uma sessão de 5,5 min em menos de 10 s.
14. `pytest` verde, com cobertura em `preprocess`, `quality`, `spectral`, `connectivity`,
    `markers`, `normalize`, `stats`.

---

## 10. Anti-objetivos

- Não calcular biomarcadores em tempo real. Nada de barra, nada de EMA, nada de feedback.
- Não inventar melhoria quando os dados não a mostram.
- Não normalizar meditação de olhos fechados contra calibração de olhos abertos.
- Não escolher os três marcadores por intuição nem por dimensão média de efeito.
- Nada de linguagem clínica, diagnóstica, ou de "idade cerebral".
- Nada de limiares absolutos entre pessoas.
- Não usar Oz/Pz na v1.
- Não recolher emails na v1 (ver 11).
- Não usar ICA com 6 canais.
- Não introduzir torch, tensorflow ou sklearn.

---

## 11. Add-on para depois

Recolha opcional de email para envio do relatório. Antes de escrever uma linha:
consentimento explícito em ecrã separado com finalidade escrita; só o email, sem nome nem
idade; prazo de retenção e apagamento automático. O EEG é dado biométrico e associá-lo a um
identificador direto muda o enquadramento de RGPD. Considerar enviar só o PDF e apagar o
raw. Deixar a arquitetura pronta (o relatório já é um PDF num caminho conhecido) e nada mais.

---

## 12. Dependências

`numpy`, `scipy`, `pyyaml`, `pandas`, `pyarrow`, `matplotlib`, `specparam`, `pyqtgraph`,
`PyQt6`, `brainflow`, `pylsl`, `pytest`. `Unicorn.dll` vem do Unicorn Suite, não do pip.
Python 3.11 **x64**, Windows, PowerShell, venv.
