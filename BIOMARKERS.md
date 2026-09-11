# Biomarcadores — mantra-eeg

Marcadores computáveis com F3, F4, C3, C4 (Fp1/Fp2 só para deteção de artefacto ocular;
Oz/Pz excluídos por qualidade). Sessão de calibração + ~4 min de meditação, 250 Hz.

Todos os marcadores são calculados **post-hoc**, sobre a gravação completa, depois da
sessão terminar. Nenhum é calculado em tempo real na v1.

Convenções:
- Potência **relativa** = potência da banda / potência total 1–45 Hz.
- Todos os valores apresentados são **variação face à calibração da própria pessoa**,
  normalizados por **mediana e MAD** (não média e desvio padrão: 30 s úteis de calibração
  dão uma estimativa demasiado pobre para o desvio padrão), na mesma condição ocular.
- Bandas: teta 4–7 (teta2 6–8), alfa 8–12 (alfa1/baixo 8–10, alfa2/alto 10–12),
  SMR 12–15, beta 15–30 Hz. Beta começa em 15 Hz para não sobrepor ao SMR.
- Épocas de potência: 4,0 s com hop de 2,0 s. Épocas de conectividade: 10,0 s com hop de
  5,0 s. Ver `SPEC.md` 3.2.

---

## Assinaturas das funções

Contrato comum. Todas as funções de marcador são **puras**: sem estado, sem I/O, sem leitura
de config global. Recebem dados e um objeto de bandas, devolvem um `float` (ou `nan`).

```python
# spectral.py ------------------------------------------------------------------
def welch_psd(x: np.ndarray, sfreq: float, seg_s: float = 2.0
              ) -> tuple[np.ndarray, np.ndarray]:
    """x: (n_ch, n_samples) de uma época. Devolve (freqs, psd) com psd (n_ch, n_freqs)."""

def band_power(psd: np.ndarray, freqs: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """Integra a PSD na banda (regra do trapézio). Devolve (n_ch,) em µV²."""

def relative_power(psd, freqs, band, total=(1.0, 45.0)) -> np.ndarray:
    """band_power(band) / band_power(total), por canal. Adimensional, 0–1."""

# connectivity.py --------------------------------------------------------------
def coherence_band(x: np.ndarray, y: np.ndarray, sfreq: float,
                   band: tuple[float, float], seg_s: float = 1.0) -> float:
    """Coerência de magnitude quadrada (MSC) média na banda, entre dois canais.
    Exige janela longa: >=19 sub-segmentos. Ver B10."""

def wpli_band(x, y, sfreq, band, seg_s=1.0) -> float:
    """weighted Phase Lag Index médio na banda. Cego a acoplamento a lag zero
    por construção — é isso que o torna uma verificação e não um substituto. Ver B10."""

# markers.py -------------------------------------------------------------------
# kind="psd": recebem a PSD da época já calculada.
def alpha_rel(psd, freqs, chans: dict[str, int], bands) -> float:
    """mean over F3,F4,C3,C4 de relative_power(alpha)."""

def beta_rel(psd, freqs, chans, bands) -> float:      # mean F3,F4, banda beta
def smr_rel(psd, freqs, chans, bands) -> float:       # mean C3,C4, banda smr
def theta_rel(psd, freqs, chans, bands) -> float:     # mean F3,F4, banda theta
def alpha_beta(psd, freqs, chans, bands) -> float:    # alpha_rel_fc / beta_rel_f
def faa(psd, freqs, chans, bands) -> float:
    """log(band_power(alpha)[F4]) - log(band_power(alpha)[F3]).
    Potência ABSOLUTA (log-ratio, a normalização cancela). nan se F3 ou F4 mau. Ver B6."""

def ap_exponent(psd, freqs, chans, bands, fit_range=(2.0, 40.0)) -> float:
    """Expoente aperiódico via specparam, média dos 4 canais. Ver B7."""

def alpha_peak_freq(psd, freqs, chans, bands) -> float:   # do ajuste specparam
def spectral_entropy(psd, freqs, band=(1.0, 45.0)) -> float
def corrected_band_power(psd, freqs, band, fit_range=(2.0, 40.0)) -> float
    """Altura do pico gaussiano do specparam na banda. Isola oscilação de broadband."""

# kind="pair": recebem o sinal no tempo, não a PSD.
def coh_alpha1(x: np.ndarray, sfreq, chans, bands) -> float:   # MSC F3-F4 em 8-10 Hz
def coh_theta2(x, sfreq, chans, bands) -> float                # MSC F3-F4 em 6-8 Hz
def wpli_alpha1(x, sfreq, chans, bands) -> float               # wPLI F3-F4 em 8-10 Hz

# kind="timeseries"
def lziv(x: np.ndarray, chans, bands) -> float:
    """Complexidade de Lempel-Ziv sobre binarização pela mediana, média dos canais,
    normalizada pelo comprimento da sequência. Ver B8."""

# quality.py -------------------------------------------------------------------
def emg_index(psd, freqs) -> np.ndarray:      # relative_power(30-45 Hz), por canal
def line_index(psd, freqs) -> np.ndarray:     # P(48-52) / P(1-45), por canal
def motion_index(accel: np.ndarray, gyro: np.ndarray) -> float
```

`emg_index` e `motion_index` não são marcadores: são as séries com que cada marcador é
correlacionado na tabela de diagnóstico (`SPEC.md` 5.2). Um marcador cuja série
correlaciona com o EMG está a medir músculo.

---

## Núcleo do índice (bem sustentados)

### B1. Alfa relativo frontocentral

```
alpha_rel = mean_ch( P(8–12 Hz) / P(1–45 Hz) )   ch = F3, F4, C3, C4
```
Direção: **sobe** com relaxamento, **desce** com stress.

É o marcador mais sólido do conjunto. Na meta-análise de EEG espectral e stress psicossocial,
a potência alfa desce de forma consistente e foi o único dos três índices mais usados com
efeito significativo (g = 0,60; p = 0,001), independentemente da fase de stress
(antecipatória, reativa, recuperação) [3]. Do lado da meditação, o alfa aumenta em todos
os estados meditativos, sem depender do nível de profundidade [4].

### B2. Beta relativo frontal

```
beta_rel = mean_ch( P(15–30 Hz) / P(1–45 Hz) )   ch = F3, F4
```
Direção: **desce** com relaxamento.

Na mesma meta-análise a beta mostra tendência a subir com stress, mas o efeito agregado não
foi significativo (g = −0,31; p = 0,29) [3]. Em meditação concentrativa observou-se
decréscimo de beta central (13–25 Hz) e gama baixa (25–48 Hz) nas absorções mais profundas
[4]. Usar como componente do índice e do rácio, não isoladamente.

**Requisito de protocolo:** evocação mental do mantra, sem vocalização. EMG mandibular
contamina 15–30 Hz e destrói este marcador.

### B3. Razão alfa/beta (índice de relaxamento)

```
alpha_beta = alpha_rel_fc / beta_rel_f
```
Direção: **sobe** com relaxamento. O inverso (beta/alfa, BAR) é o "índice de stress".

Os rácios alfa/beta e teta/beta correlacionam-se negativamente com stress e discriminam
stress de baseline de repouso (40 sujeitos, stressor em realidade virtual) [7]. Há valores
de referência publicados: em estado relaxado a BAR foi 0,701, contra média de 2,403 em
não-jogadores após 1 h de jogo estratégico, regressando a 0,709 após 12 min de música de
tom baixo [8]. Outro trabalho propõe β/α ≥ 1,5 como indicador de stress [9]. Tratar estes
números como ordem de magnitude, não como limiares transferíveis: o teu hardware,
referência e montagem são diferentes.

### B4. SMR em C3/C4

```
smr_rel = mean_ch( P(12–15 Hz) / P(1–45 Hz) )   ch = C3, C4
```
Direção: **sobe** com calma alerta e quietude motora.

O protocolo validado usa exatamente potência **relativa** de SMR no elétrodo **C3**: em 33
sujeitos saudáveis, o treino aumentou a atividade SMR em C3 com especificidade de banda, e
o aumento de SMR correlacionou negativamente com a severidade da ansiedade [10]. A
especificidade face à banda adjacente está demonstrada: treino de SMR (12–15 Hz) e de beta1
(15–18 Hz) produziram ambos cansaço, mas o aumento de calma apareceu só com SMR [11]. Uma
única sessão de ↑SMR/↓teta reduziu ansiedade (d = 0,9 entre grupos) e cortisol salivar
(d = 0,7) face a neurofeedback sham [12] — relevante porque a tua sessão é única e curta.

Bónus: o SMR cai com movimento, o que o torna também um indicador implícito de quietude
corporal.

---

## Painel próprio, fora do índice de calma

### B5. Teta frontal

```
theta_rel = mean_ch( P(4–7 Hz) / P(1–45 Hz) )   ch = F3, F4
```
Rótulo a usar: **atenção internalizada**, não "calma".

**A direção é contestada e é por isso que este marcador não entra no índice composto.**

A favor de subir: estados de bem-estar acompanham-se de sincronização de teta frontal
anterior e da linha média, com centro de gravidade na região pré-frontal esquerda; os
scores subjetivos de experiência emocional correlacionaram com teta e os de atenção
internalizada com teta e alfa baixo [1]. Em meditação concentrativa, o teta frontal medial
e temporo-parietal aumentou nas absorções mais profundas [4]. Em revisão sistemática de 11
estudos de meditação Om, a modulação da banda teta foi o achado EEG mais consistente,
acompanhada de alterações de alfa indicativas de estado relaxado mas atento [5].

Contra: teta e alfa relacionam-se com a profundidade de meditação de forma precisamente
inversa — a amplitude de teta correlacionou **positivamente** com "hindrances" e cada vez
mais **negativamente** com o aumento dos níveis de profundidade [2]. E num protocolo
within-subject de 2026, o teta **desceu** durante meditação face ao controlo ativo, embora
dentro de cada prática mais teta se associasse a maior mindfulness reportado [6].

Referência mais próxima do protocolo desta banca: em repetição **silenciosa** de OM, a
energia absoluta e relativa de teta aumentou significativamente face ao estado anterior,
sem alterações na repetição silenciosa de "one" nem em repouso, em 15 sujeitos com ~5 anos
de prática de dharana [13].

### B6. ALAY / assimetria alfa frontal

```
faa = ln( P_abs_alpha[F4] ) − ln( P_abs_alpha[F3] )
```
Única métrica em potência absoluta (é um log-ratio, a normalização cancela). Exige F3 **e**
F4 bons na época, senão `NaN`.

Fórmula tal como usada na meta-análise de FAA e depressão: diferença entre o logaritmo dos
valores espectrais absolutos no ritmo alfa em F4 e F3, (ln µV² HD − ln µV² HE) [14].
**Confirmar a convenção de sinal do ALAY interno da Neroes contra esta**, ou todos os
gráficos saem invertidos.

Enquadramento correto: o modelo aproximação/afastamento postula que maior atividade frontal
esquerda relativa indica propensão para aproximar ou envolver-se com um estímulo, e menor
atividade frontal esquerda relativa indica menor motivação de aproximação ou maior
afastamento [15]. A FAA evolui em quatro escalas temporais distintas, da escala
segundo-a-segundo à do desenvolvimento, o que sustenta usá-la como processo
auto-regulatório ativo e não só como traço [16]. Neurofeedback de FAA (direita − esquerda)
produziu aumento de assimetria conduzido por maior alfa à direita e redução coerente de
afeto negativo e sintomas de ansiedade [17].

**Não usar como índice de stress.** Na meta-análise de stress psicossocial a FAA teve efeito
agregado nulo (g = 0,01; p = 0,93) e variação inconsistente entre estudos [3]. E numa
meta-análise em depressão, o efeito médio foi −0,03 (IC [−0,07; 0,01]), não significativo
[14].

**Limitação de duração.** A literatura metodológica recomenda aquisições longas para índices
fiáveis de assimetria e o par frontomedial é o menos fiável. Em 4 min, apresentar apenas a
variação intra-sujeito face à calibração, nunca o valor absoluto como leitura de "controlo
emocional".

---

## Exploratórios, com direção não estabelecida

### B7. Expoente aperiódico (specparam / FOOOF, 2–40 Hz)

Calcular por fase (calibração, terços do mantra) e usar para **corrigir** a potência de
banda, de modo a garantir que as subidas de alfa e teta são oscilatórias e não deslocamentos
broadband.

**Não usar como índice de calma, e não afirmar direção.** Numa revisão sistemática de 45
estudos, espectros mais inclinados associaram-se a controlo inibitório, resolução de
conflito e codificação, e espectros mais planos a envolvimento sensorial, flexibilidade
cognitiva e evocação [18]. Mas a validação farmacológica direta em murganhos concluiu que o
expoente aperiódico **não** constitui um marcador universalmente fiável do rácio
excitação/inibição: moduladores positivos de GABA-A aumentaram o expoente como previsto,
mas cetamina, antagonistas de GABA-A em dose subconvulsiva e a inibição de interneurónios
com DREADDs não seguiram a previsão [19].

### B8. Complexidade de Lempel-Ziv e entropia espectral

Existe relação forte, inversa e monotónica não trivial entre o declive 1/f e a complexidade
de Lempel-Ziv, consistente às escalas de ECoG e EEG, contrastando repouso com anestesia por
propofol [20]. Útil como marcador exploratório do estado global, com a nota de que é
largamente redundante com B7.

### B9. Frequência de pico do alfa

Estimada pelo specparam por fase. Exploratório.

### B10. Coerência alfa1 interhemisférica frontal — **o marcador específico de mantra**

```
coh_alpha1 = mean( MSC(F3, F4)[8–10 Hz] )        # e opcionalmente C3–C4, F3–C3, F4–C4
coh_theta2 = mean( MSC(F3, F4)[6–8 Hz] )
```

Direção: **sobe** durante prática com mantra e durante escuta de recitação em sânscrito.

Este é o marcador com maior especificidade para meditação baseada em mantra, e a razão é
que a literatura de Meditação Transcendental convergiu na coerência e não na potência.
Comparando MT com repouso de olhos fechados, a potência alfa **não** diferiu
significativamente, mas a coerência alfa frontal e ântero-posterior foi mais alta, com
efeitos presentes **no primeiro minuto** e mantidos ao longo dos 10 min de sessão [21]. Já
em 1981 a coerência alfa frontal se mostrou discriminador mais sensível da técnica do que a
potência alfa, explicando resultados nulos anteriores entre MT e relaxamento genérico [22].
Num longitudinal de um ano, a coerência frontal foi mais sensível do que a assimetria
lateral [23] — isto é, mais sensível do que o ALAY (B6).

Quadro completo face a repouso: maior log-potência alfa1 frontal, menor potência beta1 e
gama frontal e parietal, e maior coerência interhemisférica alfa1 frontal e parietal [24].
As experiências de transcendência são marcadas por inalação lenta e coerência frontal alfa1
(8–10 Hz) elevada [25].

**Escuta de sânscrito.** Em 37 sujeitos, a coerência teta2 e alfa1 frontal, parietal e
fronto-parietal foi significativamente mais alta durante escuta de recitação védica ao vivo
do que durante a própria prática de MT; a coerência teta2 é interpretada como atenção a
processos mentais internos [26].

**Restrições de implementação (obrigatórias):**
- Janela deslizante de **8–10 s** com sub-segmentos de **1 s** (≥15 segmentos). A janela de
  4 s com sub-segmentos de 2 s dá 3 segmentos e é insuficiente para estimar coerência. Esta
  métrica tem cadência própria, mais lenta que as de potência.
- Exige F3 **e** F4 simultaneamente bons na janela inteira, senão `NaN`. O gate de qualidade
  fica mais restritivo para esta métrica do que para as outras.
- **Volume conduction e referência comum inflacionam a coerência** entre elétrodos próximos.
  Duas mitigações, ambas obrigatórias: apresentar só variação face à calibração (o
  enviesamento da referência é constante entre condições), e calcular **wPLI** em paralelo
  como verificação de robustez. Aviso: existe trabalho que reporta aumento de sincronia de
  fase alfa **a lag zero** durante MT, e o wPLI é cego a isso por construção. Reportar as
  duas medidas e sinalizar divergência; não escolher uma às cegas.
- Toda esta literatura é de **olhos fechados**. Com olhos abertos e vídeo a transferência é
  incerta. É o argumento mais forte a favor de reintroduzir o modo cego na v2.

### B11. Notas sobre escuta passiva vs evocação mental

A escuta passiva é o caso mais fraco. O grupo do estudo da repetição silenciosa nota
explicitamente que, em trabalho anterior, o EEG durante a **escuta** de repetição de OM não
diferiu significativamente dos períodos de controlo, e foi essa a razão para passarem à
repetição silenciosa, onde o teta subiu apenas com OM [13]. Reforça a decisão de protocolo:
áudio como guia, evocação mental como tarefa.

Contexto convergente de outras modalidades, útil para a narrativa mas não computável com
este hardware: OM verbal e OM ouvido ativam áreas semelhantes das redes atencional,
frontoparietal de controlo e de modo padrão [27]; durante OM há redução das saídas da
ínsula, cingulado anterior e córtex orbitofrontal, incluindo para a amígdala [28]; e a
revisão sistemática de 24 estudos de cântico reporta ativação pré-frontal, da ínsula e do
giro cingulado com desativação da rede de modo padrão, e aumento de teta nos estudos de
EEG, mas com heterogeneidade que impediu meta-análise formal [29].

**Resultado nulo a conhecer:** uma análise de microestados EEG após OM verbal e OM ouvido,
com 128 canais e 23 sujeitos, identificou três topografias mas não encontrou efeitos
significativos do cântico de curta duração [30].

---

## Explicitamente excluídos

| Marcador | Motivo |
|---|---|
| Teta/beta como índice de calma | Desce com stress [7] mas sobe com divagação mental. Calcular e gravar, não mostrar. |
| Delta e gama frontais | Com elétrodos frontais é EOG e EMG. |
| Coerência em bandas fora de alfa1/teta2 | Sem suporte específico para mantra. Ver B10, que é a exceção justificada. |
| Qualquer classificador ML | Sem dados de treino, sem tempo, sem validação. |

---

## Índice composto revisto

```
calm_index = w_a * z(alpha_rel) - w_b * z(beta_rel) + w_s * z(smr_rel)
```

Pesos default (condição olhos abertos, que é o caso do modo cruz + vídeo):
`w_a = 0.35`, `w_b = 0.40`, `w_s = 0.25`.

Com olhos abertos o alfa é fraco e ruidoso, por isso o peso desloca-se para beta, que é o
que a literatura de stress sustenta melhor nessa condição. Para uma variante de olhos
fechados: `w_a = 0.50`, `w_b = 0.30`, `w_s = 0.20`.

`theta_rel` e `faa` são apresentados em painéis próprios, não somados ao índice.

**Regra invariante:** a condição ocular da calibração tem de ser igual à da meditação. Cruz
de fixação com olhos abertos seguida de vídeo com olhos abertos é consistente. Cruz com
olhos abertos seguida de meditação com olhos fechados não é comparável, porque o fecho das
pálpebras multiplica o alfa por si só.

---

## Referências

[1] Aftanas L, Golocheikine S. (2001). Human anterior and frontal midline theta and lower
alpha reflect emotionally positive state and internalized attention: high-resolution EEG
investigation of meditation. *Neuroscience Letters*.
https://consensus.app/papers/details/2372ab79dc9d536282128f06d838e6ca/

[2] Katyal S, et al. (2021). Alpha and theta oscillations are inversely related to
progressive levels of meditation depth. *Neuroscience of Consciousness*.
https://consensus.app/papers/details/2c2bbd7956985a0b8cef5e081f1fd9cc/

[3] Vanhollebeke G, et al. (2022). The neural correlates of psychosocial stress: A
systematic review and meta-analysis of spectral analysis EEG studies. *Neurobiology of
Stress*. https://consensus.app/papers/details/abc9d3c41d955adbaaabb464ac775722/

[4] DeLosAngeles D, et al. (2016). Electroencephalographic correlates of states of
concentrative meditation. *International Journal of Psychophysiology*.
https://consensus.app/papers/details/3eb450dd273553ec90e63aca638d14d4/

[5] Kumar R, et al. (2026). Neurophysiological Effects of Om Meditation: Evidence from EEG
Studies. *International Journal of Yoga*.
https://consensus.app/papers/details/cd7bb6e4d3125fc38f28812550c99119/

[6] White ML, et al. (2026). Meditation-Specific Neural Predictors of State Mindfulness
During Eyes Open Meditation. *Mindfulness*.
https://consensus.app/papers/details/0c8c7a047bb2522592ce2b2a30fef6b3/

[7] Wen TY, et al. (2020). Electroencephalogram (EEG) stress analysis on alpha/beta ratio
and theta/beta ratio. *Indonesian Journal of Electrical Engineering and Computer Science*.
https://consensus.app/papers/details/fe995c97afd55ead968ff14136c47cbe/

[8] Roy S, et al. (2021). EEG based stress analysis using rhythm specific spectral feature
for video gameplay. *Computers in Biology and Medicine*.
https://consensus.app/papers/details/a2f0e8138dde53809eb1771d0e3a31d3/

[9] Yusuf MSU, et al. (2019). Stress Identification during Sustained Mental Task and Brain
Relaxation Modeling with β/α Band Power Ratio. *ICEICT*.
https://consensus.app/papers/details/2a45a9b641ae524899ee98d6f0a24334/

[10] Liu S, et al. (2021). Sensorimotor rhythm neurofeedback training relieves anxiety in
healthy people. *Cognitive Neurodynamics*.
https://consensus.app/papers/details/ff84ede042ac58618609672ee01aba12/

[11] Gruzelier J. (2014). Differential effects on mood of 12-15 (SMR) and 15-18 (beta1) Hz
neurofeedback. *International Journal of Psychophysiology*.
https://consensus.app/papers/details/d88004f236a258be8e6d18cf24218181/

[12] Gadea M, et al. (2020). Effects of a single session of SMR neurofeedback training on
anxiety and cortisol levels. *Neurophysiologie Clinique*.
https://consensus.app/papers/details/ab20ac93b0b25e028b18cdfc812ea607/

[13] Pal S, et al. (2022). Changes in Brain Waves During Silent Repetition of OM: A
Crossover Study from India. *Journal of Religion and Health*.
https://consensus.app/papers/details/c49b27d4cb6c58fa8630c627c48b20b0/

[14] Horato N, et al. (2022). The relationship between emotional regulation and hemispheric
lateralization in depression: a systematic review and a meta-analysis. *Translational
Psychiatry*. https://consensus.app/papers/details/82face7d74585a039e2616509c111697/

[15] Allen JJB, et al. (2017). Frontal EEG alpha asymmetry and emotion: From neural
underpinnings and methodological considerations to psychopathology and social cognition.
*Psychophysiology*. https://consensus.app/papers/details/c6911a99b3c3504c871058244361590d/

[16] Perone S, et al. (2024). Frontal Alpha Asymmetry dynamics: A window into active
self-regulatory processes. *Biological Psychology*.
https://consensus.app/papers/details/4935a3a4956b5c74a7fa8ddff7c8f4ca/

[17] Mennella R, et al. (2017). Frontal alpha asymmetry neurofeedback for the reduction of
negative affect and anxiety. *Behaviour Research and Therapy*.
https://consensus.app/papers/details/99ccf5f274e759029ce91a34697bb71f/

[18] Hemmerich K, et al. (2026). Aperiodic EEG features of cognition: a systematic review.
*Reviews in the Neurosciences*.
https://consensus.app/papers/details/17d2bc5899fb569b9e1aaa82755db798/

[19] Salvatore SV, et al. (2024). Periodic and aperiodic changes to cortical EEG in response
to pharmacological manipulation. *Journal of Neurophysiology*.
https://consensus.app/papers/details/e2eacfc337e4517fb5058667a6c50043/

[20] Medel V, et al. (2020). Complexity and 1/f slope jointly reflect brain states.
*Scientific Reports*.
https://consensus.app/papers/details/58de83347efb52febde179b1e3e34ea3/

[21] Travis F, Wallace RK. (1999). Autonomic and EEG patterns during eyes-closed rest and
transcendental meditation (TM) practice: the basis for a neural model of TM practice.
*Consciousness and Cognition*.
https://consensus.app/papers/details/9cabac8296605d199f0b9565b93a878a/

[22] Dillbeck MC, Bronson EC. (1981). Short-term longitudinal effects of the transcendental
meditation technique on EEG power and coherence. *International Journal of Neuroscience*.
https://consensus.app/papers/details/d1872fe453ae5564807f5fa7a44a94fe/

[23] Travis F, et al. (2006). Cross-sectional and longitudinal study of effects of
Transcendental Meditation practice on interhemispheric frontal asymmetry and frontal
coherence. *International Journal of Neuroscience*.
https://consensus.app/papers/details/d042ab02b1235f23947253f555d68864/

[24] Travis F, et al. (2010). A self-referential default brain state: patterns of coherence,
power, and eLORETA sources during eyes-closed rest and Transcendental Meditation practice.
*Cognitive Processing*.
https://consensus.app/papers/details/2dd25f5d254e5b1096031c3f2caf1f3b/

[25] Travis F. (2014). Transcendental experiences during meditation practice. *Annals of the
New York Academy of Sciences*.
https://consensus.app/papers/details/452b7a764cd65002b187ee72b4ce4e11/

[26] Travis F, et al. (2017). Higher theta and alpha1 coherence when listening to Vedic
recitation compared to coherence during Transcendental Meditation practice. *Consciousness
and Cognition*. https://consensus.app/papers/details/beb5b09373375b35983824e873d0937d/

[27] Saini M, et al. (2023). Global Effect on Cortical Activity in Young Indian Males in
Response to "OM" Chanting: A High-Density Quantitative Electro-Encephalography Study.
*Annals of Neurosciences*.
https://consensus.app/papers/details/386d5481d9fa502ca6a6d14dbce3cfab/

[28] Rao NP, et al. (2018). Directional brain networks underlying OM chanting. *Asian
Journal of Psychiatry*.
https://consensus.app/papers/details/8812977a631a5342b791747f8587d2f2/

[29] Perry G, et al. (2025). Neural Correlates of Chanting: A Systematic Review. *WIREs
Cognitive Science*.
https://consensus.app/papers/details/60fbf498a28a533699e3c4c932347776/

[30] Tayade P, et al. (2024). Effect of short-term chanting on electroencephalographic
microstates. *Pan African Medical Journal*.
https://consensus.app/papers/details/79e4e8bd09795ca7830f25f9b06474e5/

[31] Hebert R, et al. (2005). Enhanced EEG alpha time-domain phase synchrony during
Transcendental Meditation: Implications for cortical integration theory. *Signal
Processing*. https://consensus.app/papers/details/8f1dfaf85d3a5a61b715c22116ed738d/

Contexto adicional: Polich J. (2006). Meditation states and traits: EEG, ERP, and
neuroimaging studies. *Psychological Bulletin* — revisão clássica, ativação de teta e alfa
relacionada com proficiência de prática.
https://consensus.app/papers/details/00e4c11293e95b099ef0a80384c0fca3/
