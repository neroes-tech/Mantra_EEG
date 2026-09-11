# Nomenclatura dos marcadores — mantra-eeg



## Grupo 1 — Relaxamento: os ingredientes e o resultado

| `id` | `label_public` | Direção | Explicação |
|---|---|---|---|
| `alpha_rel` | calma (alfa) | ↑ | Ritmo que o cérebro produz quando não está a processar o exterior. |
| `beta_rel` | Esforço mental | ↓ | Ritmo do pensamento ativo e do esforço. Pode subir se estiveres a interpretar o significado do mantra em vez de o repetir. |
| `alpha_beta` | Relaxamento | ↑ | O resultado da relação entre as duas anteriores: quanta calma há por cada unidade de esforço. |

**Painel agrupado obrigatório**, os três no mesmo eixo, com legenda:

> As duas primeiras linhas são ingredientes. O relaxamento é o resultado da relação entre
> elas: sobe quando a calma cresce mais do que o esforço. Se as duas subirem, o relaxamento
> pode até descer, e isso não é um erro de medição.

O `beta_rel` é o único marcador do relatório em que **descer é o que procuramos**. A seta no
gráfico não é opcional. E a nota sobre interpretar o mantra tem de estar visível: é a razão
mais provável para o esforço mental subir numa pessoa que está a fazer a prática bem.

---

## Grupo 2 — Corpo e atenção

| `id` | `label_public` | Direção | Explicação |
|---|---|---|---|
| `smr_rel` | Quietude corporal | ↑ | Ritmo das áreas motoras, que sobe quando o corpo está parado e a mente alerta. |
| `theta_rel` | Foco interior | ~ | Ritmo associado a atenção virada para dentro em vez de para o exterior. |

Cautela a imprimir em pequeno junto ao "Foco interior": a direção deste marcador é debatida
na literatura. Há trabalho que o associa a estados de bem-estar profundo e há trabalho que o
relaciona inversamente com a profundidade de meditação. Ver `BIOMARKERS.md` B5.

---

## Grupo 3 — O efeito do mantra

**Este é o grupo mais importante do relatório**, porque é o único específico de prática com
mantra em vez de meditação genérica. Merece a secção mais destacada.

| `id` | `label_public` | Direção | Explicação |
|---|---|---|---|
| `coh_alpha1` | Sintonia com Mantra | ↑ | O quanto os dois lados do cérebro oscilam em conjunto. É o marcador que a investigação sobre meditação com mantra identificou como o mais sensível. |
| `coh_theta2` | Sintonia do foco interior | ↑ | A mesma sintonia, mas no ritmo da atenção virada para dentro. Sobe em quem ouve recitação em sânscrito. |

### metricas a serem excluidas do efeito do mantra
| `imcoh_alpha1` | Sintonia — verificação sem contaminação | ↑ | Confirma que a sintonia é real e não um efeito da forma como os elétrodos estão ligados. |
| `wpli_alpha1` | Sintonia — verificação por desfasamento | ↑ | Segunda confirmação, por um método diferente. |

Texto de apoio para esta secção, com base científica:

> Em meditação com mantra, a potência das ondas muda pouco: o que muda é o quanto os dois
> lados do cérebro passam a oscilar em conjunto. É por isso que este é o grupo que olhamos
> com mais atenção. Num estudo com 37 pessoas, esta sintonia foi ainda mais alta durante a
> escuta de recitação védica ao vivo do que durante a própria prática de meditação.
. Ver `BIOMARKERS.md` B10.

---

## Grupo 4 — Controlo emocional

| `id` | `label_public` | Escala | Explicação |
|---|---|---|---|
| `faa` | Controlo emocional | **bipolar**, zero na calibração | A atividade frontal inclina-se para um dos lados conforme o estado emocional. Durante o mantra, a inclinação tende para o lado associado a aproximação e a afeto positivo. |

Requisitos de apresentação, todos obrigatórios:

- **Zero ao centro**, ancorado no valor de calibração da pessoa. Nunca uma barra de 0 a 100.
- **Os dois lados nomeados**: "aproximação / afeto positivo" e "afastamento / retração".
- Direção `up` assumindo a convenção `ln(alfa F4) − ln(alfa F3)`. **Se o ALAY interno da
  Neroes seguir a convenção inversa, o rótulo inverte-se.** Confirmar antes do evento.

Texto defensável, para usar no relatório:

> Durante o mantra, a tua atividade frontal inclinou-se para o lado que a investigação
> associa a aproximação e a afeto positivo.

Texto a **não** usar: "o teu controlo emocional aumentou X %". É uma afirmação de
capacidade, e a assimetria alfa frontal não a sustenta numa sessão de quatro minutos: o
efeito agregado para stress é nulo (g = 0,01) e num longitudinal de um ano a assimetria
lateral não mudou em nenhuma condição enquanto a coerência frontal mudou. Ver
`BIOMARKERS.md` B6 e referências [3], [14], [23].

---

## Grupo 5 — Outras medidas a serem excluídas do teu sinal (descritivas)

Apresentadas **sem valência**. O texto diz "mudou" ou "manteve-se", nunca "melhorou"  não têmuma direcao definida, podem sair e ser excluidoas da aplicaoe e da analise.

| `id` | `label_public` | Explicação |
|---|---|---|
| `ap_exponent` | Perfil global da atividade | A inclinação geral do espectro. Muda entre estados de consciência, mas a investigação ainda não estabeleceu qual a direção desejável. |
| `lziv` | Variedade da atividade | O quanto o sinal é variado em vez de repetitivo. Também sem direção estabelecida. |
| `spectral_entropy` | Dispersão espectral | O quanto a energia está espalhada por várias frequências em vez de concentrada numa. |
| `alpha_peak_power` | Alfa oscilatório puro | A parte das ondas de calma que é oscilação verdadeira, separada do fundo do sinal. |
| `alpha_peak_freq` | Velocidade do ritmo alfa | A frequência exata a que o teu ritmo de calma oscila. |

Cabeçalho da secção, para a pessoa:

> Estas três medidas descrevem o teu sinal mas a ciência ainda não sabe qual o lado
> desejável. Mostramo-las porque mudam, não porque saibamos interpretá-las a teu favor.

Esta frase é o que separa uma demonstração científica de uma leitura de aura. Vale a pena
mantê-la exatamente assim.

Se `alpha_peak_power` e `alpha_rel` divergirem numa sessão, a subida do alfa relativo era
aperiódica e não oscilatória. Mostrar as duas lado a lado no painel de espectros.

---