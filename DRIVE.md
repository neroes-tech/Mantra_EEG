# Envio para o Google Drive

## O que é preciso, uma vez

**1. Um cliente OAuth na Google Cloud Console** (conta Neroes)

1. <https://console.cloud.google.com> → criar projeto (ou usar um existente)
2. **APIs e serviços → Biblioteca** → activar **Google Drive API**
3. **APIs e serviços → Ecrã de consentimento OAuth**
   - Tipo: **Interno** (só contas @neroes.tech) — evita a revisão da Google
   - Nome da app, email de suporte, email de contacto
4. **APIs e serviços → Credenciais → Criar credenciais → ID de cliente OAuth**
   - Tipo: **Aplicação de ambiente de trabalho**
   - Copiar o **ID de cliente** e o **segredo**

O "segredo" de uma aplicação de ambiente de trabalho não é realmente secreto —
a própria Google o documenta assim. Não há problema em ficar na config.

**2. O id da pasta partilhada**

Do link de partilha, a parte a seguir a `/folders/`:

```
https://drive.google.com/drive/folders/1AbC-dEfGhIJKlmno?usp=sharing
                                        ^^^^^^^^^^^^^^^^
```

**3. Preencher a config**

```yaml
drive:
  enabled: true
  client_id: "....apps.googleusercontent.com"
  client_secret: "GOCSPX-..."
  folder_id: "1AbC-dEfGhIJKlmno"
```

## Em cada computador, uma vez

Abrir a app → **Ctrl+I** → **Iniciar sessão Google** → escolher a conta Neroes.

O browser abre, a pessoa autoriza, e o *refresh token* fica guardado em
`data/drive_token.json`. **Nas vezes seguintes não há ecrã de login**, mesmo
depois de reiniciar a máquina.

## O que acontece a cada sessão

1. A sessão termina e é gravada em
   `data/sessions/FBE_BCI_mantra/<id>_<HHMMSSDDMMAAAA>/`
2. É comprimida para `.zip` (medido: 2,4x — 5,1 MB passam a 2,1 MB)
3. Entra na fila e é enviada em fundo, sem tocar na interface
4. **Só depois de o Drive confirmar**, a cópia local é apagada

Sem rede, a sessão fica em fila. A fila é lida do disco ao arrancar, portanto
fechar a aplicação não perde nada. O painel Ctrl+I mostra quantas faltam e
quantos MB, e tem um botão para repetir.

Pior caso, um dia inteiro sem rede: 100 sessões = **210 MB** em fila.

## Se o envio for recusado com 403

O âmbito por defeito é `drive.file` — só ficheiros criados pela aplicação. Se a
Google recusar a escrita numa pasta pré-existente, trocar para acesso completo:

```yaml
  scope: "https://www.googleapis.com/auth/drive"
```

e voltar a iniciar sessão (apagar `data/drive_token.json` primeiro).

## Alternativa sem login: conta de serviço

Para quiosques desatendidos é mais robusto — não há login a expirar a meio do
evento nem browser a abrir por cima da demo. Criar uma conta de serviço,
partilhar a pasta com o email dela, e distribuir a chave JSON.

Está previsto na configuração mas **ainda não implementado**: precisa de
assinar um JWT, o que traz a dependência `cryptography`. Diga se prefere este
caminho.
