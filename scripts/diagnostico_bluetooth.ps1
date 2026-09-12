# Diagnostico de Bluetooth para o Unicorn, sem instalar nada.
#
# Correr na maquina onde o headset nao e reconhecido:
#
#     powershell -ExecutionPolicy Bypass -File diagnostico_bluetooth.ps1
#
# ou abrir o PowerShell na pasta e colar o conteudo.
#
# O que procura, por ordem de probabilidade:
#   1. Unicorn Suite a segurar o dispositivo (so um processo de cada vez)
#   2. porta serie sobre Bluetooth em falta: o Unicorn fala por RFCOMM, e
#      sem uma porta COM ativa nao ha por onde falar, mesmo emparelhado
#   3. emparelhado pelo Bluetooth interno em vez do dongle da g.tec
#   4. headset desligado (continua a aparecer emparelhado)

$ErrorActionPreference = "SilentlyContinue"
$problemas = @()

Write-Output ""
Write-Output "=============================================================="
Write-Output " DIAGNOSTICO BLUETOOTH - UNICORN"
Write-Output " $(Get-Date -Format 'yyyy-MM-dd HH:mm')   $env:COMPUTERNAME"
Write-Output "=============================================================="

# --------------------------------------------------------------------------- #
Write-Output ""
Write-Output "1. PROCESSOS QUE PODEM TER O DISPOSITIVO ABERTO"
$donos = Get-Process | Where-Object { $_.ProcessName -match 'Unicorn|MantraEEG' }
if ($donos) {
    $donos | Select-Object Id, ProcessName | Format-Table -AutoSize | Out-String | Write-Output
    $problemas += "FECHAR: $($donos.ProcessName -join ', ') esta a correr. O Unicorn so aceita UM processo de cada vez: fechar pela bandeja do sistema, nao so a janela."
} else {
    Write-Output "   nenhum. bom."
}

# --------------------------------------------------------------------------- #
Write-Output ""
Write-Output "2. HEADSET EMPARELHADO"
$headsets = Get-PnpDevice | Where-Object { $_.FriendlyName -like 'UN-*' }
if ($headsets) {
    $headsets | Select-Object Status, FriendlyName | Format-Table -AutoSize | Out-String | Write-Output
    if (-not ($headsets | Where-Object { $_.Status -eq 'OK' })) {
        $problemas += "O headset esta emparelhado mas nao esta OK. Desligar e voltar a ligar o headset (LED azul a piscar), e se preciso remover o emparelhamento e emparelhar de novo (PIN 0000)."
    }
} else {
    Write-Output "   NENHUM dispositivo UN-* encontrado."
    $problemas += "O headset nao esta emparelhado nesta maquina. Definicoes > Bluetooth > Adicionar dispositivo, PIN 0000, com o dongle da g.tec espetado."
}

# --------------------------------------------------------------------------- #
Write-Output ""
Write-Output "3. PORTAS SERIE SOBRE BLUETOOTH   (a causa mais comum)"
$portas = Get-PnpDevice -Class Ports | Where-Object { $_.FriendlyName -match 'Bluetooth|COM' }
if ($portas) {
    $portas | Select-Object Status, FriendlyName | Format-Table -AutoSize | Out-String | Write-Output
    $ativas = @($portas | Where-Object { $_.Status -eq 'OK' })
    Write-Output "   portas ativas: $($ativas.Count)"
    if ($ativas.Count -eq 0) {
        $problemas += "Nenhuma porta serie sobre Bluetooth ativa. O Unicorn fala por RFCOMM: sem porta COM nao ha por onde falar, mesmo estando emparelhado. Remover o emparelhamento e emparelhar de novo: e ao emparelhar aceitar o servico de porta serie."
    }
} else {
    Write-Output "   NENHUMA porta serie sobre Bluetooth."
    $problemas += "Nao ha portas serie sobre Bluetooth. O emparelhamento nao criou o servico RFCOMM que o Unicorn precisa. Remover o emparelhamento e repetir."
}

# --------------------------------------------------------------------------- #
Write-Output ""
Write-Output "4. ADAPTADORES BLUETOOTH"
$radios = Get-PnpDevice -Class Bluetooth | Where-Object {
    $_.FriendlyName -match 'Radio|Wireless|Adapter|CSR|Generic'
}
$radios | Select-Object Status, FriendlyName | Format-Table -AutoSize | Out-String | Write-Output
$ativos = @($radios | Where-Object { $_.Status -eq 'OK' })
if ($ativos.Count -gt 1) {
    $problemas += "Ha mais do que um adaptador Bluetooth ativo ($($ativos.Count)). O Windows pode ter emparelhado pelo interno em vez do dongle da g.tec, e o Unicorn so e homologado com o dongle. Desativar o adaptador interno no Gestor de Dispositivos e emparelhar de novo."
}
if ($ativos.Count -eq 0) {
    $problemas += "Nenhum adaptador Bluetooth ativo. Espetar o dongle da g.tec e esperar que o Windows instale o controlador."
}

# --------------------------------------------------------------------------- #
Write-Output ""
Write-Output "5. BIBLIOTECA DO UNICORN"
$dlls = @(
    "$env:ProgramFiles\gtec\Unicorn Suite\Hybrid Black\Unicorn.dll",
    ".\_internal\brainflow\lib\Unicorn.dll",
    "$PSScriptRoot\_internal\brainflow\lib\Unicorn.dll"
)
$achou = $false
foreach ($p in $dlls) {
    if (Test-Path $p) { Write-Output "   existe:   $p"; $achou = $true }
    else { Write-Output "   em falta: $p" }
}
if (-not $achou) {
    Write-Output "   (nenhuma encontrada aqui: normal se este script nao estiver na pasta MantraEEG)"
}

# --------------------------------------------------------------------------- #
Write-Output ""
Write-Output "=============================================================="
if ($problemas.Count -eq 0) {
    Write-Output " Nada de errado encontrado no Bluetooth."
    Write-Output ""
    Write-Output " Se mesmo assim a aplicacao nao reconhece:"
    Write-Output "   - desligar e voltar a ligar o headset (LED azul a piscar)"
    Write-Output "   - correr Diagnostico.exe --selftest na pasta MantraEEG"
} else {
    Write-Output " $($problemas.Count) coisa(s) a resolver, por ordem:"
    Write-Output ""
    $i = 1
    foreach ($p in $problemas) {
        Write-Output "  $i. $p"
        Write-Output ""
        $i++
    }
}
Write-Output "=============================================================="
