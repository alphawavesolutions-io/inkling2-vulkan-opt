param(
    [string]$RuntimeDir = (Join-Path $PSScriptRoot 'llama-b12628-inkling-opt'),
    [string]$ModelPath  = (Join-Path $PSScriptRoot 'models\Inkling-2-Mini-Uncensored-AD-Q4_K.gguf'),
    [string]$MmprojPath = (Join-Path $PSScriptRoot 'models\mmproj-Inkling-2-Mini-Uncensored-F16.gguf'),
    [string]$Alias      = 'inkling-2-mini-uncensored',
    [int]$Port          = 18089,
    [int]$CtxPerSlot    = 65536,
    [int]$Parallel      = 2,
    [int]$IdleSleepSec  = 600
)

$ErrorActionPreference = 'Stop'
$runtime = Join-Path $RuntimeDir 'llama-server.exe'
$endpoint = "http://127.0.0.1:$Port"
$ctxTotal = $CtxPerSlot * $Parallel

$listener = Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    $running = Get-Process -Id $listener.OwningProcess -ErrorAction Stop
    $metadata = Invoke-RestMethod "$endpoint/props" -TimeoutSec 5
    if ($running.Path -ieq $runtime -and $metadata.model_path -ieq $ModelPath -and $metadata.total_slots -eq $Parallel) { return }
    throw "Port $Port is already in use."
}

foreach ($required in @($runtime, $ModelPath)) {
    if (!(Test-Path -LiteralPath $required -PathType Leaf)) { throw "Missing $required" }
}

$env:LLAMA_ARG_CHAT_TEMPLATE_KWARGS = '{"reasoning_effort":"low"}'
$arguments = @(
    '--model', ('"{0}"' -f $ModelPath),
    '--alias', $Alias, '--host', '127.0.0.1', '--port', $Port,
    '--ctx-size', $ctxTotal, '--parallel', $Parallel, '--no-kv-unified',
    '--batch-size', '512', '--ubatch-size', '256',
    '--device', 'Vulkan0', '--n-gpu-layers', 'all',
    '--jinja', '--cache-ram', '0',
    '-ctk', 'f16', '-ctv', 'f16',
    '--spec-type', 'none', '--sleep-idle-seconds', $IdleSleepSec, '--no-webui'
)
if ($MmprojPath -and (Test-Path -LiteralPath $MmprojPath -PathType Leaf)) {
    $arguments += @('--mmproj', ('"{0}"' -f $MmprojPath), '--no-mmproj-offload')
}

$process = Start-Process -FilePath $runtime -ArgumentList $arguments -WindowStyle Hidden -PassThru `
    -WorkingDirectory $PSScriptRoot `
    -RedirectStandardOutput (Join-Path $PSScriptRoot 'server.out.log') `
    -RedirectStandardError  (Join-Path $PSScriptRoot 'server.err.log')

$ready = $false
try {
    $deadline = (Get-Date).AddMinutes(3)
    while (!$process.HasExited -and (Get-Date) -lt $deadline) {
        try { $ready = (Invoke-RestMethod "$endpoint/health" -TimeoutSec 4).status -eq 'ok' } catch {}
        if ($ready) { return }
        Start-Sleep -Seconds 2
    }
    throw "Inkling did not become ready. See $(Join-Path $PSScriptRoot 'server.err.log')"
} finally {
    if (!$ready -and !$process.HasExited) {
        Stop-Process -Id $process.Id
        if (!$process.WaitForExit(45000)) { throw "The failed Inkling process is still shutting down." }
    }
}
