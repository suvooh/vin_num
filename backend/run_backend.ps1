$ErrorActionPreference = "Stop"

$backendRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $backendRoot

$python = Join-Path $backendRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

$port = if ($env:VIN_NUM_API_PORT) { $env:VIN_NUM_API_PORT } else { "8002" }
Write-Host "Starting the VIN & number plate API (and UI) on http://127.0.0.1:$port ..."
& $python -m app.main
