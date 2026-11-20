# Run the app on an Android phone over USB or wireless debugging.
#
#   .\run_on_device.ps1                         # the only connected device
#   .\run_on_device.ps1 -Serial 192.168.1.3:40123
#   .\run_on_device.ps1 -Pair 192.168.1.3:39031 -Code 737460 -Connect 192.168.1.3:40123
#
# The app is completely offline: the models and the ONNX runtime are inside the APK and there is no
# server, so no tunnel or backend is needed. adb is used only to install and start it.
#
# Wireless debugging uses TWO ports: the one in the "Pair device with pairing code" dialog (used
# once, with the code) and a different "IP address & Port" at the top of the Wireless debugging
# screen (used to connect). Keep the phone awake and charging - the link drops when it sleeps.
param(
  [string]$Serial,
  [string]$Pair,
  [string]$Code,
  [string]$Connect,
  [switch]$Release
)
$ErrorActionPreference = "Stop"
$sdk = if ($env:ANDROID_HOME) { $env:ANDROID_HOME } else { "$env:LOCALAPPDATA\Android\Sdk" }
$adb = Join-Path $sdk "platform-tools\adb.exe"
if (-not (Test-Path $adb)) { throw "adb not found at $adb (set ANDROID_HOME)" }

if ($Pair -and $Code) { & $adb pair $Pair $Code }
if ($Connect) { & $adb connect $Connect; if (-not $Serial) { $Serial = $Connect } }

$lines = & $adb devices | Select-Object -Skip 1 | Where-Object { $_ -match "\sdevice$" }
if (-not $lines) { throw "No device attached. Check the cable / the phone's Wireless debugging screen, then try -Connect <ip:port>." }
if (-not $Serial) {
  if (@($lines).Count -gt 1) { throw "More than one device; pass -Serial. Attached:`n$($lines -join "`n")" }
  $Serial = ($lines[0] -split "\s+")[0]
}
Write-Host "Device: $Serial"

Set-Location $PSScriptRoot
$mode = if ($Release) { "--release" } else { "--debug" }
flutter run -d $Serial $mode
