$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$FrontendDir = Join-Path $Root "frontend"
$DefaultPythonExe = Join-Path $env:USERPROFILE ".conda\envs\BUCAD\python.exe"
$PythonExe = if ($env:BUCAD_PYTHON_EXE) { $env:BUCAD_PYTHON_EXE } else { $DefaultPythonExe }
$BackendPort = 8000
$FrontendPort = 5173
$FrontendUrl = "http://127.0.0.1:$FrontendPort"

function Import-DotEnvFile {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    foreach ($line in Get-Content -LiteralPath $Path) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#") -or -not $trimmed.Contains("=")) {
            continue
        }

        $parts = $trimmed.Split("=", 2)
        $name = $parts[0].Trim()
        $value = $parts[1].Trim()
        if ($value.StartsWith('"') -and $value.EndsWith('"')) {
            $value = $value.Substring(1, $value.Length - 2)
        } elseif ($value.StartsWith("'") -and $value.EndsWith("'")) {
            $value = $value.Substring(1, $value.Length - 2)
        }

        if ($name -and -not (Test-Path "Env:$name")) {
            Set-Item -Path "Env:$name" -Value $value
        }
    }
}

function Test-CommandExists {
    param([Parameter(Mandatory = $true)][string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Test-PortListening {
    param([Parameter(Mandatory = $true)][int]$Port)
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

function ConvertTo-SingleQuotedPowerShell {
    param([Parameter(Mandatory = $true)][string]$Value)
    return "'" + $Value.Replace("'", "''") + "'"
}

function Start-DevWindow {
    param(
        [Parameter(Mandatory = $true)][string]$Title,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [Parameter(Mandatory = $true)][string]$Command
    )

    $safeTitle = ConvertTo-SingleQuotedPowerShell $Title
    $safeWorkingDirectory = ConvertTo-SingleQuotedPowerShell $WorkingDirectory
    $windowCommand = "`$Host.UI.RawUI.WindowTitle = $safeTitle; Set-Location -LiteralPath $safeWorkingDirectory; $Command"

    Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        $windowCommand
    )
}

if (-not (Test-Path -LiteralPath $FrontendDir)) {
    throw "Frontend directory not found: $FrontendDir"
}

Import-DotEnvFile -Path (Join-Path $Root ".env")
Import-DotEnvFile -Path (Join-Path $FrontendDir ".env")

if (-not (Test-CommandExists "npm")) {
    throw "npm was not found in PATH."
}

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "BUCAD Python not found: $PythonExe"
}

if (Test-PortListening $BackendPort) {
    Write-Host "Backend already running on port $BackendPort."
} else {
    $safePythonExe = ConvertTo-SingleQuotedPowerShell $PythonExe
    Start-DevWindow `
        -Title "BUCAD Backend API" `
        -WorkingDirectory $Root `
        -Command "& $safePythonExe -m backend.server --host 127.0.0.1 --port $BackendPort"
    Write-Host "Started backend on port $BackendPort."
}

if (Test-PortListening $FrontendPort) {
    Write-Host "Frontend already running on port $FrontendPort."
} else {
    Start-DevWindow `
        -Title "BUCAD Frontend" `
        -WorkingDirectory $FrontendDir `
        -Command "if (-not (Test-Path -LiteralPath 'node_modules')) { npm install }; npm run dev -- --host 127.0.0.1 --port $FrontendPort"
    Write-Host "Started frontend on port $FrontendPort."
}

Start-Sleep -Seconds 3
Start-Process $FrontendUrl
Write-Host "Opened $FrontendUrl"
