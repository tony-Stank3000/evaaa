param(
    [string]$WorldPath
)

$ErrorActionPreference = "Stop"
$modSource = Join-Path $PSScriptRoot "mods\evaaa_luanti"

if (-not (Test-Path (Join-Path $modSource "init.lua")) -or
    -not (Test-Path (Join-Path $modSource "mod.conf"))) {
    throw "EVAAA mod files were not found at $modSource. Run this script from your cloned repository."
}

if ([string]::IsNullOrWhiteSpace($WorldPath)) {
    $candidateRoots = @(
        (Join-Path $env:APPDATA "Luanti\worlds"),
        (Join-Path $env:APPDATA "Minetest\worlds"),
        (Join-Path $env:USERPROFILE ".luanti\worlds"),
        (Join-Path $env:USERPROFILE ".minetest\worlds")
    ) | Select-Object -Unique

    $worlds = @()
    foreach ($root in $candidateRoots) {
        if (Test-Path $root) {
            $worlds += Get-ChildItem -LiteralPath $root -Directory |
                Where-Object { Test-Path (Join-Path $_.FullName "world.mt") }
        }
    }
    $worlds = @($worlds | Sort-Object FullName -Unique)

    if ($worlds.Count -eq 0) {
        throw "No worlds were found in common Luanti data folders. Open Luanti, use its Open User Data Directory option, then rerun with -WorldPath 'FULL\PATH\TO\WORLD'."
    }

    Write-Host "Detected Luanti worlds:"
    for ($i = 0; $i -lt $worlds.Count; $i++) {
        Write-Host ("[{0}] {1}" -f ($i + 1), $worlds[$i].FullName)
    }
    $selection = Read-Host "Enter the number for your Mineclonia world"
    $index = 0
    if (-not [int]::TryParse($selection, [ref]$index) -or
        $index -lt 1 -or $index -gt $worlds.Count) {
        throw "Invalid world selection."
    }
    $WorldPath = $worlds[$index - 1].FullName
}

$WorldPath = (Resolve-Path -LiteralPath $WorldPath).Path
if (-not (Test-Path (Join-Path $WorldPath "world.mt"))) {
    throw "WorldPath must point to a Luanti world folder containing world.mt: $WorldPath"
}

$worldMods = Join-Path $WorldPath "worldmods"
$modDestination = Join-Path $worldMods "evaaa_luanti"
New-Item -ItemType Directory -Path $worldMods -Force | Out-Null
if (Test-Path $modDestination) {
    $backup = "$modDestination.backup-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    Copy-Item -LiteralPath $modDestination -Destination $backup -Recurse
    Write-Host "Backed up existing mod to: $backup"
}
New-Item -ItemType Directory -Path $modDestination -Force | Out-Null
Copy-Item -Path (Join-Path $modSource "*") -Destination $modDestination -Recurse -Force
Write-Host "Installed EVAAA Luanti mod to: $modDestination"

$dataRoot = Split-Path -Parent (Split-Path -Parent $WorldPath)
$configPath = Join-Path $dataRoot "minetest.conf"
if (Test-Path $configPath) {
    $configText = Get-Content -LiteralPath $configPath -Raw
    $match = [regex]::Match($configText, '(?m)^\s*secure\.http_mods\s*=\s*(.*?)\s*$')
    if ($match.Success) {
        $existing = @($match.Groups[1].Value -split '[,;\s]+' | Where-Object { $_ })
        if ($existing -notcontains "evaaa_luanti") {
            $existing += "evaaa_luanti"
            $updatedLine = "secure.http_mods = " + ($existing -join ",")
            $configText = [regex]::Replace(
                $configText,
                '(?m)^\s*secure\.http_mods\s*=\s*.*?$',
                [System.Text.RegularExpressions.MatchEvaluator]{ param($m) $updatedLine },
                1
            )
        }
    } else {
        if ($configText.Length -gt 0 -and -not $configText.EndsWith("`n")) { $configText += "`r`n" }
        $configText += "secure.http_mods = evaaa_luanti`r`n"
    }
    $configBackup = "$configPath.backup-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    Copy-Item -LiteralPath $configPath -Destination $configBackup
    Set-Content -LiteralPath $configPath -Value $configText -Encoding UTF8
    Write-Host "Updated mod HTTP allowlist in $configPath (backup: $configBackup)"
} else {
    Write-Warning "Could not find minetest.conf at $configPath. Add this line to the Luanti configuration manually: secure.http_mods = evaaa_luanti"
}

Write-Host ""
Write-Host "Installation complete. Restart Luanti and open the selected world."
Write-Host "The Python trainer's local bridge listens at 127.0.0.1:8766; the mod cannot work until a trainer starts it."
Write-Host "Do not disable Luanti's broader security settings."
