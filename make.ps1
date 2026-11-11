<#
.SYNOPSIS
    NBA Quant Lab task runner for PowerShell: the Makefile's targets for Windows (PowerShell 5.1 or 7).

.DESCRIPTION
    Usage:  .\make.ps1 <target> [arguments]

      install [python]  create .venv and install the package (64-bit Python 3.10 - 3.13; 3.13 recommended)
      data              download (pinned, SHA-256 verified) + reconstruct 6 seasons + database   (~3 min)
      models            features, team model, minutes, rates, opponent factors, simulator constants (~5 min)
      backtest          simulation backtests: 2023-24 -> blend weight -> blend; 2024-26 test    (~1.5 h, 4 cores)
      report            baselines, gradient boosting, matchup ablation, team-first model, summary (~15 min)
      docs              re-render the README results, docs/REPORT.md and docs/report.html
      test              run the test suite
      dashboard         start the dashboard at http://localhost:8501 (Ctrl+C to stop)
      all               data, models, backtest, report
      quantlab <args>   run any quantlab command, e.g.  .\make.ps1 quantlab project --game-id 22501132
      package           git bundle + source zip in dist\
      clean-data        delete the downloaded and generated data (asks first)

    If Windows refuses to run scripts, allow them for this window first:
        Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

.EXAMPLE
    .\make.ps1 install
    .\make.ps1 all
    .\make.ps1 dashboard
#>
param([string]$Target = "help")

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$OnWindows = [System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT

# Arguments after the target are passed on. PowerShell reads an unquoted comma list (--season 2024-25,2025-26) as
# an array; join it back so the program receives the list as typed.
$Rest = @(foreach ($a in $args) { if ($a -is [System.Array]) { ($a | ForEach-Object { "$_" }) -join "," } else { "$a" } })

function Write-Step([string]$Text) {
    Write-Host ""
    Write-Host "== $Text" -ForegroundColor Cyan
}

function Invoke-Native {
    # Run an external program; stop if it fails. Native stderr output (pip, git and Python warnings) is not an
    # error by itself, so the exit code decides.
    param([string]$Exe, [string[]]$Arguments)
    if (-not (Get-Command $Exe -ErrorAction SilentlyContinue)) {
        throw "'$Exe' was not found. Is it installed and on PATH?"
    }
    Write-Host ("> " + $Exe + " " + ($Arguments -join " ")) -ForegroundColor DarkGray
    $eap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Exe @Arguments
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $eap
    }
    if ($code -ne 0) { throw "'$Exe $($Arguments -join ' ')' failed with exit code $code" }
}

function Get-VenvPython {
    foreach ($p in @((Join-Path $Root ".venv\Scripts\python.exe"), (Join-Path $Root ".venv/bin/python"))) {
        if (Test-Path -LiteralPath $p -PathType Leaf) { return $p }
    }
    throw "No virtual environment in $Root\.venv - run:  .\make.ps1 install"
}

function Invoke-Quantlab {
    # quantlab through "python -m" (the safest way to start worker processes on Windows)
    param([string[]]$Arguments, [switch]$OneThread)
    $py = Get-VenvPython
    $old = $env:QUANTLAB_THREADS
    if ($OneThread) { $env:QUANTLAB_THREADS = "1" }   # as in the Makefile: LightGBM single-threaded for these steps
    try {
        Invoke-Native $py (@("-m", "quantlab.cli") + $Arguments)
    } finally {
        $env:QUANTLAB_THREADS = $old
    }
}

function Get-PythonInfo([string]$Exe, [string[]]$Prefix) {
    # @{Version; Platform} of an interpreter, or $null if it does not run (e.g. the Microsoft Store placeholder)
    $eap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $out = & $Exe @Prefix -c "import sys, sysconfig; print('%d.%d %s' % (sys.version_info[0], sys.version_info[1], sysconfig.get_platform()))" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $out) { return $null }
        $parts = "$out".Trim() -split "\s+"
        return @{ Version = [version]$parts[0]; Platform = $parts[1] }
    } catch {
        return $null
    } finally {
        $ErrorActionPreference = $eap
    }
}

function Find-BasePython([string]$Requested) {
    # A supported interpreter as @(exe, args...): the one asked for, else the py launcher's 3.13 .. 3.10, else
    # python / python3 on PATH (Microsoft Store aliases last; its install placeholder simply fails the check).
    $candidates = @()
    if ($Requested) {
        $candidates += , @($Requested)
    } else {
        if (Get-Command py -ErrorAction SilentlyContinue) {
            foreach ($minor in 13, 12, 11, 10) { $candidates += , @("py", "-3.$minor") }
        }
        $store = @()
        foreach ($name in "python", "python3") {
            $cmd = Get-Command $name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
            if (-not $cmd) { continue }
            if ($cmd.Source -like "*\WindowsApps\*") { $store += , @($cmd.Source) } else { $candidates += , @($cmd.Source) }
        }
        $candidates += $store
    }
    foreach ($c in $candidates) {
        $exe = $c[0]
        $prefix = @($c | Select-Object -Skip 1)
        $info = Get-PythonInfo $exe $prefix
        if ($null -eq $info) { continue }
        $label = "$exe $($prefix -join ' ')".Trim()
        if ($info.Version -lt [version]"3.10" -or $info.Version -ge [version]"3.14") {
            Write-Warning "Skipping Python $($info.Version) ($label): this project needs Python 3.10 - 3.13 (numpy < 2.3 has no wheels for newer versions)."
            continue
        }
        if ($OnWindows -and $info.Platform -ne "win-amd64") {
            Write-Warning "Skipping $label ($($info.Platform)): the dependencies need the 64-bit (x64) Python from python.org."
            continue
        }
        return , (@($exe) + $prefix)
    }
    throw ("No suitable Python found. Install Python 3.13 (Windows installer, 64-bit) from https://www.python.org/downloads/ " +
           "(tick 'Add python.exe to PATH'), or pass one:  .\make.ps1 install C:\path\to\python.exe")
}

function Test-VcRuntime {
    # numba and LightGBM need the Microsoft Visual C++ 2015-2022 runtime (msvcp140.dll, vcomp140.dll).
    if (-not $OnWindows) { return }
    $missing = @("msvcp140.dll", "vcomp140.dll") | Where-Object {
        -not (Test-Path -LiteralPath (Join-Path $env:SystemRoot "System32\$_"))
    }
    if ($missing) {
        Write-Warning ("Missing $($missing -join ', '): install the Microsoft Visual C++ Redistributable (x64) from " +
                       "https://aka.ms/vs/17/release/vc_redist.x64.exe (or: winget install Microsoft.VCRedist.2015+.x64).")
    }
}

function Show-Help {
    # the DESCRIPTION section of the comment block above (read as text: works in any folder name)
    $text = Get-Content -LiteralPath $PSCommandPath -Raw
    $start = $text.IndexOf(".DESCRIPTION")
    $end = $text.IndexOf(".EXAMPLE")
    if ($start -ge 0 -and $end -gt $start) {
        Write-Host $text.Substring($start + 12, $end - $start - 12).TrimEnd()
    }
}

function Invoke-Target([string]$Name) {
    switch ($Name) {
        "help" { Show-Help }

        "install" {
            Write-Step "Creating .venv and installing NBA Quant Lab"
            Test-VcRuntime
            $base = Find-BasePython ($Rest | Select-Object -First 1)
            $exe = $base[0]
            $prefix = @($base | Select-Object -Skip 1)
            Invoke-Native $exe ($prefix + @("-m", "venv", ".venv"))
            $py = Get-VenvPython
            Invoke-Native $py @("-m", "pip", "install", "--upgrade", "pip")
            Invoke-Native $py @("-m", "pip", "install", "-e", ".[dev]")
            Write-Step "Checking that the compiled libraries load"
            Invoke-Native $py @("-c", "import quantlab, numba, lightgbm; print('numba', numba.__version__, '/ lightgbm', lightgbm.__version__, ': OK')")
            Write-Host "Installed. Next:  .\make.ps1 all   (then .\make.ps1 dashboard)" -ForegroundColor Green
        }

        "data" {
            Write-Step "Data: download, reconstruct, database"
            foreach ($cmd in "fetch", "build", "database") { Invoke-Quantlab @($cmd) }
        }

        "models" {
            Write-Step "Models: features, team, minutes, rates, opponent factors, simulator constants"
            foreach ($cmd in "features", "team", "minutes", "rates", "opponent", "constants") { Invoke-Quantlab @($cmd) }
        }

        "backtest" {
            Write-Step "Backtests: 2023-24 (tuning) -> blend weight -> blend; 2024-25 and 2025-26 (test)"
            Invoke-Quantlab @("backtest", "--season", "2023-24", "--modes", "none,team") -OneThread
            Invoke-Quantlab @("blendweight")
            Invoke-Quantlab @("backtest", "--season", "2023-24", "--modes", "blend") -OneThread
            Invoke-Quantlab @("backtest", "--season", "2024-25,2025-26", "--modes", "none,team,blend") -OneThread
        }

        "report" {
            Write-Step "Report: baselines, gradient boosting, ablation, team-first model, summary"
            Invoke-Quantlab @("baselines") -OneThread
            Invoke-Quantlab @("ablation") -OneThread
            Invoke-Quantlab @("topdown")
            Invoke-Quantlab @("report")
        }

        "docs" {
            Write-Step "Docs: check every claim, then render"
            $py = Get-VenvPython
            Invoke-Native $py @("scripts/docs/render_report.py", "--check")
            Invoke-Native $py @("scripts/docs/render_docs.py", "--check")
            Invoke-Native $py @("scripts/docs/render_report.py")
            Invoke-Native $py @("scripts/docs/render_docs.py")
            Invoke-Native $py @("scripts/make_report_page.py")
        }

        "test" {
            $py = Get-VenvPython
            Invoke-Native $py (@("-m", "pytest") + $Rest)
        }

        "dashboard" {
            Write-Step "Dashboard: open http://localhost:8501 in your browser (Ctrl+C to stop)"
            Invoke-Quantlab (@("dashboard") + $Rest)
        }

        "all" {
            foreach ($t in "data", "models", "backtest", "report") { Invoke-Target $t }
            Write-Host "Pipeline complete. Start the dashboard with:  .\make.ps1 dashboard" -ForegroundColor Green
        }

        "quantlab" {
            $qlArgs = $Rest
            if ($qlArgs.Count -eq 0) { $qlArgs = @("--help") }
            Invoke-Quantlab $qlArgs
        }

        "package" {
            Write-Step "Package: git bundle (full history) + source zip (HEAD, no data) in dist\"
            New-Item -ItemType Directory -Force -Path "dist" | Out-Null
            Invoke-Native "git" @("bundle", "create", "dist/nba-quant-lab.bundle", "--all")
            Invoke-Native "git" @("archive", "--format=zip", "--prefix=nba-quant-lab/", "-o", "dist/nba-quant-lab-src.zip", "HEAD")
            Invoke-Native "git" @("bundle", "verify", "dist/nba-quant-lab.bundle")
            Get-ChildItem "dist" | Format-Table Name, Length -AutoSize | Out-String | Write-Host
        }

        "clean-data" {
            $dirs = @("raw", "staged", "processed", "features", "models", "backtest") | ForEach-Object { Join-Path "data" $_ }
            $answer = Read-Host "Delete $($dirs -join ', ')? Type yes to confirm"
            if ($answer -ne "yes") {
                Write-Host "Nothing deleted."
            } else {
                foreach ($d in $dirs) {
                    if (Test-Path -LiteralPath $d) { Remove-Item -LiteralPath $d -Recurse -Force }
                }
                Write-Host "Deleted. Rebuild with:  .\make.ps1 all"
            }
        }

        default {
            Write-Host "Unknown target '$Name'." -ForegroundColor Red
            Show-Help
            throw "Unknown target '$Name' (see the list above)"
        }
    }
}

# Python reads and writes UTF-8 whatever the Windows code page, and the console shows its output as UTF-8; all three
# settings are restored when the script ends, so the window is left as it was.
$saved = @{ Utf8 = $env:PYTHONUTF8; IoEnc = $env:PYTHONIOENCODING; Console = $null }
$pushed = $false
try {
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    try {
        $saved.Console = [Console]::OutputEncoding
        [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false
    } catch {
        $saved.Console = $null      # hosts without a console (e.g. the ISE) keep their encoding
    }
    Push-Location -LiteralPath $Root
    $pushed = $true
    Invoke-Target $Target
} catch {
    Write-Host ""
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} finally {
    if ($pushed) { Pop-Location }
    $env:PYTHONUTF8 = $saved.Utf8
    $env:PYTHONIOENCODING = $saved.IoEnc
    if ($null -ne $saved.Console) {
        try { [Console]::OutputEncoding = $saved.Console } catch { }
    }
}
