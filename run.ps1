param(
    [switch]$Evaluate,
    [switch]$BusinessRegression,
    [string]$PythonPath
)

$localPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if ($PythonPath) {
    $pythonExe = $PythonPath
} elseif (Test-Path -LiteralPath $localPython) {
    $pythonExe = $localPython
} else {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        throw "Python was not found. Create .venv in this directory or pass -PythonPath."
    }
    $pythonExe = $pythonCommand.Source
}

if ($Evaluate -and $BusinessRegression) {
    throw "Choose either -Evaluate or -BusinessRegression, not both."
}
$entryPoint = if ($BusinessRegression) {
    "evaluate_business_regression.py"
} elseif ($Evaluate) {
    "evaluate.py"
} else {
    "main.py"
}
$entryPath = Join-Path $PSScriptRoot $entryPoint
& $pythonExe $entryPath
$processExitCode = $LASTEXITCODE

exit $processExitCode
