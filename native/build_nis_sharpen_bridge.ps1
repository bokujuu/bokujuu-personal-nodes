param(
    [string]$NisPath = $env:NVIDIA_NIS_SDK
)

$ErrorActionPreference = "Stop"

if (-not $NisPath) {
    throw "Specify -NisPath or set NVIDIA_NIS_SDK to an NVIDIA Image Scaling SDK checkout."
}

$nisRoot = (Resolve-Path -LiteralPath $NisPath).Path
$includePath = Join-Path $nisRoot "NIS"
$required = @("NIS_Config.h", "NIS_Main.hlsl", "NIS_Scaler.h")
foreach ($name in $required) {
    if (-not (Test-Path -LiteralPath (Join-Path $includePath $name))) {
        throw "Missing NVIDIA Image Scaling SDK file: $(Join-Path $includePath $name)"
    }
}

$nativeRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$sourcePath = Join-Path $nativeRoot "nis_sharpen_bridge.cpp"
$outputDirectory = Join-Path $nativeRoot "bin"
$outputPath = Join-Path $outputDirectory "bokujuu_nis_sharpen_bridge.dll"
$objectPath = Join-Path $outputDirectory "bokujuu_nis_sharpen_bridge.obj"
$importLibraryPath = Join-Path $outputDirectory "bokujuu_nis_sharpen_bridge.lib"
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw "Visual Studio Build Tools were not found."
}
$installationPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $installationPath) {
    throw "Visual Studio C++ Build Tools were not found."
}
$vcvars = Join-Path $installationPath "VC\Auxiliary\Build\vcvars64.bat"

$command = @(
    "call `"$vcvars`" >nul",
    "cl /nologo /std:c++20 /O2 /EHsc /MD /LD /Fo:`"$objectPath`" /I`"$includePath`" `"$sourcePath`" /link d3d11.lib dxgi.lib d3dcompiler.lib /IMPLIB:`"$importLibraryPath`" /OUT:`"$outputPath`""
) -join " && "

& cmd.exe /d /s /c $command
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $outputPath)) {
    throw "NIS sharpen bridge build failed with exit code $LASTEXITCODE."
}

Get-Item -LiteralPath $outputPath
