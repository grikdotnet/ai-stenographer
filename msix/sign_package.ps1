# sign_package.ps1
# PowerShell script to sign MSIX packages with test certificate
#
# Usage:
#   .\sign_package.ps1 -Package "AIStenographer_1.0.0.0_x64.msix"
#   .\sign_package.ps1 -Package "AIStenographer_1.0.0.0_x64.msix" -Verify

param(
    [Parameter(Mandatory=$true)]
    [string]$Package,

    [string]$Certificate = "AIStenographer_TestCert.pfx",
    [string]$Password = "test123",
    [string]$TimestampServer = "http://timestamp.digicert.com",
    [switch]$Verify = $false
)

# Ensure we're in the right directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "AI Stenographer - MSIX Package Signer" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if package exists
if (-not (Test-Path $Package)) {
    Write-Host "ERROR: Package not found: $Package" -ForegroundColor Red
    Write-Host ""
    Write-Host "Available MSIX packages:" -ForegroundColor Yellow
    Get-ChildItem -Filter "*.msix" | ForEach-Object {
        Write-Host "  $($_.Name)" -ForegroundColor Gray
    }
    exit 1
}

# Check if certificate exists
if (-not (Test-Path $Certificate)) {
    Write-Host "ERROR: Certificate not found: $Certificate" -ForegroundColor Red
    Write-Host ""
    Write-Host "Run create_test_certificate.ps1 first to generate a certificate." -ForegroundColor Yellow
    exit 1
}

# Find SignTool.exe
Write-Host "Locating SignTool.exe..." -ForegroundColor Yellow

$SignToolPaths = @(
    "C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64\signtool.exe",
    "C:\Program Files (x86)\Windows Kits\10\bin\10.0.19041.0\x64\signtool.exe",
    "C:\Program Files (x86)\Windows Kits\10\bin\10.0.18362.0\x64\signtool.exe"
)

# Search for SignTool in Windows SDK directories
$WindowsKitsPath = "C:\Program Files (x86)\Windows Kits\10\bin"
if (Test-Path $WindowsKitsPath) {
    $foundSignTools = Get-ChildItem -Path $WindowsKitsPath -Filter "signtool.exe" -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.FullName -like "*x64*" }
    foreach ($tool in $foundSignTools) {
        $SignToolPaths += $tool.FullName
    }
}

$SignTool = $null
foreach ($path in $SignToolPaths) {
    if (Test-Path $path) {
        $SignTool = $path
        break
    }
}

if (-not $SignTool) {
    Write-Host "ERROR: SignTool.exe not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Install Windows SDK from:" -ForegroundColor Yellow
    Write-Host "  https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Or specify the path manually:" -ForegroundColor Yellow
    Write-Host '  $SignTool = "C:\Path\To\signtool.exe"' -ForegroundColor Gray
    exit 1
}

Write-Host "Found: $SignTool" -ForegroundColor Green
Write-Host ""

# Sign the package
Write-Host "Signing package: $Package" -ForegroundColor Green
Write-Host "  Certificate: $Certificate" -ForegroundColor Gray
Write-Host "  Timestamp: $TimestampServer" -ForegroundColor Gray
Write-Host ""

try {
    $certPath = Resolve-Path $Certificate
    $packagePath = Resolve-Path $Package

    $signArgs = @(
        "sign",
        "/fd", "SHA256",
        "/a",
        "/f", $certPath,
        "/p", $Password,
        "/t", $TimestampServer,
        $packagePath
    )

    Write-Host "Executing: signtool.exe $($signArgs -join ' ')" -ForegroundColor Gray
    Write-Host ""

    & $SignTool @signArgs

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "Package signed successfully!" -ForegroundColor Green
        Write-Host ""
    } else {
        Write-Host ""
        Write-Host "ERROR: Signing failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
} catch {
    Write-Host "ERROR: Failed to sign package" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

# Verify signature if requested
if ($Verify) {
    Write-Host "Verifying signature..." -ForegroundColor Yellow
    Write-Host ""

    try {
        $verifyArgs = @(
            "verify",
            "/pa",
            $packagePath
        )

        & $SignTool @verifyArgs

        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host "Signature verified successfully!" -ForegroundColor Green
            Write-Host ""
        } else {
            Write-Host ""
            Write-Host "WARNING: Signature verification failed!" -ForegroundColor Red
            Write-Host "Make sure the certificate is installed to Trusted Root." -ForegroundColor Yellow
            Write-Host ""
        }
    } catch {
        Write-Host "ERROR: Failed to verify signature" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
    }
}

# Display package info
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Package Information" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

$packageInfo = Get-Item $Package
Write-Host "  File: $($packageInfo.Name)" -ForegroundColor Gray
Write-Host "  Size: $([math]::Round($packageInfo.Length / 1MB, 2)) MB" -ForegroundColor Gray
Write-Host "  Modified: $($packageInfo.LastWriteTime)" -ForegroundColor Gray
Write-Host ""

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Install certificate to Trusted Root (if not already done)" -ForegroundColor Gray
Write-Host "     Import-Certificate -FilePath 'AIStenographer_TestCert.cer' -CertStoreLocation 'Cert:\LocalMachine\Root'" -ForegroundColor Cyan
Write-Host ""
Write-Host "  2. Install MSIX package for testing" -ForegroundColor Gray
Write-Host "     Add-AppxPackage -Path '$Package'" -ForegroundColor Cyan
Write-Host ""
Write-Host "  3. Verify signature (optional)" -ForegroundColor Gray
Write-Host "     .\sign_package.ps1 -Package '$Package' -Verify" -ForegroundColor Cyan
Write-Host ""
