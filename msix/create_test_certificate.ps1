# create_test_certificate.ps1
# PowerShell script to create a self-signed certificate for MSIX package testing
#
# IMPORTANT: This certificate is for TESTING ONLY!
# Microsoft Store will re-sign the package with a trusted certificate during submission.
#
# Requirements:
# - Windows 10/11 with PowerShell 5.1 or later
# - Administrator privileges (for installation to Trusted Root)
#
# Usage:
#   .\create_test_certificate.ps1

param(
    [string]$CertPassword = "test123",
    [string]$Subject = "CN=3F8691C4-05D3-45C7-AB1E-113776D7E567",
    [string]$FriendlyName = "AI Stenographer Test Certificate",
    [string]$OutputDir = ".",
    [switch]$SkipInstall = $false
)

# Ensure we're in the right directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "AI Stenographer - Test Certificate Generator" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "WARNING: This certificate is for TESTING ONLY!" -ForegroundColor Yellow
Write-Host "The Microsoft Store will re-sign with a trusted certificate." -ForegroundColor Yellow
Write-Host ""

# Check if certificate already exists
$existingCert = Get-ChildItem -Path "Cert:\CurrentUser\My" | Where-Object { $_.Subject -eq $Subject -and $_.FriendlyName -eq $FriendlyName }

if ($existingCert) {
    Write-Host "Found existing certificate:" -ForegroundColor Yellow
    Write-Host "  Subject: $($existingCert.Subject)" -ForegroundColor Gray
    Write-Host "  Thumbprint: $($existingCert.Thumbprint)" -ForegroundColor Gray
    Write-Host "  Expiry: $($existingCert.NotAfter)" -ForegroundColor Gray
    Write-Host ""

    $response = Read-Host "Do you want to create a new certificate? (Y/N)"
    if ($response -ne "Y" -and $response -ne "y") {
        Write-Host "Using existing certificate..." -ForegroundColor Green
        $cert = $existingCert
        $skipCreation = $true
    } else {
        Write-Host "Removing existing certificate..." -ForegroundColor Yellow
        Remove-Item -Path "Cert:\CurrentUser\My\$($existingCert.Thumbprint)" -Force
        $skipCreation = $false
    }
} else {
    $skipCreation = $false
}

# Create new certificate if needed
if (-not $skipCreation) {
    Write-Host "Creating new self-signed certificate..." -ForegroundColor Green
    Write-Host "  Subject: $Subject" -ForegroundColor Gray
    Write-Host "  Friendly Name: $FriendlyName" -ForegroundColor Gray
    Write-Host ""

    try {
        $cert = New-SelfSignedCertificate `
            -Type Custom `
            -Subject $Subject `
            -KeyUsage DigitalSignature `
            -FriendlyName $FriendlyName `
            -CertStoreLocation "Cert:\CurrentUser\My" `
            -TextExtension @("2.5.29.37={text}1.3.6.1.5.5.7.3.3", "2.5.29.19={text}") `
            -NotAfter (Get-Date).AddYears(2)

        Write-Host "Certificate created successfully!" -ForegroundColor Green
        Write-Host "  Thumbprint: $($cert.Thumbprint)" -ForegroundColor Gray
        Write-Host "  Valid until: $($cert.NotAfter)" -ForegroundColor Gray
        Write-Host ""
    } catch {
        Write-Host "ERROR: Failed to create certificate" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        exit 1
    }
}

# Export certificate to PFX (with private key)
$pfxPath = Join-Path $OutputDir "AIStenographer_TestCert.pfx"
Write-Host "Exporting certificate to PFX..." -ForegroundColor Green
Write-Host "  File: $pfxPath" -ForegroundColor Gray

try {
    $password = ConvertTo-SecureString -String $CertPassword -Force -AsPlainText
    Export-PfxCertificate -Cert "Cert:\CurrentUser\My\$($cert.Thumbprint)" `
        -FilePath $pfxPath `
        -Password $password `
        -Force | Out-Null

    Write-Host "PFX exported successfully!" -ForegroundColor Green
    Write-Host ""
} catch {
    Write-Host "ERROR: Failed to export PFX" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

# Export certificate to CER (public key only, for installation)
$cerPath = Join-Path $OutputDir "AIStenographer_TestCert.cer"
Write-Host "Exporting certificate to CER..." -ForegroundColor Green
Write-Host "  File: $cerPath" -ForegroundColor Gray

try {
    Export-Certificate -Cert "Cert:\CurrentUser\My\$($cert.Thumbprint)" `
        -FilePath $cerPath `
        -Force | Out-Null

    Write-Host "CER exported successfully!" -ForegroundColor Green
    Write-Host ""
} catch {
    Write-Host "ERROR: Failed to export CER" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

# Install certificate to Trusted Root (requires admin)
if (-not $SkipInstall) {
    Write-Host "Installing certificate to Trusted Root store..." -ForegroundColor Yellow
    Write-Host "This requires administrator privileges!" -ForegroundColor Yellow
    Write-Host ""

    # Check if running as administrator
    $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

    if (-not $isAdmin) {
        Write-Host "WARNING: Not running as administrator!" -ForegroundColor Red
        Write-Host "You will need to install the certificate manually:" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  Option 1: Re-run this script as Administrator" -ForegroundColor Gray
        Write-Host "  Option 2: Double-click $cerPath and install to Trusted Root" -ForegroundColor Gray
        Write-Host "  Option 3: Run this command as Administrator:" -ForegroundColor Gray
        Write-Host "    Import-Certificate -FilePath '$cerPath' -CertStoreLocation 'Cert:\LocalMachine\Root'" -ForegroundColor Cyan
        Write-Host ""
    } else {
        try {
            Import-Certificate -FilePath $cerPath -CertStoreLocation "Cert:\LocalMachine\Root" | Out-Null
            Write-Host "Certificate installed to Trusted Root successfully!" -ForegroundColor Green
            Write-Host ""
        } catch {
            Write-Host "ERROR: Failed to install certificate" -ForegroundColor Red
            Write-Host $_.Exception.Message -ForegroundColor Red
            Write-Host ""
            Write-Host "You can install manually by double-clicking: $cerPath" -ForegroundColor Yellow
            Write-Host ""
        }
    }
}

# Summary
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Certificate Generation Complete!" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Files created:" -ForegroundColor Green
Write-Host "  $pfxPath (private key, password: $CertPassword)" -ForegroundColor Gray
Write-Host "  $cerPath (public key)" -ForegroundColor Gray
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Build MSIX package using build_msix_distribution.py" -ForegroundColor Gray
Write-Host "  2. Sign MSIX package with this certificate" -ForegroundColor Gray
Write-Host "  3. Install certificate to Trusted Root (if not already done)" -ForegroundColor Gray
Write-Host "  4. Install MSIX package for testing" -ForegroundColor Gray
Write-Host ""
Write-Host "REMEMBER: This is for TESTING ONLY!" -ForegroundColor Yellow
Write-Host "Microsoft Store will use its own trusted certificate." -ForegroundColor Yellow
Write-Host ""
