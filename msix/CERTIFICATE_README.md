# MSIX Certificate Guide

## Overview

This guide explains how to create and use test certificates for MSIX package signing during development and testing.

**IMPORTANT:** Self-signed certificates are for **local testing only**. When you submit to the Microsoft Store, Microsoft will automatically re-sign your package with a trusted certificate.

## Quick Start

### 1. Create Test Certificate

Run the PowerShell script to generate a self-signed certificate:

```powershell
cd msix
.\create_test_certificate.ps1
```

This will create two files:
- `AIStenographer_TestCert.pfx` - Private key (password: `test123`)
- `AIStenographer_TestCert.cer` - Public key (for installation)

### 2. Install Certificate (Administrator Required)

If the script didn't auto-install (requires admin), manually install:

```powershell
# Run as Administrator
Import-Certificate -FilePath "AIStenographer_TestCert.cer" -CertStoreLocation "Cert:\LocalMachine\Root"
```

Or double-click `AIStenographer_TestCert.cer` and:
1. Click "Install Certificate..."
2. Select "Local Machine"
3. Choose "Place all certificates in the following store"
4. Select "Trusted Root Certification Authorities"
5. Click "Finish"

### 3. Sign MSIX Package

After building the MSIX package, sign it using Windows SDK's SignTool:

```powershell
# Locate SignTool.exe (usually in Windows SDK)
$SignTool = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64\signtool.exe"

# Sign the package
& $SignTool sign /fd SHA256 /a /f "AIStenographer_TestCert.pfx" /p "test123" /t http://timestamp.digicert.com "AIStenographer_1.0.0.0_x64.msix"
```

### 4. Verify Signature

```powershell
& $SignTool verify /pa "AIStenographer_1.0.0.0_x64.msix"
```

---

## Detailed Information

### Certificate Requirements

For MSIX packages, the certificate must:
- Have **Code Signing** capability (EKU: 1.3.6.1.5.5.7.3.3)
- Match the **Publisher** field in `AppxManifest.xml`
- Be installed in the **Trusted Root Certification Authorities** store

### Publisher Matching

The certificate's `Subject` field **must exactly match** the `Publisher` in `AppxManifest.xml`:

**AppxManifest.xml:**
```xml
<Identity
  Name="AI.Stenographer"
  Publisher="CN=Grigori Kochanov"
  Version="1.0.0.0" />
```

**Certificate:**
```powershell
Subject: CN=Grigori Kochanov
```

If these don't match, the MSIX installation will fail.

### Timestamping

Always use a timestamp server when signing:

```powershell
/t http://timestamp.digicert.com
```

**Why?**
- Allows the package to be installed even after the certificate expires
- Proves the package was signed when the certificate was valid
- Required for production packages

### Certificate Expiration

The test certificate is valid for **2 years** from creation. After expiration:
- Re-run `create_test_certificate.ps1` to generate a new certificate
- Re-sign all MSIX packages with the new certificate
- Re-install the new certificate to Trusted Root

---

## Troubleshooting

### "Certificate chain processed, but terminated in a root certificate which is not trusted"

**Solution:** Install the `.cer` file to Trusted Root Certification Authorities:
```powershell
Import-Certificate -FilePath "AIStenographer_TestCert.cer" -CertStoreLocation "Cert:\LocalMachine\Root"
```

### "The publisher name specified in the package doesn't match the subject distinguished name of the signing certificate"

**Solution:** Ensure the `Publisher` in `AppxManifest.xml` matches the certificate's `Subject`:
- Check manifest: `CN=Grigori Kochanov`
- Check certificate: `(Get-PfxCertificate "AIStenographer_TestCert.pfx").Subject`

### "No certificates were found that met all the given criteria"

**Solution:** The certificate is missing or not in the correct store:
```powershell
# List certificates in Personal store
Get-ChildItem -Path "Cert:\CurrentUser\My" | Where-Object { $_.Subject -like "*Grigori Kochanov*" }

# If missing, re-run create_test_certificate.ps1
```

### SignTool not found

**Solution:** Install Windows SDK or locate existing installation:
```powershell
# Common paths
dir "C:\Program Files (x86)\Windows Kits\10\bin\*\x64\signtool.exe" -Recurse

# Or install Windows SDK
# https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/
```

---

## Security Notes

### DO NOT use test certificates for production!

- Test certificates are **not trusted** by other computers
- Microsoft Store **requires** Microsoft's trusted certificate
- Self-signed certificates are **only for local development**

### Certificate Password

The default password is `test123`. This is fine for local testing, but:
- Change it if sharing the certificate
- Never commit `.pfx` files to public repositories
- Use a secure password manager for production certificates

### Certificate Storage

The script stores certificates in:
- **Personal Store:** `Cert:\CurrentUser\My` (private key)
- **Trusted Root:** `Cert:\LocalMachine\Root` (public key)

To remove the certificate:
```powershell
# Remove from Personal store
Get-ChildItem -Path "Cert:\CurrentUser\My" | Where-Object { $_.Subject -eq "CN=Grigori Kochanov" } | Remove-Item

# Remove from Trusted Root (requires admin)
Get-ChildItem -Path "Cert:\LocalMachine\Root" | Where-Object { $_.Subject -eq "CN=Grigori Kochanov" } | Remove-Item
```

---

## Microsoft Store Submission

When submitting to the Microsoft Store:

1. **Do NOT sign** the package yourself - upload unsigned
2. Microsoft Partner Center will **automatically sign** with a trusted certificate
3. The Store certificate will:
   - Match your Publisher name
   - Be trusted by all Windows devices
   - Automatically renew with updates

Your test certificate is **only for local sideloading testing**.

---

## References

- [Create certificate for package signing](https://learn.microsoft.com/en-us/windows/msix/package/create-certificate-package-signing)
- [Sign package overview](https://learn.microsoft.com/en-us/windows/msix/package/signing-package-overview)
- [MSIX certificates developer guide](https://www.advancedinstaller.com/msix-certificates-developer.html)
- [Self-signed certificate tutorial](https://www.tmurgent.com/TmBlog/?p=3461)

---

## Quick Reference

### Generate Certificate
```powershell
.\create_test_certificate.ps1
```

### Install Certificate (Admin)
```powershell
Import-Certificate -FilePath "AIStenographer_TestCert.cer" -CertStoreLocation "Cert:\LocalMachine\Root"
```

### Sign Package
```powershell
signtool.exe sign /fd SHA256 /a /f "AIStenographer_TestCert.pfx" /p "test123" /t http://timestamp.digicert.com "AIStenographer_1.0.0.0_x64.msix"
```

### Verify Signature
```powershell
signtool.exe verify /pa "AIStenographer_1.0.0.0_x64.msix"
```

### Check Certificate
```powershell
Get-PfxCertificate "AIStenographer_TestCert.pfx" | Format-List Subject, NotAfter, Thumbprint
```
