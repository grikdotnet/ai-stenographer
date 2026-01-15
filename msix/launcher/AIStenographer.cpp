// Native launcher for Python runtime with the main application bytecode

#include <windows.h>
#include <shlwapi.h>
#include <appmodel.h>
#include <string>

#pragma comment(lib, "shlwapi.lib")
#pragma comment(lib, "kernel32.lib")

/**
 * Detects if the application is running inside an MSIX package.
 *
 * Uses GetCurrentPackageFullName() API - returns ERROR_INSUFFICIENT_BUFFER
 * when running in a package context, APPMODEL_ERROR_NO_PACKAGE otherwise.
 *
 * @return true if running as MSIX package, false otherwise
 */
bool IsRunningAsMSIX() {
    UINT32 length = 0;
    LONG result = GetCurrentPackageFullName(&length, nullptr);
    return (result == ERROR_INSUFFICIENT_BUFFER);
}

/**
 * Main entry point for the AI Stenographer launcher.
 *
 * Algorithm:
 * 1. Get the directory of the current executable
 * 2. Construct paths to Python runtime and main.pyc
 * 3. Launch Python with the application bytecode
 * 4. Exit immediately without waiting (background operation)
 *
 * @return 0 on success, 1 on failure
 */
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    // Get executable directory
    wchar_t exePath[MAX_PATH];
    if (!GetModuleFileNameW(NULL, exePath, MAX_PATH)) {
        MessageBoxW(NULL, L"Failed to get executable path", L"AI Stenographer - Error", MB_ICONERROR);
        return 1;
    }
    PathRemoveFileSpecW(exePath);

    // Build paths to Python runtime and application
    std::wstring pythonPath = std::wstring(exePath) + L"\\_internal\\runtime\\pythonw.exe";
    std::wstring appPath = std::wstring(exePath) + L"\\_internal\\app\\main.pyc";

    // Verify Python executable exists
    if (GetFileAttributesW(pythonPath.c_str()) == INVALID_FILE_ATTRIBUTES) {
        std::wstring errorMsg = L"Python runtime not found:\n" + pythonPath;
        MessageBoxW(NULL, errorMsg.c_str(), L"AI Stenographer - Error", MB_ICONERROR);
        return 1;
    }

    // Verify application bytecode exists
    if (GetFileAttributesW(appPath.c_str()) == INVALID_FILE_ATTRIBUTES) {
        std::wstring errorMsg = L"Application not found:\n" + appPath;
        MessageBoxW(NULL, errorMsg.c_str(), L"AI Stenographer - Error", MB_ICONERROR);
        return 1;
    }

    // Launch Python with application bytecode
    STARTUPINFOW si = { sizeof(si) };
    PROCESS_INFORMATION pi;

    // Build command line: "pythonw.exe" "main.pyc"
    std::wstring cmdLine = L"\"" + pythonPath + L"\" \"" + appPath + L"\"";

    // Build environment block with MSIX indicator if running in package context
    // Environment block format: VAR1=VALUE1\0VAR2=VALUE2\0\0 (double null terminated)
    std::wstring envString;
    wchar_t* envBlock = NULL;
    DWORD creationFlags = 0;

    if (IsRunningAsMSIX()) {
        // Add MSIX_PACKAGE_IDENTITY to signal Store mode to Python application
        envString = L"MSIX_PACKAGE_IDENTITY=1";
        envString += L'\0';

        // Append current environment variables
        wchar_t* currentEnv = GetEnvironmentStringsW();
        if (currentEnv) {
            wchar_t* p = currentEnv;
            while (*p) {
                size_t len = wcslen(p);
                envString += p;
                envString += L'\0';
                p += len + 1;
            }
            FreeEnvironmentStringsW(currentEnv);
        }
        envString += L'\0';  // Double null terminator
        envBlock = &envString[0];
        creationFlags = CREATE_UNICODE_ENVIRONMENT;
    }

    if (!CreateProcessW(
        NULL,                   // Application name (NULL = use command line)
        &cmdLine[0],           // Command line (modifiable buffer required)
        NULL,                   // Process security attributes
        NULL,                   // Thread security attributes
        FALSE,                  // Inherit handles
        creationFlags,          // Creation flags (CREATE_UNICODE_ENVIRONMENT if MSIX)
        envBlock,               // Environment block (with MSIX indicator if in package)
        exePath,               // Working directory (executable directory)
        &si,                   // Startup info
        &pi))                  // Process information
    {
        DWORD error = GetLastError();
        wchar_t errorMsg[512];
        swprintf_s(errorMsg, 512, L"Failed to launch AI Stenographer\n\nError code: %lu", error);
        MessageBoxW(NULL, errorMsg, L"AI Stenographer - Error", MB_ICONERROR);
        return 1;
    }

    // Don't wait for process to exit (background operation)
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    return 0;
}
