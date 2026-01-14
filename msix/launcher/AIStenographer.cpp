// AIStenographer.cpp
// Native launcher for AI Stenographer MSIX package
// Launches the Python runtime with the main application bytecode

#include <windows.h>
#include <shlwapi.h>
#include <string>

#pragma comment(lib, "shlwapi.lib")

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

    if (!CreateProcessW(
        NULL,                   // Application name (NULL = use command line)
        &cmdLine[0],           // Command line (modifiable buffer required)
        NULL,                   // Process security attributes
        NULL,                   // Thread security attributes
        FALSE,                  // Inherit handles
        0,                      // Creation flags
        NULL,                   // Environment block
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
