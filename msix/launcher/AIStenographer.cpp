/**
 * Native launcher for AI Stenographer with splash screen.
 *
 * Shows a splash window immediately on launch, then spawns the Python runtime.
 * The splash remains visible until the Python application creates its main window,
 * addressing the first-run delay caused by Windows Defender scanning.
 */

#include <windows.h>
#include <shlwapi.h>
#include <appmodel.h>
#include <string>
#include <gdiplus.h>

#include "resource.h"

#pragma comment(lib, "shlwapi.lib")
#pragma comment(lib, "kernel32.lib")
#pragma comment(lib, "gdiplus.lib")
#pragma comment(lib, "ole32.lib")

// Timer ID for polling Python window
#define TIMER_CHECK_WINDOW 1
// Poll interval in milliseconds
#define POLL_INTERVAL_MS 100
// Maximum wait time before giving up (120 seconds)
#define MAX_WAIT_MS 120000

// Global state for splash window
static HWND g_splashWindow = NULL;
static Gdiplus::Image* g_splashImage = NULL;
static ULONG_PTR g_gdiplusToken = 0;
static DWORD g_startTime = 0;

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
 * Loads the splash image from embedded resource using GDI+.
 *
 * Algorithm:
 * 1. Find the resource in the executable
 * 2. Create an IStream from the resource data
 * 3. Load image from stream using GDI+
 *
 * @return GDI+ Image pointer, or NULL on failure
 */
Gdiplus::Image* LoadSplashImageFromResource() {
    // Find the embedded image resource
    HRSRC hResource = FindResourceW(NULL, MAKEINTRESOURCEW(IDR_SPLASH_IMAGE), RT_RCDATA);
    if (!hResource) {
        return NULL;
    }

    // Get resource size and data
    DWORD resourceSize = SizeofResource(NULL, hResource);
    HGLOBAL hMemory = LoadResource(NULL, hResource);
    if (!hMemory) {
        return NULL;
    }

    void* resourceData = LockResource(hMemory);
    if (!resourceData) {
        return NULL;
    }

    // Create a global memory block for the stream
    HGLOBAL hBuffer = GlobalAlloc(GMEM_MOVEABLE, resourceSize);
    if (!hBuffer) {
        return NULL;
    }

    void* buffer = GlobalLock(hBuffer);
    if (!buffer) {
        GlobalFree(hBuffer);
        return NULL;
    }

    // Copy resource data to the buffer
    memcpy(buffer, resourceData, resourceSize);
    GlobalUnlock(hBuffer);

    // Create IStream from the memory block
    IStream* stream = NULL;
    if (FAILED(CreateStreamOnHGlobal(hBuffer, TRUE, &stream))) {
        GlobalFree(hBuffer);
        return NULL;
    }

    // Load image from stream
    Gdiplus::Image* image = Gdiplus::Image::FromStream(stream);
    stream->Release();

    // Check if image loaded successfully
    if (image && image->GetLastStatus() != Gdiplus::Ok) {
        delete image;
        return NULL;
    }

    return image;
}

/**
 * Window procedure for the splash window.
 *
 * Handles:
 * - WM_PAINT: Draws the splash image
 * - WM_TIMER: Checks if Python window appeared
 * - WM_DESTROY: Cleanup
 */
LRESULT CALLBACK SplashWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);

            if (g_splashImage) {
                Gdiplus::Graphics graphics(hdc);
                graphics.DrawImage(g_splashImage, 0, 0);
            }

            EndPaint(hwnd, &ps);
            return 0;
        }

        case WM_TIMER: {
            if (wParam == TIMER_CHECK_WINDOW) {
                // Check if any Python application window exists:
                // - Main window: "Speech-to-Text Display"
                // - Model download dialog: "Required Models Not Found"
                // - Loading window: "Loading AI Stenographer..."
                HWND pythonWindow = FindWindowW(NULL, L"Speech-to-Text Display");
                if (!pythonWindow) {
                    pythonWindow = FindWindowW(NULL, L"Required Models Not Found");
                }
                if (!pythonWindow) {
                    pythonWindow = FindWindowW(NULL, L"Loading AI Stenographer...");
                }

                // Check timeout
                DWORD elapsed = GetTickCount() - g_startTime;

                if (pythonWindow || elapsed >= MAX_WAIT_MS) {
                    // Python window found or timeout - close splash
                    KillTimer(hwnd, TIMER_CHECK_WINDOW);
                    DestroyWindow(hwnd);
                }
            }
            return 0;
        }

        case WM_DESTROY: {
            PostQuitMessage(0);
            return 0;
        }
    }

    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

/**
 * Creates and shows the splash window.
 *
 * Algorithm:
 * 1. Register a custom window class
 * 2. Calculate centered position
 * 3. Create borderless popup window
 * 4. Start timer for Python window detection
 *
 * @return Window handle, or NULL on failure
 */
HWND CreateSplashWindow(HINSTANCE hInstance) {
    // Register window class
    WNDCLASSEXW wc = { sizeof(wc) };
    wc.lpfnWndProc = SplashWndProc;
    wc.hInstance = hInstance;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
    wc.lpszClassName = L"AIStenographerSplash";

    if (!RegisterClassExW(&wc)) {
        return NULL;
    }

    // Get image dimensions (default to 700x700 if image not loaded)
    int width = 700;
    int height = 700;
    if (g_splashImage) {
        width = g_splashImage->GetWidth();
        height = g_splashImage->GetHeight();
    }

    // Center on screen
    int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);
    int x = (screenWidth - width) / 2;
    int y = (screenHeight - height) / 2;

    // Create borderless popup window (topmost so it stays visible)
    HWND hwnd = CreateWindowExW(
        WS_EX_TOPMOST,              // Stay on top during Defender scan
        L"AIStenographerSplash",
        L"AI Stenographer",
        WS_POPUP,                   // No title bar or borders
        x, y, width, height,
        NULL, NULL, hInstance, NULL
    );

    if (hwnd) {
        ShowWindow(hwnd, SW_SHOW);
        UpdateWindow(hwnd);

        // Start timer to check for Python window
        g_startTime = GetTickCount();
        SetTimer(hwnd, TIMER_CHECK_WINDOW, POLL_INTERVAL_MS, NULL);
    }

    return hwnd;
}

/**
 * Launches the Python runtime with the application bytecode.
 *
 * @param exePath Directory containing the launcher executable
 * @return true on success, false on failure
 */
bool LaunchPythonApplication(const wchar_t* exePath) {
    // Build paths to Python runtime and application
    std::wstring pythonPath = std::wstring(exePath) + L"\\_internal\\runtime\\pythonw.exe";
    std::wstring appPath = std::wstring(exePath) + L"\\_internal\\app\\main.pyc";

    // Verify Python executable exists
    if (GetFileAttributesW(pythonPath.c_str()) == INVALID_FILE_ATTRIBUTES) {
        std::wstring errorMsg = L"Python runtime not found:\n" + pythonPath;
        MessageBoxW(NULL, errorMsg.c_str(), L"AI Stenographer - Error", MB_ICONERROR);
        return false;
    }

    // Verify application bytecode exists
    if (GetFileAttributesW(appPath.c_str()) == INVALID_FILE_ATTRIBUTES) {
        std::wstring errorMsg = L"Application not found:\n" + appPath;
        MessageBoxW(NULL, errorMsg.c_str(), L"AI Stenographer - Error", MB_ICONERROR);
        return false;
    }

    // Launch Python with application bytecode
    STARTUPINFOW si = { sizeof(si) };
    PROCESS_INFORMATION pi;

    // Build command line: "pythonw.exe" "main.pyc"
    std::wstring cmdLine = L"\"" + pythonPath + L"\" \"" + appPath + L"\"";

    // Build environment block with MSIX indicator if running in package context
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
        creationFlags,          // Creation flags
        envBlock,               // Environment block
        exePath,               // Working directory
        &si,                   // Startup info
        &pi))
    {
        DWORD error = GetLastError();
        wchar_t errorMsg[512];
        swprintf_s(errorMsg, 512, L"Failed to launch AI Stenographer\n\nError code: %lu", error);
        MessageBoxW(NULL, errorMsg, L"AI Stenographer - Error", MB_ICONERROR);
        return false;
    }

    // Close process handles (we don't need to wait)
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    return true;
}

/**
 * Main entry point for the AI Stenographer launcher.
 *
 * Algorithm:
 * 1. Initialize GDI+ for image loading
 * 2. Load splash image from embedded resource
 * 3. Show splash window immediately
 * 4. Launch Python application
 * 5. Run message loop until Python window appears
 * 6. Cleanup and exit
 *
 * @return 0 on success, 1 on failure
 */
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    // Initialize GDI+
    Gdiplus::GdiplusStartupInput gdiplusStartupInput;
    if (Gdiplus::GdiplusStartup(&g_gdiplusToken, &gdiplusStartupInput, NULL) != Gdiplus::Ok) {
        MessageBoxW(NULL, L"Failed to initialize graphics", L"AI Stenographer - Error", MB_ICONERROR);
        return 1;
    }

    // Load splash image from embedded resource
    g_splashImage = LoadSplashImageFromResource();

    // Create and show splash window (even if image failed to load)
    g_splashWindow = CreateSplashWindow(hInstance);

    // Get executable directory
    wchar_t exePath[MAX_PATH];
    if (!GetModuleFileNameW(NULL, exePath, MAX_PATH)) {
        MessageBoxW(NULL, L"Failed to get executable path", L"AI Stenographer - Error", MB_ICONERROR);
        if (g_splashWindow) DestroyWindow(g_splashWindow);
        if (g_splashImage) delete g_splashImage;
        Gdiplus::GdiplusShutdown(g_gdiplusToken);
        return 1;
    }
    PathRemoveFileSpecW(exePath);

    // Launch Python application
    if (!LaunchPythonApplication(exePath)) {
        if (g_splashWindow) DestroyWindow(g_splashWindow);
        if (g_splashImage) delete g_splashImage;
        Gdiplus::GdiplusShutdown(g_gdiplusToken);
        return 1;
    }

    // Run message loop for splash window
    if (g_splashWindow) {
        MSG msg;
        while (GetMessage(&msg, NULL, 0, 0)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    // Cleanup
    if (g_splashImage) {
        delete g_splashImage;
        g_splashImage = NULL;
    }
    Gdiplus::GdiplusShutdown(g_gdiplusToken);

    return 0;
}
