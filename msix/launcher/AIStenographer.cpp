/**
 * Native launcher for AI Stenographer with splash screen.
 *
 * Shows a splash window immediately on launch, then spawns the Python runtime.
 * The splash remains visible until the Python application creates its main window,
 * addressing the first-run delay caused by Windows Defender scanning.
 */

#include <windows.h>
#include <shlwapi.h>
#include <shlobj_core.h>
#include <appmodel.h>
#include <string>
#include <gdiplus.h>

#include "resource.h"

#pragma comment(lib, "shlwapi.lib")
#pragma comment(lib, "shell32.lib")
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
static bool g_isFirstRun = false;

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
 * Constructs the path to the first-run flag file in MSIX AppData directory.
 *
 * Algorithm:
 * 1. Get %LOCALAPPDATA% path
 * 2. Search for AI.Stenographer_* package folder in Packages directory
 * 3. Build path to flag file in LocalCache root
 *
 * @return Path to .first_run_complete flag file, or empty string if not found
 */
std::wstring GetFlagFilePath() {
    wchar_t localAppData[MAX_PATH];
    if (!ExpandEnvironmentStringsW(L"%LOCALAPPDATA%", localAppData, MAX_PATH)) {
        return L"";
    }

    std::wstring packagesDir = std::wstring(localAppData) + L"\\Packages";

    // Search for *AIStenographer_* package folder
    // Package name format: <Publisher>.<AppName>_<Hash>
    // Example: GrigoriKochanov.AIStenographer_33pmfdd1f766m
    WIN32_FIND_DATAW findData;
    std::wstring searchPattern = packagesDir + L"\\*AIStenographer_*";
    HANDLE hFind = FindFirstFileW(searchPattern.c_str(), &findData);

    if (hFind == INVALID_HANDLE_VALUE) {
        return L"";
    }

    std::wstring packageFolder = packagesDir + L"\\" + findData.cFileName;
    FindClose(hFind);

    // Place flag in same directory as Python app data
    // Matches PathResolver._get_msix_local_cache_path() behavior
    return packageFolder + L"\\LocalCache\\Local\\AI-Stenographer\\.first_run_complete";
}

/**
 * Detects if this is the first run of the application in MSIX environment.
 *
 * Algorithm:
 * 1. Check if running in MSIX package (only relevant for MSIX)
 * 2. Get flag file path in AppData
 * 3. Check if flag file exists
 *
 * @return true if first run (flag file doesn't exist), false otherwise
 */
bool IsFirstRun() {
    if (!IsRunningAsMSIX()) {
        return false;  // Only relevant for MSIX
    }

    std::wstring flagPath = GetFlagFilePath();
    if (flagPath.empty()) {
        return true;  // Assume first run if path not found
    }

    return (GetFileAttributesW(flagPath.c_str()) == INVALID_FILE_ATTRIBUTES);
}

/**
 * Creates the first-run flag file to mark successful application startup.
 *
 * Algorithm:
 * 1. Check if running in MSIX package (only relevant for MSIX)
 * 2. Get flag file path in AppData
 * 3. Create empty flag file
 *
 * The flag is created when Python application successfully starts and creates
 * its window, indicating that first run completed successfully.
 */
void CreateFirstRunFlag() {
    if (!IsRunningAsMSIX()) {
        return;
    }

    std::wstring flagPath = GetFlagFilePath();
    if (flagPath.empty()) {
        return;
    }

    // Ensure parent directory exists (Python app creates it during init)
    // Get parent directory path
    size_t lastSlash = flagPath.find_last_of(L"\\");
    if (lastSlash != std::wstring::npos) {
        std::wstring parentDir = flagPath.substr(0, lastSlash);

        // Create directory structure if needed (Python should have done this)
        // Use SHCreateDirectoryExW for recursive creation
        SHCreateDirectoryExW(NULL, parentDir.c_str(), NULL);
    }

    // Create empty flag file
    HANDLE hFile = CreateFileW(
        flagPath.c_str(),
        GENERIC_WRITE,
        0,
        NULL,
        CREATE_NEW,  // Only create if doesn't exist
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );

    if (hFile != INVALID_HANDLE_VALUE) {
        CloseHandle(hFile);
    }
    // Silent failure if file already exists or cannot be created
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

                // Draw splash image
                graphics.DrawImage(g_splashImage, 0, 0);

                // Draw first-run message if applicable
                if (g_isFirstRun) {
                    // Set up text rendering
                    graphics.SetTextRenderingHint(Gdiplus::TextRenderingHintAntiAlias);

                    // Create font (Arial, 14pt, bold)
                    Gdiplus::FontFamily fontFamily(L"Arial");
                    Gdiplus::Font font(&fontFamily, 14, Gdiplus::FontStyleBold, Gdiplus::UnitPoint);

                    // Create brushes for text and background
                    Gdiplus::SolidBrush textBrush(Gdiplus::Color(255, 255, 255, 255));  // White text
                    Gdiplus::SolidBrush bgBrush(Gdiplus::Color(180, 0, 0, 0));  // Semi-transparent black

                    // Message text
                    const wchar_t* message = L"Antivirus checks the application during the first run, this may take time";

                    // Measure text size
                    Gdiplus::RectF layoutRect(0, 0, (float)g_splashImage->GetWidth(), 0);
                    Gdiplus::RectF boundingBox;
                    graphics.MeasureString(message, -1, &font, layoutRect, &boundingBox);

                    // Position text at top center with padding
                    float x = ((float)g_splashImage->GetWidth() - boundingBox.Width) / 2;
                    float y = 20;  // px from top

                    // Draw background rectangle with padding
                    float padding = 15;
                    Gdiplus::RectF bgRect(x - padding, y - padding,
                                          boundingBox.Width + 2*padding,
                                          boundingBox.Height + 2*padding);
                    graphics.FillRectangle(&bgBrush, bgRect);

                    // Draw text
                    Gdiplus::PointF origin(x, y);
                    graphics.DrawString(message, -1, &font, origin, &textBrush);
                }
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
                    // Python window found or timeout - create flag and close splash
                    if (pythonWindow && g_isFirstRun) {
                        CreateFirstRunFlag();
                    }
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

    // Detect first run
    g_isFirstRun = IsFirstRun();

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
