@echo off
REM Build automation helper for STT distribution

:menu
cls
echo ============================================================
echo STT Distribution Builder
echo ============================================================
echo.
echo 1. Full Build (download Python, install dependencies)
echo 2. Quick Rebuild (only update application code)
echo 3. Watch and Auto-Rebuild (development mode)
echo 4. Run Application (from built distribution)
echo 5. Clean Build Directory
echo 6. Exit
echo.
echo ============================================================
set /p choice="Select option (1-6): "

if "%choice%"=="1" goto full_build
if "%choice%"=="2" goto quick_rebuild
if "%choice%"=="3" goto watch
if "%choice%"=="4" goto run_app
if "%choice%"=="5" goto clean
if "%choice%"=="6" goto end
goto menu

:full_build
echo.
echo Running full build...
echo.
python build_distribution.py
if errorlevel 1 (
    echo.
    echo BUILD FAILED
    pause
    goto menu
)
echo.
echo BUILD SUCCESSFUL
pause
goto menu

:quick_rebuild
echo.
echo Running quick rebuild...
echo.
python rebuild_quick.py
if errorlevel 1 (
    echo.
    echo REBUILD FAILED
    pause
    goto menu
)
echo.
echo REBUILD SUCCESSFUL
pause
goto menu

:watch
echo.
echo Starting file watcher...
echo Press Ctrl+C to stop watching
echo.
python watch_and_rebuild.py
pause
goto menu

:run_app
echo.
echo Running application from dist...
echo.
if not exist "dist\STT-Stenographer\_internal\runtime\pythonw.exe" (
    echo Error: Distribution not built yet!
    echo Run option 1 first to build the distribution.
    pause
    goto menu
)
cd dist\STT-Stenographer
_internal\runtime\pythonw.exe _internal\app\main.pyc
cd ..\..
pause
goto menu

:clean
echo.
echo Cleaning build directory...
echo.
if exist "dist\STT-Stenographer" (
    rmdir /s /q "dist\STT-Stenographer"
    echo Build directory cleaned
) else (
    echo Build directory doesn't exist
)
pause
goto menu

:end
echo.
echo Goodbye!
exit /b 0
