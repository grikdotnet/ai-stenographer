STT Build Automation - Quick Reference
=======================================

INITIAL SETUP (One Time)
-------------------------
1. Run full build:
   python build_distribution.py

   This creates: dist/STT-Stenographer/
   Time: 5-10 minutes


DAILY DEVELOPMENT
-----------------
Option 1: Manual Rebuild (after code changes)
   python rebuild_quick.py
   Time: 5-10 seconds

Option 2: Auto Rebuild (recommended for active development)
   python watch_and_rebuild.py
   (Watches src/, main.py, config/, stenographer.jpg)
   (Press Ctrl+C to stop)

Option 3: Interactive Menu (Windows)
   build.bat
   (Select from menu: build, rebuild, watch, run, clean)


TESTING BUILT APPLICATION
--------------------------
cd dist\STT-Stenographer
_internal\runtime\pythonw.exe _internal\app\main.pyc


WHEN TO USE EACH BUILD TYPE
----------------------------
Full Build (build_distribution.py):
  - First time setup
  - After changing requirements.txt (dependencies)
  - After Python version change
  - Before creating release

Quick Rebuild (rebuild_quick.py):
  - After editing any .py file
  - After changing config files
  - After updating assets (images)
  - Does NOT reinstall dependencies

Watch Mode (watch_and_rebuild.py):
  - During active development
  - Automatically rebuilds on file save
  - Saves time vs manual rebuilds


WHAT GETS COMPILED
-------------------
Source (.py) → Bytecode (.pyc):
  - src/*.py → src/*.pyc
  - main.py → main.pyc

All .py files are REMOVED from distribution
Only .pyc files are included (source code hidden)


BUILD OUTPUT
------------
dist/STT-Stenographer/
├── _internal/
│   ├── runtime/        # Python 3.13.0 (signed)
│   ├── Lib/            # Dependencies
│   ├── app/            # Your code (.pyc only)
│   └── models/         # Downloaded at runtime
└── (More files added in Steps 7-10)


CURRENT STATUS
--------------
✓ Step 1: Python runtime downloaded and configured
✓ Step 2: Pip enabled in embedded Python
✓ Automation: Quick rebuild and file watcher ready

Remaining: Steps 3-10 (dependencies, packaging, docs)


DOCUMENTATION
-------------
See BUILD_AUTOMATION.md for detailed guide


TROUBLESHOOTING
---------------
"Build directory doesn't exist"
  → Run: python build_distribution.py

"Module not found" when running
  → Full rebuild: python build_distribution.py

Watcher not detecting changes
  → Restart watcher, check file is saved
