### Python Environment

Run Python commands from the repository root.

PowerShell:

```powershell
.\venv\Scripts\python.exe -m pytest tests/ -q
```

Bash:

```bash
source venv/Scripts/activate
python -m pytest tests/ -q
```

---

### Server + Tk Client (default)

Starts both the server and the Tk GUI client:

```
python main.py
```

Verbose (debug) mode:

```
python main.py -v
```

File input instead of microphone (for testing):

```
python main.py --input-file=tests/fixtures/en_short.wav -v
```

---

### Server Only

Starts the server without spawning a client. Prints the WebSocket URL and a QR code, then blocks until Ctrl+C.

```
python main.py --server-only --port=62062
```

With a random port:

```
python main.py --server-only
```


Debug mode:

```
python main.py --server-only -v
```

---

### Tauri Client

Run from `src/client/tauri/`. Requires the server to already be running.

Development (hot-reload):

PowerShell:

```powershell
$env:STT_SERVER_URL="ws://127.0.0.1:<port>"
npm run tauri:dev
```

Bash:

```bash
STT_SERVER_URL=ws://127.0.0.1:<port> npm run tauri:dev
```

With file input instead of microphone:

PowerShell:

```powershell
$env:STT_SERVER_URL="ws://127.0.0.1:<port>"
$env:STT_INPUT_FILE="..\..\..\tests\fixtures\en.wav"
npm run tauri:dev
```

Bash:

```bash
STT_SERVER_URL=ws://127.0.0.1:<port> STT_INPUT_FILE=../../../tests/fixtures/en.wav npm run tauri:dev
```

Production build:

```
npm run tauri:build
```

Headless Rust binary (no window, no Node):

PowerShell:

```powershell
cd src/client/tauri/src-tauri
cargo build
.\target\debug\stt-tauri-client.exe --headless --server-url=ws://127.0.0.1:<port>
.\target\debug\stt-tauri-client.exe --headless --server-url=ws://127.0.0.1:<port> --input-file=..\..\..\..\tests\fixtures\en.wav
```

Bash:

```bash
cd src/client/tauri/src-tauri
cargo build
./target/debug/stt-tauri-client --headless --server-url=ws://127.0.0.1:<port>
./target/debug/stt-tauri-client --headless --server-url=ws://127.0.0.1:<port> --input-file=../../../../tests/fixtures/en.wav
```

---

### Testing

Python server application tests (all except GUI):

PowerShell:

```powershell
.\venv\Scripts\python.exe -m pytest tests/ -q
```

Bash:

```bash
source venv/Scripts/activate
python -m pytest tests/ -q
```

Tauri frontend tests (Vitest):

```
cd src/client/tauri
npm test
```

Tauri Rust tests:

```
cd src/client/tauri/src-tauri
cargo test
```

