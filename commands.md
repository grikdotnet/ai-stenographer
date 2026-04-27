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

### Python Server

Starts the headless Python WebSocket server. It prints a parseable `Server listening on ws://...` line and a QR code, then blocks until Ctrl+C.

```
python main.py
```

Verbose (debug) mode:

```
python main.py -v
```

Fixed port:

```
python main.py --port=62062
```

Provision the model before serving:

```
python main.py --download-model
```

`--server-only` is still accepted for compatibility but no longer changes behavior.

---

### Tauri Client

Run from `client/tauri/`.

Development with a Tauri-owned Python server:

PowerShell:

```powershell
cd client/tauri
npm run tauri:dev
```

Bash:

```bash
cd client/tauri
npm run tauri:dev
```

Development with an external Python server:

PowerShell:

```powershell
python main.py --port=62062
cd client/tauri
$env:STT_SERVER_URL="ws://127.0.0.1:62062"
npm run tauri:dev
```

Bash:

```bash
python main.py --port=62062
cd client/tauri
STT_SERVER_URL=ws://127.0.0.1:62062 npm run tauri:dev
```

With file input instead of microphone:

PowerShell:

```powershell
$env:STT_INPUT_FILE="..\..\tests\fixtures\en.wav"
npm run tauri:dev
```

Bash:

```bash
STT_INPUT_FILE=../../tests/fixtures/en.wav npm run tauri:dev
```

Production build:

```
npm run tauri:build
```

Headless Rust binary with a Tauri-owned Python server:

PowerShell:

```powershell
cd client/tauri/src-tauri
cargo build
.\target\debug\stt-tauri-client.exe --headless
.\target\debug\stt-tauri-client.exe --headless --input-file=..\..\..\tests\fixtures\en.wav
```

Bash:

```bash
cd client/tauri/src-tauri
cargo build
./target/debug/stt-tauri-client --headless
./target/debug/stt-tauri-client --headless --input-file=../../../tests/fixtures/en.wav
```

Headless Rust binary with an external server:

PowerShell:

```powershell
cd client/tauri/src-tauri
cargo build
.\target\debug\stt-tauri-client.exe --headless --server-url=ws://127.0.0.1:<port>
.\target\debug\stt-tauri-client.exe --headless --server-url=ws://127.0.0.1:<port> --input-file=..\..\..\tests\fixtures\en.wav
```

Bash:

```bash
cd client/tauri/src-tauri
cargo build
./target/debug/stt-tauri-client --headless --server-url=ws://127.0.0.1:<port>
./target/debug/stt-tauri-client --headless --server-url=ws://127.0.0.1:<port> --input-file=../../../tests/fixtures/en.wav
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
cd client/tauri
npm test
```

Tauri Rust tests:

```
cd client/tauri/src-tauri
cargo test
```

