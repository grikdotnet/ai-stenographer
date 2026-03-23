## Commands

Activate venv before running any Python commands:

```bash
source venv/Scripts/activate
```

---

### Server + Tk Client (default)

Starts both the server and the Tk GUI client:

```bash
python main.py
```

Verbose (debug) mode:

```bash
python main.py -v
```

File input instead of microphone (for testing):

```bash
python main.py --input-file=tests/fixtures/en_short.wav -v
```

---

### Server Only

Starts the server without spawning a client. Prints the WebSocket URL and a QR code, then blocks until Ctrl+C.

```bash
python main.py --server-only --port=62062
```

With a random port:

```bash
python main.py --server-only
```


Debug mode:

```bash
python main.py --server-only -v
```

> Models must be pre-provisioned — in `--server-only` mode there is no download GUI. The server exits with code 1 if models are missing.

---

### Tauri Client

Run from `src/client/tauri/`. Requires the server to already be running (`--server-only`).

Development (hot-reload):

```bash
STT_SERVER_URL=ws://127.0.0.1:<port> npm run tauri:dev
```

With file input instead of microphone:

```bash
STT_SERVER_URL=ws://127.0.0.1:<port> STT_INPUT_FILE=tests/fixtures/en.wav npm run tauri:dev
```

Production build:

```bash
npm run tauri:build
```

Headless Rust binary (no window, no Node):

```bash
cd src/client/tauri/src-tauri
cargo build
.\target\debug\stt-tauri-client.exe --headless --server-url=ws://127.0.0.1:<port>
.\target\debug\stt-tauri-client.exe --headless --server-url=ws://127.0.0.1:<port> --input-file=..\..\..\tests\fixtures\en.wav
```

---

### WinUI Client

Run from `src/client/winui/`. Requires the server to already be running (`--server-only`).

Build:

```bash
dotnet build SttClient.sln
```

Run (debug build):

```bash
.\build\SttClient\SttClient.exe --server-url=ws://127.0.0.1:<port>
```

---

### Testing

Python tests (all except GUI):

```bash
python -m pytest tests/
```

Python GUI tests (Tk — kept separate due to `mainloop` limitations):

```bash
python -m pytest tests/gui -m gui
```

Tauri frontend tests (Vitest):

```bash
cd src/client/tauri
npm test
```

Tauri Rust tests:

```bash
cd src/client/tauri/src-tauri
cargo test
```

WinUI tests:

```bash
cd src/client/winui
dotnet test SttClient.sln
```
