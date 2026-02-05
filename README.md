# liquid-audio-chat (Rust)

Low-latency, low-resource LFM2.5-Audio chat client. Rust port of `liquid_audio_chat.py`.

## Build

```bash
cargo build --release
```

Release build enables LTO, single codegen unit, and strip for a small, fast binary.

## Run

```bash
./target/release/liquid-audio-chat [OPTIONS]
```

Options:

- `--base-url URL` — Server base URL (default: `http://127.0.0.1:8080/v1`)
- `--mode asr|tts|interleaved` — Initial mode (default: `interleaved`)
- `--max-tokens N` — Max tokens (default: 512)
- `--no-audio-playback` — Disable speaker playback (no audio out)

## Commands (same as Python)

- `/mode <asr|tts|interleaved>` — Switch mode
- `/reset` — Reset context (interleaved only)
- `/record` — Record from mic then transcribe/process (Enter to stop)
- `/wav <path>` — Load WAV and transcribe/process
- `/help` — Help
- `/quit` or `/exit` — Exit

## Design (efficiency / low latency)

- **Audio out**: cpal output stream with a lock-free channel; stream task pushes decoded PCM, callback pulls with minimal buffering and a small leftover buffer to avoid underruns.
- **Audio in**: cpal input stream; record in a thread until Enter, then encode to WAV.
- **Streaming**: NDJSON/SSE parsed in a tight loop; base64 audio decoded and pushed to the playback channel immediately (no extra buffering).
- **Single binary**: no interpreter; release profile with LTO and `codegen-units=1` for speed and size.
