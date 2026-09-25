# AutoShorts Setup & Usage Guide

[AutoShorts](https://github.com/JayWebtech/autoshorts) (v0.1.3) is a local desktop app (Tauri 2 + React + Rust + SQLite) that turns long videos/podcasts into ranked 9:16 vertical clips with burned-in captions.

**Pipeline:** import media → extract audio (ffmpeg) → transcribe (Whisper/Deepgram) → LLM ranks viral moments (30–90s each) → render cropped H.264 clips.

---

## 1. Install

### Required on every OS
| Tool | Why | Install |
|---|---|---|
| `ffmpeg` + `ffprobe` | audio extract, crop, captions | macOS `brew install ffmpeg` · Win `winget install Gyan.FFmpeg` · Linux `sudo apt install ffmpeg` |

### Optional
| Tool | Enables | Install |
|---|---|---|
| `yt-dlp` | "Import from YouTube" button | `pip install yt-dlp` / `brew install yt-dlp` |
| Whisper | offline transcription | `pip3 install openai-whisper` (or `brew install whisper-cli`) |
| Ollama | offline LLM | https://ollama.com (app auto-installs it on macOS) |

On macOS, if captions get skipped, your ffmpeg lacks `drawtext`: `brew tap homebrew-ffmpeg/ffmpeg && brew install homebrew-ffmpeg/ffmpeg/ffmpeg`.

### Option A: Prebuilt installer (easiest)
Download from [Releases](https://github.com/JayWebtech/autoshorts/releases/tag/autoshorts):
- **macOS:** `aarch64.dmg` (Apple Silicon) or `x64.dmg` (Intel) → drag to Applications → `xattr -cr /Applications/AutoShorts.app` (unsigned).
- **Windows:** `.msi` → SmartScreen: *More info → Run anyway*.
- **Linux:** `sudo dpkg -i autoshorts_*.deb` or `chmod +x autoshorts_*.AppImage && ./autoshorts_*.AppImage`.

### Option B: Build from source
Needs Node 18+, Rust (`rustup`), and on Linux:
```bash
sudo apt install libwebkit2gtk-4.1-dev libgtk-3-dev libayatana-appindicator3-dev librsvg2-dev libssl-dev pkg-config build-essential
```
Then:
```bash
git clone https://github.com/JayWebtech/autoshorts && cd autoshorts
cp .env.example .env        # optional, see §4
npm install
npm run tauri:dev           # dev mode with hot reload
npm run tauri:build         # installers -> src-tauri/target/release/bundle/
```
Verified: builds cleanly on Ubuntu 24.04 (Rust release build ~4 min) and launches to the onboarding screen.

---

## 2. First launch: onboarding

Pick a mode:

**Fully Offline (Ollama + Whisper)** is free and private, but lower quality.
- Choose model: `llama3.2 3B`, `qwen2.5 3B`, or `qwen2.5 7B`; the app pulls it through Ollama (`localhost:11434`) with a progress bar.
- Install Whisper when prompted.
- ⚠️ The author says small local models often give bad timestamps and clips that are too short. Use them only for privacy.

**Cloud APIs** (recommended)
- **Transcription:** Deepgram key ([console.deepgram.com](https://console.deepgram.com), has free credit).
- **Moment detection:** DeepSeek (cheapest, <$0.001/run, recommended), Claude (best hooks, ~$0.01–0.05/run), or Groq.
- Click **Save & Start**.

More providers (Gemini, OpenAI, OpenRouter) are available later in **API Settings**.

---

## 3. Daily workflow

1. **Import:** click **Import Media** (local video/audio) or **Import from YouTube** (paste URL; yt-dlp checks the license, flags non-Creative-Commons videos, and downloads to `~/Downloads/AutoShorts_<id>.mp4`).
2. **Choose caption style:** `modern-box` (default), `classic-outline`, `minimal-shadow`, `vibrant-yellow-box`, `vibrant-cyan`, `vibrant-green`, `vibrant-red`.
3. **Pick engines** in the project view: Transcription (Local Whisper / Deepgram) and LLM (Ollama / Claude / DeepSeek / Gemini / OpenAI / OpenRouter / Groq).
4. **Transcribe:** audio is extracted to 16 kHz mono WAV, then transcribed with word-level timestamps.
5. **Find moments:** the LLM returns candidates with `start`, `end`, `score`, `hook`, and `rationale`, ranked by score, each 30–90s long with a strong 3-second hook.
6. **Select clips:** use the slider to set how many top candidates to cut, or toggle cards one by one.
7. **Cut Selected:** each clip is center-cropped to 9:16, captions are burned in (libx264 CRF 18, AAC 192k), and saved to:
   ```
   ~/Documents/AutoShorts/<source-name>/clips/clip-01_flat.mp4
   ```
   If caption rendering fails, the clip is re-rendered without captions and a warning appears. An `.srt` file is also written.
8. **Manage projects:** open, rename, or delete projects from the dashboard (**All Projects**).

---

## 4. Configuration reference

UI settings are stored in the webview's localStorage and override `.env`. The `.env` file (repo root, loaded at startup) sets fallback values:

| Variable | Purpose / default |
|---|---|
| `DEEPGRAM_API_KEY` | cloud transcription |
| `LLM_PROVIDER` | `deepseek`, `claude`, `local`, `gemini`, `openai`, `openrouter`, `groq` |
| `DEEPSEEK_API_KEY` / `DEEPSEEK_MODEL` | DeepSeek |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` | default `claude-3-5-sonnet-latest` (consider setting a current model) |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | default `gemini-2.5-flash` |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | default `gpt-4o-mini` |
| `OPENROUTER_API_KEY` / `OPENROUTER_MODEL` | OpenRouter |
| `GROQ_API_KEY` / `GROQ_MODEL` | default `llama-3.3-70b-versatile` |
| `OLLAMA_MODEL` | local model name |

- **Change keys/engines:** click the gear icon (**API Settings**).
- **Start over:** click **Reset App Configuration & Onboarding**. This clears localStorage, including all saved keys.

---

## 5. Tips & gotchas
- Keys in the UI are stored **in plain text** in localStorage. Don't use it on shared machines.
- The crop is always **center-only** (no face tracking), so it suits single speakers framed in the middle.
- Clip length is fixed at 30–90s by the prompt. Change it by editing `src-tauri/src/llm.rs` and rebuilding.
- Warning banners in the UI mean a tool or key is missing (ffmpeg, yt-dlp, Whisper, Ollama, keys).
- Only upload clips you have rights to. The YouTube license check is advisory.
