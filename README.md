# Local Multimodal AI File Analyzer

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU%2FGPU-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Transformers-Offline%20Cache-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A zero-API, 100% offline multimodal AI inference pipeline built in Python that processes **images, documents, long-form audio, and videos** locally using open-source Hugging Face Transformers models.

---

## 1. Problem Statement

Most multimodal AI applications rely on paid cloud APIs (such as OpenAI, Anthropic, or Gemini) to analyze mixed media. This creates four major bottlenecks for developers, researchers, and privacy-sensitive workflows:
- **Recurring API Costs & Rate Limits:** Processing batches of high-resolution images, 1-hour audio recordings, or multi-minute videos quickly exhausts free tiers and incurs per-token/per-minute charges.
- **Data Privacy Exposure:** Uploading sensitive local documents, internal recordings, or private images to third-party servers compromises data confidentiality.
- **Fragmented Tooling:** Users typically need separate tools for OCR/vision analysis, audio speech-to-text, video scene summarization, and PDF/DOCX parsing.
- **Internet Dependency:** Cloud-dependent pipelines fail completely in offline or air-gapped environments.

---

## 2. The Solution

**Local Multimodal AI File Analyzer** unifies four distinct media pipelines into a single, self-contained Python CLI application that runs **100% offline** from cached local model weights:
1. **Batch Vision & Line-by-Line OCR:** Analyzes up to 5 images simultaneously, producing structured visual descriptions alongside exact top-to-bottom transcription of all visible text, numbers, dates, and IDs.
2. **Long-Form Audio Transcription & Summarization:** Normalizes diverse audio containers into `16kHz` mono PCM WAV streams and transcribes recordings up to **1 hour** long with multilingual support (`English`, `Telugu`, `Hindi`, or Auto-detect), followed by an automated executive summary.
3. **Dual-Stream Video Understanding:** Samples evenly spaced visual key frames across videos up to **5 minutes** long, transcribes the embedded audio track in parallel, and fuses both streams into a unified scene-by-scene and spoken summary.
4. **Local Document Intelligence:** Extracts clean text from `.pdf`, `.docx`, `.txt`, `.md`, and `.csv` files and generates structured summaries without external network calls.

---

## 3. System Architecture

The application follows a modular **4-stage local inference architecture** orchestrated by `ai_multimodal_analyzer.py`:

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                        1. INPUT INGESTION & SANITIZATION                     │
│  Drag-and-Drop CLI / Folder Path Resolver (clean_user_input_paths)           │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                     2. MEDIA CLASSIFICATION & VALIDATION                     │
│  Extension & FFmpeg Stream Inspector (detect_media_type)                     │
│  Enforces Limits: ≤ 5 Images | ≤ 3600s Audio | ≤ 300s Video                  │
└───────┬──────────────────────┬──────────────────────┬────────────────────────┘
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐      ┌─────────┐
│ IMAGE PIPELINE│      │AUDIO PIPELINE │      │VIDEO PIPELINE │      │DOC PIPE │
│               │      │               │      │               │      │         │
│ PIL RGB Conv. │      │ FFmpeg 16kHz  │      │ Key-Frame     │      │ PyPDF / │
│      ▼        │      │ Mono WAV      │      │ Extractor     │      │ docx /  │
│ Qwen2.5-VL-3B │      │      ▼        │      │ (≤ 5 Frames)  │      │ Text    │
│ Vision + OCR  │      │ Whisper-Small │      │      +        │      │ Parser  │
│               │      │ (30s Chunking)│      │ Audio Track   │      │         │
└───────┬───────┘      └───────┬───────┘      └───────┬───────┘      └────┬────┘
        │                      │                      │                   │
        │                      └──────────┬───────────┴───────────────────┘
        │                                 ▼
        │              ┌──────────────────────────────────────┐
        │              │    3. LOCAL TEXT SUMMARIZATION       │
        │              │    Qwen2.5-0.5B-Instruct Pipeline    │
        │              │    (Structured Key-Point Synthesis)  │
        │              └──────────────────┬───────────────────┘
        ▼                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        4. STRUCTURED TERMINAL OUTPUT                         │
│  Metadata Report + Line-by-Line OCR / Timestamps / Transcript / Summary      │
└──────────────────────────────────────────────────────────────────────────────┘
---

## 4. Model Stack & Supported Modalities

| Modality | Supported File Formats | Hard Limits | Local AI Model / Engine | Processing Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **Images** | `.jpg`, `.jpeg`, `.png`, `.webp`, `.avif`, `.tiff`, `.tif`, `.bmp`, `.gif`, `.ico` | Max **5 images** per batch | `Qwen/Qwen2.5-VL-3B-Instruct` | Dynamic resolution (`256x28x28` to `768x28x28` pixels) with deterministic greedy decoding (`do_sample=False`) |
| **Audio** | `.mp3`, `.wav`, `.m4a`, `.aac`, `.ogg`, `.flac`, `.opus`, `.wma` | Max **1 hour** (`3,600s`) | `openai/whisper-small` + `Qwen/Qwen2.5-0.5B-Instruct` | FFmpeg `16kHz` mono normalization + 30s chunking with 5s stride overlap |
| **Video** | `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, `.flv`, `.3gp`, `.wmv` | Max **5 minutes** (`300s`) | `Qwen2.5-VL-3B-Instruct` + `whisper-small` + `Qwen2.5-0.5B-Instruct` | Uniform temporal key-frame sampling (1 frame per ~15s, up to 5 frames) + parallel audio transcription |
| **Documents** | `.pdf`, `.docx`, `.txt`, `.md`, `.csv` | First **2,000 words** per summary | `Qwen/Qwen2.5-0.5B-Instruct` | Native page/paragraph extraction via `pypdf` and `python-docx` followed by instruction-tuned summarization |

---

## 5. Key Engineering Details

- **Global Session-Level Model Caching:** Model weights and processors (`_VLM_MODEL`, `_VLM_PROCESSOR`, `_WHISPER_PIPE`, `_TEXT_PIPE`) are lazy-loaded once into global memory and reused across subsequent files in the same session, eliminating repeated cold-start latency.
- **Zero-Network Execution:** Configured with `torch.no_grad()` inference mode and local Hugging Face cache loading so all vision, speech, and language tasks execute completely offline.
- **Self-Contained FFmpeg Bundling:** Uses `imageio-ffmpeg` to invoke a bundled FFmpeg binary directly via `subprocess`, removing external system PATH installation dependencies on Windows and Linux.
- **Automatic Temporary Cleanup:** All intermediate `16kHz` WAV audio conversions and extracted video frames are written inside `tempfile.TemporaryDirectory()` contexts that automatically purge upon completion.
- **Shell-Safe Path Sanitization:** `clean_user_input_paths()` strips Windows PowerShell drag-and-drop prefixes (`& '...'`), nested quotes, and multi-file tokens while supporting whole-directory batch ingestion.

---

## 6. Project Structure

    ai-multimodal-application/
    ├── ai_multimodal_analyzer.py   # Core end-to-end multimodal CLI pipeline
    └── README.md                   # Architecture, specifications, and setup documentation

---

## 7. Installation & Quick Start

### Prerequisites
- **Python 3.10+**
- ~8–10 GB free disk space for local Hugging Face model cache (`Qwen2.5-VL-3B-Instruct`, `whisper-small`, `Qwen2.5-0.5B-Instruct`)

### Step 1: Clone the Repository

    git clone https://github.com/CHgopi07/ai-multimodal-application.git
    cd ai-multimodal-application

### Step 2: Set Up a Virtual Environment & Install Dependencies

    python -m venv venv
    # Windows PowerShell:
    .\venv\Scripts\Activate.ps1
    # Linux / macOS:
    # source venv/bin/activate

    pip install torch transformers qwen-vl-utils pillow librosa imageio-ffmpeg pypdf python-docx numpy

### Step 3: Run the Analyzer

    python ai_multimodal_analyzer.py

When prompted with `Drag & drop file(s) or folder here:`, paste or drag any supported image batch (up to 5), document, audio file, or video file into the terminal.

---

## 8. Future Roadmap
- [ ] Add interactive **Streamlit** web interface alongside the CLI
- [ ] Support `4-bit` / `8-bit` quantization (`bitsandbytes`) and CUDA GPU acceleration flags for faster VLM inference
- [ ] Export structured JSON and Markdown reports automatically for batch runs
