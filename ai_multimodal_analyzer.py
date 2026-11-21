import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Suppress harmless Hugging Face deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import docx
import imageio_ffmpeg
import librosa
import numpy as np
import pypdf
import torch
from PIL import Image, ExifTags
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    pipeline,
)

# Limit CPU thread contention on Windows systems
CPU_THREADS = min(4, os.cpu_count() or 1)
torch.set_num_threads(CPU_THREADS)


# ==============================================================================
# 1. CONFIGURATION & CONSTANTS
# ==============================================================================
@dataclass(frozen=True)
class AppConfig:
    """Centralized configuration for supported file formats, limits, and models."""

    max_images_per_batch: int = 5
    max_audio_seconds: int = 3600   # 1 hour
    max_video_seconds: int = 300    # 5 minutes
    max_summary_input_words: int = 2500

    image_exts: Tuple[str, ...] = (
        ".jpg", ".jpeg", ".png", ".webp", ".avif",
        ".tiff", ".tif", ".bmp", ".gif", ".ico",
    )
    doc_exts: Tuple[str, ...] = (
        ".pdf", ".docx", ".txt", ".md", ".csv",
    )
    audio_exts: Tuple[str, ...] = (
        ".mp3", ".wav", ".m4a", ".aac",
        ".ogg", ".flac", ".opus", ".wma",
    )
    video_exts: Tuple[str, ...] = (
        ".mp4", ".mkv", ".mov", ".avi",
        ".webm", ".flv", ".3gp", ".wmv",
    )

    vlm_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    whisper_model_id: str = "openai/whisper-small"
    text_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"

    supported_languages: Dict[str, Optional[str]] = field(default_factory=lambda: {
        "": None,
        "auto": None,
        "telugu": "telugu",
        "te": "telugu",
        "english": "english",
        "en": "english",
        "hindi": "hindi",
        "hi": "hindi",
        "tamil": "tamil",
        "ta": "tamil",
        "kannada": "kannada",
        "kn": "kannada",
        "malayalam": "malayalam",
        "ml": "malayalam",
    })


CONFIG = AppConfig()
USE_CUDA = torch.cuda.is_available()
DEVICE_STR = "cuda" if USE_CUDA else "cpu"
PIPE_DEVICE = 0 if USE_CUDA else -1


# ==============================================================================
# 2. DATA STRUCTURES FOR MEDIA & ANALYSIS RESULTS
# ==============================================================================
@dataclass
class MediaMetadata:
    """Stores inspected technical metadata for a single input file."""

    file_path: str
    file_name: str
    extension: str
    media_type: str
    format_name: str
    size_bytes: int
    modified_time: str
    width: Optional[int] = None
    height: Optional[int] = None
    color_mode: Optional[str] = None
    duration_seconds: Optional[float] = None
    has_audio: bool = False
    has_video: bool = False
    audio_codec: Optional[str] = None
    video_codec: Optional[str] = None
    sample_rate: Optional[str] = None
    page_count: Optional[int] = None
    extra_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    @property
    def formatted_duration(self) -> str:
        return format_duration_hms(self.duration_seconds)


@dataclass
class AnalysisResult:
    """Stores complete analysis output for display and report exporting."""

    metadata: MediaMetadata
    processing_time_sec: float
    language_selected: Optional[str] = None
    custom_question: Optional[str] = None
    visual_description: Optional[str] = None
    extracted_text: Optional[str] = None
    transcript: Optional[str] = None
    visual_timeline: Optional[str] = None
    ai_summary: Optional[str] = None
    custom_answer: Optional[str] = None
    error_message: Optional[str] = None


# ==============================================================================
# 3. GENERAL FORMATTING & PATH UTILITIES
# ==============================================================================
def format_duration_hms(seconds: Optional[float]) -> str:
    """Formats floating-point seconds into HH:MM:SS."""
    if seconds is None or seconds < 0:
        return "Unknown"
    total = int(round(seconds))
    hrs, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def clean_user_input_paths(raw_input: str) -> List[str]:
    """Parses PowerShell/Windows drag-and-drop strings into valid file paths."""
    cleaned = raw_input.strip()
    if cleaned.startswith("& "):
        cleaned = cleaned[2:].strip()

    try:
        tokens = shlex.split(cleaned, posix=False)
    except ValueError:
        tokens = [cleaned]

    resolved_files: List[str] = []
    seen_paths = set()

    for token in tokens:
        path = token.strip().strip('"').strip("'")
        if not path:
            continue

        normalized = os.path.normpath(path)
        if os.path.isdir(normalized):
            for entry in sorted(os.listdir(normalized)):
                full_entry = os.path.join(normalized, entry)
                if os.path.isfile(full_entry) and full_entry not in seen_paths:
                    seen_paths.add(full_entry)
                    resolved_files.append(full_entry)
        elif os.path.isfile(normalized) and normalized not in seen_paths:
            seen_paths.add(normalized)
            resolved_files.append(normalized)

    return resolved_files


def normalize_language_input(raw_lang: str) -> Optional[str]:
    """Normalizes language input or returns raw string if user typed another Whisper language."""
    key = raw_lang.strip().lower()
    if key in CONFIG.supported_languages:
        return CONFIG.supported_languages[key]
    return key if key else None


def print_banner() -> None:
    """Prints application startup header and hardware diagnostics."""
    print("=======================================================================")
    print("     Local Multimodal AI File Analyzer (100% Offline & API-Free)       ")
    print("=======================================================================")
    gpu_name = torch.cuda.get_device_name(0) if USE_CUDA else "None (Using CPU)"
    print(f"Hardware Mode : {DEVICE_STR.upper()} | GPU: {gpu_name} | CPU Threads: {CPU_THREADS}")
    print("Supported Inputs:")
    print(f"  - Images    : Up to {CONFIG.max_images_per_batch} files ({', '.join(e.upper() for e in CONFIG.image_exts)})")
    print(f"  - Documents : {', '.join(e.upper() for e in CONFIG.doc_exts)}")
    print(f"  - Audio     : <= 1 hr ({', '.join(e.upper() for e in CONFIG.audio_exts)})")
    print(f"  - Video     : <= 5 min ({', '.join(e.upper() for e in CONFIG.video_exts)})")
    print("=======================================================================\n")


# ==============================================================================
# 4. MEDIA STREAM & METADATA INSPECTOR
# ==============================================================================
class MediaInspector:
    """Inspects files on disk using PIL, PyPDF, Python-DOCX, and bundled FFmpeg."""

    def __init__(self) -> None:
        self.ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    def probe_av_streams(self, file_path: str) -> Dict[str, Any]:
        """Runs FFmpeg probe to extract duration, audio/video streams, and codecs."""
        cmd = [self.ffmpeg_exe, "-i", file_path]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
        )
        stderr = proc.stderr

        duration = None
        dur_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
        if dur_match:
            hrs = float(dur_match.group(1))
            mins = float(dur_match.group(2))
            secs = float(dur_match.group(3))
            duration = hrs * 3600 + mins * 60 + secs

        audio_match = re.search(r"Stream #\d+:\d+.*Audio:\s*([^,\r\n]+)", stderr)
        video_match = re.search(r"Stream #\d+:\d+.*Video:\s*([^,\r\n]+)", stderr)
        sr_match = re.search(r"(\d+)\s*Hz", stderr)

        return {
            "duration": duration,
            "has_audio": bool(audio_match),
            "has_video": bool(video_match),
            "audio_codec": audio_match.group(1).strip() if audio_match else None,
            "video_codec": video_match.group(1).strip() if video_match else None,
            "sample_rate": f"{sr_match.group(1)} Hz" if sr_match else None,
        }

    def inspect_file(self, file_path: str) -> MediaMetadata:
        """Builds a complete MediaMetadata object for any input file."""
        stat = os.stat(file_path)
        file_name = os.path.basename(file_path)
        ext = os.path.splitext(file_name)[1].lower()
        mod_time = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")

        base_kwargs = {
            "file_path": file_path,
            "file_name": file_name,
            "extension": ext,
            "size_bytes": stat.st_size,
            "modified_time": mod_time,
        }

        if ext in CONFIG.image_exts:
            with Image.open(file_path) as img:
                exif_summary = {}
                raw_exif = img.getexif()
                if raw_exif:
                    for tag_id, val in raw_exif.items():
                        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                        if tag_name in {"Make", "Model", "DateTime", "Software"}:
                            exif_summary[tag_name] = str(val)

                return MediaMetadata(
                    media_type="image",
                    format_name=(img.format or ext.lstrip(".")).upper(),
                    width=img.width,
                    height=img.height,
                    color_mode=img.mode,
                    extra_info=exif_summary,
                    **base_kwargs,
                )

        if ext in CONFIG.doc_exts:
            page_count = None
            if ext == ".pdf":
                try:
                    reader = pypdf.PdfReader(file_path)
                    page_count = len(reader.pages)
                except Exception:
                    page_count = None

            return MediaMetadata(
                media_type="document",
                format_name=ext.lstrip(".").upper(),
                page_count=page_count,
                **base_kwargs,
            )

        if ext in CONFIG.audio_exts:
            probe = self.probe_av_streams(file_path)
            return MediaMetadata(
                media_type="audio",
                format_name=ext.lstrip(".").upper(),
                duration_seconds=probe["duration"],
                has_audio=True,
                has_video=False,
                audio_codec=probe["audio_codec"],
                sample_rate=probe["sample_rate"],
                **base_kwargs,
            )

        if ext in CONFIG.video_exts:
            probe = self.probe_av_streams(file_path)
            # Automatically classify audio-only MP4/MKV containers as audio
            if probe["has_audio"] and not probe["has_video"]:
                return MediaMetadata(
                    media_type="audio",
                    format_name=f"{ext.lstrip('.').upper()} (Audio-Only)",
                    duration_seconds=probe["duration"],
                    has_audio=True,
                    has_video=False,
                    audio_codec=probe["audio_codec"],
                    sample_rate=probe["sample_rate"],
                    **base_kwargs,
                )

            return MediaMetadata(
                media_type="video",
                format_name=ext.lstrip(".").upper(),
                duration_seconds=probe["duration"],
                has_audio=probe["has_audio"],
                has_video=probe["has_video"],
                audio_codec=probe["audio_codec"],
                video_codec=probe["video_codec"],
                sample_rate=probe["sample_rate"],
                **base_kwargs,
            )

        return MediaMetadata(
            media_type="unsupported",
            format_name=ext or "UNKNOWN",
            **base_kwargs,
        )


# ==============================================================================
# 5. OFFLINE AI MODEL MANAGER (LAZY LOADING & SESSION CACHING)
# ==============================================================================
class LocalModelManager:
    """Manages lazy loading and caching of local Hugging Face models."""

    def __init__(self) -> None:
        self._vlm_model = None
        self._vlm_processor = None
        self._whisper_pipe = None
        self._text_pipe = None

    def get_vision_language_model(self):
        """Loads Qwen2.5-VL-3B-Instruct once and caches it for the session."""
        if self._vlm_model is None or self._vlm_processor is None:
            print(
                f"[Model Loader] Loading Vision-Language model ({CONFIG.vlm_model_id}) "
                f"on {DEVICE_STR.upper()}...",
                flush=True,
            )
            self._vlm_processor = AutoProcessor.from_pretrained(
                CONFIG.vlm_model_id,
                size={
                    "shortest_edge": 128 * 28 * 28,
                    "longest_edge": 256 * 28 * 28,
                },
            )
            model_dtype = torch.float16 if USE_CUDA else torch.float32
            self._vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                CONFIG.vlm_model_id,
                dtype=model_dtype,
                device_map=DEVICE_STR,
                attn_implementation="sdpa",
            )
            self._vlm_model.eval()
        return self._vlm_model, self._vlm_processor

    def get_whisper_pipeline(self):
        """Loads Whisper-small once with 30s chunking for long audio files."""
        if self._whisper_pipe is None:
            print(
                f"[Model Loader] Loading Speech-to-Text model ({CONFIG.whisper_model_id}) "
                f"on {DEVICE_STR.upper()}...",
                flush=True,
            )
            self._whisper_pipe = pipeline(
                "automatic-speech-recognition",
                model=CONFIG.whisper_model_id,
                chunk_length_s=30,
                stride_length_s=5,
                device=PIPE_DEVICE,
            )
        return self._whisper_pipe

    def get_text_pipeline(self):
        """Loads Qwen2.5-0.5B-Instruct once for summarization and Q&A."""
        if self._text_pipe is None:
            print(
                f"[Model Loader] Loading Text/Summary model ({CONFIG.text_model_id}) "
                f"on {DEVICE_STR.upper()}...",
                flush=True,
            )
            model_dtype = torch.float16 if USE_CUDA else torch.float32
            self._text_pipe = pipeline(
                "text-generation",
                model=CONFIG.text_model_id,
                dtype=model_dtype,
                device=PIPE_DEVICE,
            )
        return self._text_pipe


# ==============================================================================
# 6. MULTIMODAL ANALYSIS PIPELINES
# ==============================================================================
class MultimodalAnalyzer:
    """Executes specialized offline AI pipelines for images, docs, audio, and video."""

    def __init__(self, model_manager: LocalModelManager) -> None:
        self.models = model_manager
        self.ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # --------------------------------------------------------------------------
    # A. Vision Pipeline (Qwen2.5-VL)
    # --------------------------------------------------------------------------
    def analyze_pil_image(
        self,
        pil_image: Image.Image,
        prompt_text: Optional[str] = None,
        max_tokens: int = 512,
        max_dim: int = 768,
    ) -> str:
        """Runs Qwen2.5-VL inference on a PIL image with safe resolution caps."""
        model, processor = self.models.get_vision_language_model()

        img_copy = pil_image.convert("RGB")
        img_copy.thumbnail((max_dim, max_dim))

        default_prompt = (
            "Analyze this image thoroughly:\n"
            "1. Visual Overview: Describe what this image is, its layout, colors, and main subject.\n"
            "2. Line-by-Line Details: Transcribe EVERY visible line of text, number, label, amount, "
            "date, time, and ID in exact top-to-bottom order without skipping anything."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_copy},
                    {"type": "text", "text": prompt_text or default_prompt},
                ],
            }
        ]

        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(DEVICE_STR)

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )

        trimmed_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0].strip()

    def process_image(self, meta: MediaMetadata, custom_question: Optional[str] = None) -> AnalysisResult:
        """Analyzes an image file and optionally answers a custom user question."""
        start_t = time.time()
        with Image.open(meta.file_path) as img:
            description = self.analyze_pil_image(img, max_tokens=512, max_dim=768)
            custom_ans = None
            if custom_question:
                print("[Status] Answering your custom question about the image...", flush=True)
                custom_ans = self.analyze_pil_image(
                    img,
                    prompt_text=custom_question,
                    max_tokens=300,
                    max_dim=768,
                )

        return AnalysisResult(
            metadata=meta,
            processing_time_sec=round(time.time() - start_t, 2),
            custom_question=custom_question,
            visual_description=description,
            custom_answer=custom_ans,
        )

    # --------------------------------------------------------------------------
    # B. Document Pipeline (PDF, DOCX, TXT, MD, CSV)
    # --------------------------------------------------------------------------
    def extract_document_text(self, meta: MediaMetadata) -> str:
        """Extracts structured plain text from supported document formats."""
        ext = meta.extension

        if ext == ".pdf":
            reader = pypdf.PdfReader(meta.file_path)
            pages = []
            for idx, page in enumerate(reader.pages, start=1):
                txt = (page.extract_text() or "").strip()
                if txt:
                    pages.append(f"--- Page {idx} ---\n{txt}")
            return "\n\n".join(pages).strip()

        if ext == ".docx":
            doc = docx.Document(meta.file_path)
            blocks = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    cells = [c.text.strip() for c in row.cells if c.text.strip()]
                    if cells:
                        blocks.append(" | ".join(cells))
            return "\n".join(blocks).strip()

        with open(meta.file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()

    def process_document(self, meta: MediaMetadata, custom_question: Optional[str] = None) -> AnalysisResult:
        """Extracts document text, generates a summary, and answers optional Q&A."""
        start_t = time.time()
        print("[Status] Extracting text from document...", flush=True)
        raw_text = self.extract_document_text(meta)
        summary = self.summarize_text(raw_text, content_label="document")
        custom_ans = None
        if custom_question and raw_text:
            custom_ans = self.answer_question_from_text(raw_text, custom_question, "document")

        return AnalysisResult(
            metadata=meta,
            processing_time_sec=round(time.time() - start_t, 2),
            custom_question=custom_question,
            extracted_text=raw_text,
            ai_summary=summary,
            custom_answer=custom_ans,
        )

    # --------------------------------------------------------------------------
    # C. Speech-to-Text Pipeline (Whisper)
    # --------------------------------------------------------------------------
    def transcribe_audio_track(self, file_path: str, language: Optional[str] = None) -> str:
        """Normalizes audio to 16kHz mono WAV and transcribes speech via Whisper."""
        whisper_pipe = self.models.get_whisper_pipeline()
        print("[Status] Extracting and transcribing speech track...", flush=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = os.path.join(tmp_dir, "normalized_16k.wav")
            cmd = [
                self.ffmpeg_exe,
                "-y",
                "-i", file_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                wav_path,
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

            audio_array, _ = librosa.load(wav_path, sr=16000, mono=True)
            if audio_array.size == 0:
                return "[No spoken audio detected]"

            generate_kwargs: Dict[str, Any] = {"task": "transcribe"}
            if language:
                generate_kwargs["language"] = language

            result = whisper_pipe(
                {"array": audio_array.astype(np.float32), "sampling_rate": 16000},
                generate_kwargs=generate_kwargs,
            )
            return result.get("text", "").strip() or "[No speech recognized]"

    def process_audio(
        self,
        meta: MediaMetadata,
        language: Optional[str] = None,
        custom_question: Optional[str] = None,
    ) -> AnalysisResult:
        """Transcribes an audio file, summarizes it, and answers optional Q&A."""
        start_t = time.time()
        if meta.duration_seconds and meta.duration_seconds > CONFIG.max_audio_seconds:
            return AnalysisResult(
                metadata=meta,
                processing_time_sec=0.0,
                error_message="Audio file exceeds the 1-hour (3600s) maximum limit.",
            )

        transcript = self.transcribe_audio_track(meta.file_path, language=language)
        summary = self.summarize_text(transcript, content_label="audio transcript")
        custom_ans = None
        if custom_question and transcript:
            custom_ans = self.answer_question_from_text(transcript, custom_question, "audio transcript")

        return AnalysisResult(
            metadata=meta,
            processing_time_sec=round(time.time() - start_t, 2),
            language_selected=language or "Auto-Detect",
            custom_question=custom_question,
            transcript=transcript,
            ai_summary=summary,
            custom_answer=custom_ans,
        )

    # --------------------------------------------------------------------------
    # D. Video Pipeline (Audio-First + Optional Key-Frame Vision)
    # --------------------------------------------------------------------------
    def sample_video_keyframes(self, file_path: str, duration: Optional[float]) -> str:
        """Extracts evenly spaced video frames and describes each scene live."""
        safe_duration = duration if (duration and duration > 0) else 10.0
        max_frames = 5 if USE_CUDA else 2
        num_frames = min(max_frames, max(1, int(safe_duration // 15) + 1))

        timestamps = [
            round((safe_duration * (i + 0.5)) / num_frames, 2)
            for i in range(num_frames)
        ]

        print(f"[Status] Sampling {num_frames} key frame(s) from video...", flush=True)
        scene_notes: List[str] = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            for idx, ts in enumerate(timestamps, start=1):
                print(
                    f"  -> Frame {idx}/{num_frames} at {format_duration_hms(ts)}...",
                    flush=True,
                )
                frame_path = os.path.join(tmp_dir, f"frame_{idx}.jpg")
                cmd = [
                    self.ffmpeg_exe,
                    "-y",
                    "-ss", str(ts),
                    "-i", file_path,
                    "-frames:v", "1",
                    "-q:v", "2",
                    frame_path,
                ]
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                if os.path.exists(frame_path):
                    with Image.open(frame_path) as frame_img:
                        desc = self.analyze_pil_image(
                            frame_img,
                            prompt_text=(
                                "Describe what is happening in this video frame in 1-2 clear sentences "
                                "and read any visible text."
                            ),
                            max_tokens=90,
                            max_dim=512,
                        )
                    note = f"[{format_duration_hms(ts)}] {desc}"
                    print(f"     {note}\n", flush=True)
                    scene_notes.append(note)

        return "\n\n".join(scene_notes) if scene_notes else "No visual frames could be extracted."

    def process_video(
        self,
        meta: MediaMetadata,
        language: Optional[str] = None,
        run_vision: bool = True,
        custom_question: Optional[str] = None,
    ) -> AnalysisResult:
        """Processes video speech first, runs optional key-frame vision, and summarizes."""
        start_t = time.time()
        if meta.duration_seconds and meta.duration_seconds > CONFIG.max_video_seconds:
            return AnalysisResult(
                metadata=meta,
                processing_time_sec=0.0,
                error_message="Video file exceeds the 5-minute (300s) maximum limit.",
            )

        if meta.has_audio:
            transcript = self.transcribe_audio_track(meta.file_path, language=language)
            print("\n[Live Output] Speech Transcript:\n", transcript, "\n", flush=True)
        else:
            transcript = "[No audio track found in video]"

        if run_vision:
            visuals = self.sample_video_keyframes(meta.file_path, meta.duration_seconds)
            combined_context = f"Visual Scene Notes:\n{visuals}\n\nAudio Transcript:\n{transcript}"
        else:
            visuals = "[Skipped Vision frame analysis for fast CPU mode]"
            combined_context = transcript

        summary = self.summarize_text(combined_context, content_label="video")
        custom_ans = None
        if custom_question:
            custom_ans = self.answer_question_from_text(combined_context, custom_question, "video")

        return AnalysisResult(
            metadata=meta,
            processing_time_sec=round(time.time() - start_t, 2),
            language_selected=language or "Auto-Detect",
            custom_question=custom_question,
            transcript=transcript,
            visual_timeline=visuals,
            ai_summary=summary,
            custom_answer=custom_ans,
        )

    # --------------------------------------------------------------------------
    # E. Text Summarization & Custom Q&A (Qwen2.5-0.5B-Instruct)
    # --------------------------------------------------------------------------
    def summarize_text(self, raw_text: str, content_label: str = "content") -> str:
        """Generates a structured summary using Qwen2.5-0.5B-Instruct."""
        cleaned = re.sub(r"\s+", " ", raw_text or "").strip()
        if not cleaned:
            return f"No readable text or speech found in this {content_label}."

        truncated = " ".join(cleaned.split()[:CONFIG.max_summary_input_words])
        generator = self.models.get_text_pipeline()

        print("[Status] Generating overall AI summary...", flush=True)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise, accurate AI assistant. Summarize the key points "
                    "clearly using bullet points."
                ),
            },
            {
                "role": "user",
                "content": f"Provide a clear, structured summary of the following {content_label}:\n\n{truncated}",
            },
        ]

        output = generator(messages, max_new_tokens=250, do_sample=False)
        generated = output[0]["generated_text"]
        return generated[-1]["content"].strip() if isinstance(generated, list) else str(generated).strip()

    def answer_question_from_text(self, context_text: str, question: str, content_label: str) -> str:
        """Answers a specific user question based on extracted document/transcript/video text."""
        cleaned = re.sub(r"\s+", " ", context_text or "").strip()
        truncated = " ".join(cleaned.split()[:CONFIG.max_summary_input_words])
        generator = self.models.get_text_pipeline()

        print("[Status] Answering your custom question...", flush=True)
        messages = [
            {
                "role": "system",
                "content": "Answer the user's question accurately using ONLY the provided content.",
            },
            {
                "role": "user",
                "content": f"Content ({content_label}):\n{truncated}\n\nQuestion: {question}",
            },
        ]

        output = generator(messages, max_new_tokens=250, do_sample=False)
        generated = output[0]["generated_text"]
        return generated[-1]["content"].strip() if isinstance(generated, list) else str(generated).strip()


# ==============================================================================
# 7. REPORT FORMATTER & FILE EXPORTER
# ==============================================================================
class ReportFormatter:
    """Formats AnalysisResult objects for terminal display and Markdown/Text export."""

    @staticmethod
    def build_report_text(result: AnalysisResult) -> str:
        meta = result.metadata
        lines = [
            "================ Analysis Report ================",
            f"File Name       : {meta.file_name}",
            f"File Path       : {meta.file_path}",
            f"File Size       : {meta.size_mb:.2f} MB",
            f"Last Modified   : {meta.modified_time}",
            f"Detected Type   : {meta.media_type.upper()} ({meta.format_name})",
        ]

        if meta.width and meta.height:
            lines.append(f"Resolution      : {meta.width}x{meta.height} px ({meta.color_mode})")
        if meta.page_count is not None:
            lines.append(f"Page Count      : {meta.page_count}")
        if meta.duration_seconds is not None:
            lines.append(f"Duration        : {meta.formatted_duration}")
        if meta.video_codec:
            lines.append(f"Video Codec     : {meta.video_codec}")
        if meta.audio_codec:
            lines.append(f"Audio Codec     : {meta.audio_codec} ({meta.sample_rate or 'Unknown SR'})")
        if result.language_selected:
            lines.append(f"Speech Language : {result.language_selected}")
        if meta.extra_info:
            lines.append(f"EXIF Metadata   : {json.dumps(meta.extra_info)}")

        lines.append(f"Processing Time : {result.processing_time_sec} seconds")
        lines.append("-------------------------------------------------")

        if result.error_message:
            lines.append(f"ERROR: {result.error_message}")
            lines.append("=================================================")
            return "\n".join(lines)

        if result.visual_description:
            lines.append("\n[Visual Overview & Line-by-Line Details]")
            lines.append(result.visual_description)

        if result.extracted_text:
            preview = result.extracted_text[:1500]
            suffix = "\n... [Truncated for display; full text saved in exported report]" if len(result.extracted_text) > 1500 else ""
            lines.append("\n[Extracted Document Text]")
            lines.append(preview + suffix)

        if result.visual_timeline:
            lines.append("\n[Video Visual Timeline]")
            lines.append(result.visual_timeline)

        if result.transcript:
            lines.append("\n[Speech Transcript]")
            lines.append(result.transcript)

        if result.ai_summary:
            lines.append("\n[AI Summary]")
            lines.append(result.ai_summary)

        if result.custom_question and result.custom_answer:
            lines.append(f"\n[Custom Question: {result.custom_question}]")
            lines.append(result.custom_answer)

        lines.append("\n=================================================")
        return "\n".join(lines)

    @staticmethod
    def save_report_to_disk(result: AnalysisResult) -> str:
        """Saves the full un-truncated analysis report next to the source file."""
        meta = result.metadata
        base_dir = os.path.dirname(meta.file_path)
        stem = os.path.splitext(meta.file_name)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(base_dir, f"{stem}_AI_Report_{timestamp}.txt")

        full_report = ReportFormatter.build_report_text(result)
        if result.extracted_text and len(result.extracted_text) > 1500:
            full_report += "\n\n================ Full Extracted Text ================\n" + result.extracted_text

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(full_report)
        return out_path


# ==============================================================================
# 8. INTERACTIVE CLI APPLICATION CONTROLLER
# ==============================================================================
class MultimodalCLIApp:
    """Interactive command-line loop supporting multi-turn analysis and exports."""

    def __init__(self) -> None:
        self.inspector = MediaInspector()
        self.model_manager = LocalModelManager()
        self.analyzer = MultimodalAnalyzer(self.model_manager)

    def handle_batch_images(self, metadata_list: List[MediaMetadata]) -> None:
        """Processes up to 5 images sequentially in a single batch."""
        if len(metadata_list) > CONFIG.max_images_per_batch:
            print(
                f"Error: Maximum {CONFIG.max_images_per_batch} images allowed at once "
                f"(you provided {len(metadata_list)})."
            )
            return

        if not all(m.media_type == "image" for m in metadata_list):
            print("Error: Multiple files in one drop are only supported for images (up to 5).")
            return

        custom_q = input("Optional custom question for these images (or press Enter to skip): ").strip() or None

        for idx, meta in enumerate(metadata_list, start=1):
            print(f"\n>>> Processing Image [{idx}/{len(metadata_list)}]: {meta.file_name}")
            result = self.analyzer.process_image(meta, custom_question=custom_q)
            print(ReportFormatter.build_report_text(result))

    def handle_single_file(self, meta: MediaMetadata) -> None:
        """Routes a single file through the appropriate multimodal pipeline."""
        print("\n================ File Summary ================")
        print(f"File Path     : {meta.file_path}")
        print(f"File Size     : {meta.size_mb:.2f} MB")
        print(f"Last Modified : {meta.modified_time}")
        print(f"Detected Type : {meta.media_type.upper()} ({meta.format_name})")
        if meta.duration_seconds is not None:
            print(f"Duration      : {meta.formatted_duration}")
        print("----------------------------------------------")

        if meta.media_type == "unsupported":
            print(f"Unsupported file format: {meta.extension}")
            return

        custom_q = input("Optional custom question (or press Enter for standard full analysis): ").strip() or None
        result: Optional[AnalysisResult] = None

        if meta.media_type == "image":
            result = self.analyzer.process_image(meta, custom_question=custom_q)

        elif meta.media_type == "document":
            result = self.analyzer.process_document(meta, custom_question=custom_q)

        elif meta.media_type == "audio":
            raw_lang = input("Language (Enter for Auto, or type telugu/english/hindi): ")
            lang = normalize_language_input(raw_lang)
            result = self.analyzer.process_audio(meta, language=lang, custom_question=custom_q)

        elif meta.media_type == "video":
            lang = None
            if meta.has_audio:
                raw_lang = input("Language (Enter for Auto, or type telugu/english/hindi): ")
                lang = normalize_language_input(raw_lang)

            run_vision = True
            if not USE_CUDA:
                choice = input("Analyze video frames with Vision AI? (Slow on CPU: ~2-3 min) [y/N]: ").strip().lower()
                run_vision = choice == "y"

            result = self.analyzer.process_video(
                meta,
                language=lang,
                run_vision=run_vision,
                custom_question=custom_q,
            )

        if result:
            print("\n" + ReportFormatter.build_report_text(result))
            save_choice = input("\nSave full report to a .txt file? [y/N]: ").strip().lower()
            if save_choice == "y":
                saved_path = ReportFormatter.save_report_to_disk(result)
                print(f"[Saved] Report written to: {saved_path}")

    def run(self) -> None:
        """Main interactive loop."""
        print_banner()
        while True:
            raw_input = input("Drag & drop file(s) or folder here (or type 'exit' to quit): ").strip()
            if raw_input.lower() in {"exit", "quit", "q"}:
                print("Exiting Local Multimodal AI File Analyzer. Goodbye!")
                break

            files = clean_user_input_paths(raw_input)
            if not files:
                print("No valid file path detected. Please try again.\n")
                continue

            metadata_list = [self.inspector.inspect_file(path) for path in files]

            try:
                if len(metadata_list) > 1:
                    self.handle_batch_images(metadata_list)
                else:
                    self.handle_single_file(metadata_list[0])
            except KeyboardInterrupt:
                print("\n[Interrupted] Current file analysis cancelled by user.")
            except Exception as exc:
                print(f"\n[Error] Unexpected failure while analyzing file: {exc}")

            print("\n-----------------------------------------------------------------------")


# ==============================================================================
# 9. ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    app = MultimodalCLIApp()
    app.run()