import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download
from huggingface_hub.utils import tqdm as hf_tqdm

import groq_client
import ffmpeg_worker as fw

MODEL_NAME = "Systran/faster-whisper-large-v2"
GROQ_MAX_UPLOAD_BYTES = 24 * 1024 * 1024  # запас под лимит Groq в 25 МБ

if sys.platform == "win32":
    _NO_WINDOW = subprocess.CREATE_NO_WINDOW
else:
    _NO_WINDOW = 0


class TranscriptionUnavailable(Exception):
    pass


class TranscriptionCancelled(Exception):
    pass


def load_model(log=print, progress=None):
    log("Загружаю модель large-v2...")

    tqdm_class = hf_tqdm
    if progress is not None:
        class _ProgressTqdm(hf_tqdm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                progress(self.n, self.total, self.desc)

            def update(self, n=1):
                super().update(n)
                progress(self.n, self.total, self.desc)

        tqdm_class = _ProgressTqdm

    model_path = snapshot_download(MODEL_NAME, tqdm_class=tqdm_class)
    model = WhisperModel(model_path, device="cpu", compute_type="int8")
    log("Модель загружена!\n")
    return model


def _format_srt_timestamp(seconds):
    total_ms = max(0, round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_segments(output_path, segments, output_format="srt", log=print):
    with open(output_path, "w", encoding="utf-8") as f:
        for cue_index, segment in enumerate(segments, 1):
            text = segment.text.strip()
            log(f"  [{segment.start:>7.1f}s] {text}")
            if output_format == "srt":
                f.write(f"{cue_index}\n")
                f.write(
                    f"{_format_srt_timestamp(segment.start)} --> {_format_srt_timestamp(segment.end)}\n"
                )
                f.write(f"{text}\n\n")
            else:
                f.write(text + "\n")


def _extract_compressed_audio(video_path, log=print):
    log("  Извлекаю и сжимаю аудио для Groq API (ffmpeg)...")

    try:
        ffmpeg_path = fw.get_ffmpeg_path()
    except FileNotFoundError as e:
        log(f"  {e} — пропускаю Groq API для этого файла")
        return None

    fd, tmp_path_str = tempfile.mkstemp(suffix=".ogg")
    os.close(fd)
    tmp_path = Path(tmp_path_str)
    cmd = [
        ffmpeg_path, "-y", "-nostdin", "-i", str(video_path),
        "-vn", "-map", "0:a:0",
        "-ar", "16000", "-ac", "1",
        "-c:a", "libopus", "-b:a", "32k", "-application", "voip",
        str(tmp_path),
    ]

    result = subprocess.run(
        cmd, capture_output=True, encoding="utf-8", errors="replace", creationflags=_NO_WINDOW
    )

    if result.returncode != 0:
        log(f"  Не удалось извлечь аудио для Groq API ({result.stderr.strip()[-200:]})")
        tmp_path.unlink(missing_ok=True)
        return None

    size_mb = tmp_path.stat().st_size / (1024 * 1024)
    if size_mb > GROQ_MAX_UPLOAD_BYTES / (1024 * 1024):
        log(f"  Аудио после сжатия всё ещё {size_mb:.1f} МБ (> 25 МБ) — пропускаю Groq API для этого файла")
        tmp_path.unlink(missing_ok=True)
        return None

    log(f"  Аудио готово: {size_mb:.1f} МБ")
    return tmp_path


def transcribe_file(get_local_model, video_path, index, total, log=print, output_format="srt", use_local_fallback=False, abort_event=None):
    path = Path(video_path)
    output_path = path.with_suffix(".srt" if output_format == "srt" else ".txt")

    log(f"Транскрибирую: {path.name}")

    if abort_event is not None and abort_event.is_set():
        raise TranscriptionCancelled(f"Отменено пользователем: {path.name}")

    segments = None
    key_idx = None
    model_used = None
    audio_path = _extract_compressed_audio(path, log=log)
    if audio_path is not None:
        try:
            while True:
                if abort_event is not None and abort_event.is_set():
                    raise TranscriptionCancelled(f"Отменено пользователем: {path.name}")
                segments, key_idx, model_used, wait_until = groq_client.transcribe_with_fallback_keys(
                    audio_path, language="ru", log=log
                )
                if segments is not None:
                    break
                if wait_until is not None and not use_local_fallback:
                    resume_at = datetime.fromtimestamp(wait_until).strftime("%H:%M:%S")
                    log(
                        f"  Все ключи Groq временно исчерпали лимит. "
                        f"Жду до {resume_at}, затем продолжу автоматически..."
                    )
                    remaining = max(0, wait_until - time.time())
                    if abort_event is not None:
                        woke_by_abort = abort_event.wait(timeout=remaining)
                    else:
                        time.sleep(remaining)
                        woke_by_abort = False
                    if woke_by_abort:
                        log("  Отменено пользователем во время ожидания сброса лимита.")
                        raise TranscriptionCancelled(f"Отменено пользователем: {path.name}")
                    log("  Лимиты должны были обновиться — пробую снова...")
                    continue
                break
        finally:
            audio_path.unlink(missing_ok=True)

    if segments is not None:
        log(f"  ✓ Распознано через Groq API (ключ #{key_idx}, модель {model_used})")
        write_segments(output_path, segments, output_format, log=log)
        log(f"  ✓ Готово → {output_path}")
        return output_path

    if not use_local_fallback:
        log("  Groq API недоступен: ключи не настроены / все невалидны / файл слишком большой — файл пропущен")
        raise TranscriptionUnavailable(f"Groq API недоступен для {path.name}, локальный fallback отключён в настройках")

    if abort_event is not None and abort_event.is_set():
        raise TranscriptionCancelled(f"Отменено пользователем: {path.name}")

    log("  Использую локальную модель Whisper (fallback)...")
    model = get_local_model()
    local_segments, info = model.transcribe(
        str(path),
        language="ru",
        beam_size=5,
    )
    log(f"  Длительность: {info.duration:.1f} сек — начинаю локальную транскрипцию...")
    write_segments(output_path, local_segments, output_format, log=log)
    log(f"  ✓ Готово → {output_path}")
    return output_path
