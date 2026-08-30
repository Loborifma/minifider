import os
import re
import shutil
import subprocess
import sys
from functools import lru_cache

VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm",
    ".wmv", ".flv", ".m4v", ".mpg", ".mpeg",
}

FASTSTART_EXTENSIONS = {".mp4", ".mov", ".m4v"}

PRESETS = {
    "High Quality": dict(
        max_w=1280, fps=30, x264_preset="medium", crf=20,
        maxrate="6M", bufsize="12M", abitrate="160k",
    ),
    "Medium (Recommended)": dict(
        max_w=1280, fps=30, x264_preset="medium", crf=24,
        maxrate="3M", bufsize="6M", abitrate="128k",
    ),
    "Low (Smallest)": dict(
        max_w=854, fps=24, x264_preset="faster", crf=28,
        maxrate="1.5M", bufsize="3M", abitrate="96k",
    ),
}

DEFAULT_PRESET = "Medium (Recommended)"

AUDIO_LEVEL_FILTER = (
    "highpass=f=80,"
    "acompressor=threshold=-30dB:ratio=6:attack=5:release=200:makeup=10,"
    "dynaudnorm=f=250:g=15:m=15:t=0.02"
)

DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)")
TIME_MS_RE = re.compile(r"out_time_ms=(\d+)")

if sys.platform == "win32":
    _NO_WINDOW = subprocess.CREATE_NO_WINDOW
else:
    _NO_WINDOW = 0


def _base_dir() -> str:
    if getattr(sys, "frozen", False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


@lru_cache(maxsize=1)
def get_ffmpeg_path() -> str:
    candidates = (
        os.path.join(_base_dir(), "ffmpeg.exe"),
        os.path.join(_base_dir(), "assets", "ffmpeg.exe"),
        shutil.which("ffmpeg"),
    )
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError("ffmpeg.exe не найден (проверены папка сборки, ./assets/ и PATH).")


def probe_duration(ffmpeg_path: str, input_path: str) -> float | None:
    result = subprocess.run(
        [ffmpeg_path, "-i", input_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_NO_WINDOW,
    )
    match = DURATION_RE.search(result.stderr or "")
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def build_output_path(input_path: str) -> str:
    root, ext = os.path.splitext(input_path)
    candidate = f"{root}_mini{ext}"
    counter = 1
    while os.path.exists(candidate):
        candidate = f"{root}_mini({counter}){ext}"
        counter += 1
    return candidate


def build_command(ffmpeg_path: str, input_path: str, output_path: str, preset_key: str, level_audio: bool = False) -> list[str]:
    p = PRESETS[preset_key]
    ext = os.path.splitext(input_path)[1].lower()
    cmd = [
        ffmpeg_path, "-y", "-i", input_path,
        "-map", "0:v:0", "-map", "0:a:0?", "-sn",
        "-vf", f"scale='min({p['max_w']},iw)':-2,fps={p['fps']}",
        "-c:v", "libx264", "-preset", p["x264_preset"], "-crf", str(p["crf"]),
        "-maxrate", p["maxrate"], "-bufsize", p["bufsize"],
        "-c:a", "aac", "-b:a", p["abitrate"],
    ]
    if level_audio:
        cmd += ["-af", AUDIO_LEVEL_FILTER]
    if ext in FASTSTART_EXTENSIONS:
        cmd += ["-movflags", "+faststart"]
    cmd += ["-progress", "pipe:1", "-loglevel", "error", "-nostats", output_path]
    return cmd


def compress_file(input_path: str, preset_key: str, progress_cb=None, proc_holder=None, level_audio: bool = False) -> tuple[bool, str, str | None]:
    ffmpeg_path = get_ffmpeg_path()
    duration = probe_duration(ffmpeg_path, input_path)
    output_path = build_output_path(input_path)
    cmd = build_command(ffmpeg_path, input_path, output_path, preset_key, level_audio=level_audio)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_NO_WINDOW,
    )
    if proc_holder is not None:
        proc_holder.append(proc)

    last_pct = -1
    for line in proc.stdout:
        match = TIME_MS_RE.search(line)
        if match and duration:
            pct = min(100, int(int(match.group(1)) / 10000 / duration))
            if pct != last_pct and progress_cb:
                progress_cb(pct)
                last_pct = pct

    stderr_output = proc.stderr.read()
    returncode = proc.wait()

    if returncode != 0:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        tail = "\n".join(stderr_output.strip().splitlines()[-10:])
        return False, tail or f"ffmpeg завершился с кодом {returncode}", None

    if progress_cb:
        progress_cb(100)
    return True, output_path, output_path


def run_batch(
    files: list[str], preset_key: str, ui_queue, cancel_event=None, proc_holder=None,
    level_audio: bool = False, compress: bool = True, transcribe: bool = False,
    subtitle_format: str = "srt", get_local_model=None,
    skip_paths=None, current_file_event=None,
) -> None:
    import groq_client
    import transcribe_worker as tw

    use_local_fallback = groq_client.get_use_local_fallback() if transcribe else False

    total = len(files)
    for index, path in enumerate(files, start=1):
        if cancel_event is not None and cancel_event.is_set():
            for remaining_index, remaining_path in enumerate(files[index - 1:], start=index):
                ui_queue.put(("file_done", remaining_path, "cancelled", "Отменено пользователем"))
                ui_queue.put(("batch_progress", remaining_index, total))
            break

        basename = os.path.basename(path)

        if skip_paths is not None and path in skip_paths:
            ui_queue.put(("log", f"[{basename}] Отменено пользователем"))
            ui_queue.put(("file_done", path, "cancelled", "Отменено пользователем"))
            ui_queue.put(("batch_progress", index, total))
            continue

        if current_file_event is not None:
            current_file_event.clear()
        if proc_holder is not None:
            proc_holder.clear()

        compress_ok, compress_info, out_path = True, None, path

        if compress:
            ui_queue.put(("log", f"[{basename}] Сжимаю видео..."))
            try:
                compress_ok, compress_info, out_path = compress_file(
                    path, preset_key,
                    progress_cb=lambda pct, p=path: ui_queue.put(("progress", p, pct)),
                    proc_holder=proc_holder,
                    level_audio=level_audio,
                )
            except Exception as exc:
                compress_ok, compress_info, out_path = False, str(exc), path
            if not compress_ok:
                ui_queue.put(("log", f"[{basename}] Ошибка сжатия: {compress_info}"))

        cancelled = (cancel_event is not None and cancel_event.is_set()) or \
                    (current_file_event is not None and current_file_event.is_set())

        transcribe_ok, transcribe_info = True, None
        if transcribe and not cancelled:
            source = out_path if (compress and compress_ok) else path
            ui_queue.put(("log", f"[{basename}] Транскрибирую..."))
            try:
                tw.transcribe_file(
                    get_local_model, source, index, total,
                    log=lambda t, b=basename: ui_queue.put(("log", f"[{b}] {t}")),
                    output_format=subtitle_format, use_local_fallback=use_local_fallback,
                    abort_event=current_file_event,
                )
            except tw.TranscriptionCancelled:
                cancelled = True
            except tw.TranscriptionUnavailable as exc:
                transcribe_ok, transcribe_info = False, str(exc)
                ui_queue.put(("log", f"[{basename}] Ошибка транскрипции: {exc}"))
            except Exception as exc:
                transcribe_ok, transcribe_info = False, str(exc)
                ui_queue.put(("log", f"[{basename}] Ошибка транскрипции: {exc}"))

        if cancelled:
            status, info = "cancelled", "Отменено пользователем"
        elif (compress and transcribe and compress_ok and transcribe_ok) or \
             (compress and not transcribe and compress_ok) or \
             (transcribe and not compress and transcribe_ok):
            status, info = "done", "OK"
        else:
            status = "error"
            info = " / ".join(filter(None, [
                None if compress_ok else compress_info,
                None if transcribe_ok else transcribe_info,
            ])) or "Неизвестная ошибка"

        ui_queue.put(("file_done", path, status, info))
        ui_queue.put(("batch_progress", index, total))
    ui_queue.put(("all_done",))
