import json
import os
import re
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import requests

GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODELS = ["whisper-large-v3", "whisper-large-v3-turbo"]
DEFAULT_COOLDOWN_SECONDS = 3600.0


class GroqError(Exception):
    pass


class GroqInvalidKey(GroqError):
    pass


class GroqPayloadTooLarge(GroqError):
    pass


class GroqRateLimited(GroqError):
    def __init__(self, message, retry_after):
        super().__init__(message)
        self.retry_after = retry_after


class GroqTransientError(GroqError):
    pass


def _config_path():
    appdata = os.getenv("APPDATA")
    base = Path(appdata) / "minifider" if appdata else Path.home() / ".minifider"
    return base / "config.json"


def _legacy_transcriber_config_path():
    appdata = os.getenv("APPDATA")
    base = Path(appdata) / "Transcriber" if appdata else Path.home() / ".transcriber"
    return base / "config.json"


def _migrate_legacy_config_if_needed():
    path = _config_path()
    if path.exists():
        return
    legacy_path = _legacy_transcriber_config_path()
    if not legacy_path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(legacy_path, path)


def load_config():
    _migrate_legacy_config_if_needed()
    path = _config_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, ValueError):
        config = {}
    config.setdefault("groq_keys", [])
    config.setdefault("use_local_fallback", False)
    for entry in config["groq_keys"]:
        entry.setdefault("status", "ok")
        if not isinstance(entry.get("cooldown_until"), dict):
            entry["cooldown_until"] = {}
    return config


def save_config(config):
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def has_any_keys():
    return bool(load_config()["groq_keys"])


def set_keys(raw_keys):
    config = load_config()
    existing_by_key = {entry["key"]: entry for entry in config["groq_keys"]}
    new_keys = []
    seen = set()
    for raw in raw_keys:
        key = raw.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        existing = existing_by_key.get(key)
        if existing is not None:
            new_keys.append(existing)
        else:
            new_keys.append({"key": key, "status": "ok", "cooldown_until": {}})
    config["groq_keys"] = new_keys
    save_config(config)


def mark_invalid(key):
    config = load_config()
    for entry in config["groq_keys"]:
        if entry["key"] == key:
            entry["status"] = "invalid"
    save_config(config)


def mark_cooldown(key, model, until_ts):
    config = load_config()
    for entry in config["groq_keys"]:
        if entry["key"] == key:
            entry["cooldown_until"][model] = until_ts
    save_config(config)


def get_use_local_fallback():
    return bool(load_config()["use_local_fallback"])


def set_use_local_fallback(enabled):
    config = load_config()
    config["use_local_fallback"] = bool(enabled)
    save_config(config)


_DURATION_RE = re.compile(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?$")


def _parse_go_duration(text):
    match = _DURATION_RE.match(text.strip())
    if not match or not any(match.groups()):
        return None
    hours, minutes, seconds = (float(g) if g else 0.0 for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def _parse_retry_seconds(resp, default=DEFAULT_COOLDOWN_SECONDS):
    retry_after = resp.headers.get("Retry-After")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass

    for header in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        value = resp.headers.get(header)
        if value:
            parsed = _parse_go_duration(value)
            if parsed is not None:
                return parsed

    return default


def transcribe_audio(key, audio_path, language="ru", model=GROQ_MODELS[0], timeout=120, log=print):
    size_mb = Path(audio_path).stat().st_size / (1024 * 1024)
    log(f"  Отправляю в Groq API ({model}, {size_mb:.1f} МБ), жду ответ...")

    try:
        with open(audio_path, "rb") as fh:
            resp = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {key}"},
                files={"file": (Path(audio_path).name, fh, "audio/ogg")},
                data={
                    "model": model,
                    "language": language,
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": "segment",
                },
                timeout=timeout,
            )
    except requests.RequestException as e:
        raise GroqTransientError(str(e)) from e

    if resp.status_code == 200:
        data = resp.json()
        segments = [
            SimpleNamespace(start=seg["start"], end=seg["end"], text=seg["text"])
            for seg in data.get("segments", [])
        ]
        log(f"  Groq API ответил: распознано сегментов — {len(segments)}")
        return segments

    if resp.status_code in (401, 403):
        raise GroqInvalidKey(f"HTTP {resp.status_code}")

    if resp.status_code == 413:
        raise GroqPayloadTooLarge(f"HTTP {resp.status_code}")

    if resp.status_code == 429:
        raise GroqRateLimited(f"HTTP 429: {resp.text[:200]}", retry_after=_parse_retry_seconds(resp))

    raise GroqTransientError(f"HTTP {resp.status_code}: {resp.text[:200]}")


def transcribe_with_fallback_keys(audio_path, language="ru", log=print):
    config = load_config()
    keys = config["groq_keys"]
    now = time.time()
    earliest_wait = None
    any_ok_key = False

    for idx, entry in enumerate(keys, 1):
        if entry["status"] != "ok":
            continue
        any_ok_key = True
        cooldowns = entry["cooldown_until"]
        key_invalidated = False

        for model in GROQ_MODELS:
            until = cooldowns.get(model, 0)
            if until > now:
                if earliest_wait is None or until < earliest_wait:
                    earliest_wait = until
                continue

            try:
                segments = transcribe_audio(entry["key"], audio_path, language=language, model=model, log=log)
                return segments, idx, model, None
            except GroqInvalidKey:
                log(f"  Groq ключ #{idx}: недействителен (401/403) — помечаю и больше не использую")
                mark_invalid(entry["key"])
                key_invalidated = True
                break
            except GroqRateLimited as e:
                until_ts = now + e.retry_after
                log(f"  Groq ключ #{idx} ({model}): превышен лимит, пауза ~{e.retry_after:.0f}с")
                mark_cooldown(entry["key"], model, until_ts)
                if earliest_wait is None or until_ts < earliest_wait:
                    earliest_wait = until_ts
            except GroqPayloadTooLarge:
                log("  Groq: файл слишком большой даже после сжатия — пропускаю все ключи для этого файла")
                return None, None, None, None
            except GroqTransientError as e:
                log(f"  Groq ключ #{idx} ({model}): временная ошибка ({e}) — пробую дальше")

        if key_invalidated:
            continue

    if not any_ok_key:
        return None, None, None, None

    return None, None, None, earliest_wait
