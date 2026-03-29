import argparse
import hashlib
import imghdr
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from datetime import datetime
from pathlib import Path

import requests
from flask import Flask, abort, flash, jsonify, redirect, render_template, request, send_file, session, url_for

from civitai_api import fetch_model_by_id, load_api_key, save_api_key
from civitai_api import fetch_one_page
from fetch_service import (
    MODE_OPTIONS,
    PERIOD_OPTIONS,
    SORT_OPTIONS,
    build_db_path,
    fetch_for_display,
    fetch_and_store,
    refresh_option_cache,
)
from media_store import save_media_for_models
from sqlite_store import (
    get_filter_options,
    get_media_maintenance_params,
    get_model_by_row_id,
    init_db,
    model_to_row,
    save_models,
    search_models,
    update_model_maintenance_params,
    update_model_media_path,
)


APP_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DB_PATH = build_db_path(APP_DIR)
CACHE_FILE = os.path.join(APP_DIR, "dropdown_cache.json")
UI_SETTINGS_FILE = os.path.join(APP_DIR, "ui_settings.json")
DB_MEDIA_DIR = os.path.join(APP_DIR, "db_media")
DEFAULT_PER_PAGE = 20
VIEW_MODE_OPTIONS = [
    ("db", "DB表示"),
    ("browse", "Civitai表示"),
]
JOBS = {}
JOBS_LOCK = threading.Lock()
BROWSE_RESULTS = {}
BROWSE_LOCK = threading.Lock()
TABLE_SORT_FIELDS = {
    "name": "モデル名",
    "model_type": "タイプ",
    "base_model": "ベース",
    "model_url": "URL",
    "allow_no_credit": "クレジット",
    "allow_sell_images": "画像販売",
    "allow_run_paid_service": "有料利用",
    "allow_share_merges": "マージ",
    "allow_sell_model": "モデル販売",
    "allow_different_licenses": "別ライセンス",
}

app = Flask(__name__)
app.config["SECRET_KEY"] = "civitai-flask-local-dev"


def _delayed_exit():
    time.sleep(1.0)
    os._exit(0)


def restart_app_process():
    host = app.config.get("APP_HOST", "127.0.0.1")
    port = str(app.config.get("APP_PORT", 5000))
    command = [sys.executable, os.path.join(APP_DIR, "app.py"), "--host", host, "--port", port, "--skip-browser"]
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    subprocess.Popen(command, cwd=APP_DIR, creationflags=creationflags)
    threading.Thread(target=_delayed_exit, daemon=True).start()


def _open_browser_later(host: str, port: int) -> None:
    time.sleep(1.5)
    webbrowser.open(f"http://{host}:{port}/models")


def _ensure_main_db() -> str:
    init_db(MAIN_DB_PATH)
    return MAIN_DB_PATH


def _default_ui_settings() -> dict:
    cache_root = os.path.join(APP_DIR, "cache")
    download_root = os.path.join(APP_DIR, "downloads")
    return {
        "nsfw_enabled": False,
        "include_video": False,
        "info_cache_enabled": True,
        "download_preview_enabled": True,
        "download_civitai_info_enabled": True,
        "info_cache_file": os.path.join(cache_root, "civitai_info_cache.json"),
        "download_root_dir": download_root,
        "download_type_dirs": {
            "LORA": os.path.join(download_root, "LORA"),
            "Checkpoint": os.path.join(download_root, "Checkpoint"),
        },
    }


def _default_maintenance_settings() -> dict:
    return {
        "image_format": "webp",
        "image_quality": 75,
        "image_keep_exif": True,
        "image_skip_processed": True,
        "video_format": "mp4",
        "video_quality": "medium",
        "video_resolution": "original",
        "video_keep_audio": True,
        "video_skip_processed": True,
    }


def _load_maintenance_settings() -> dict:
    stored = session.get("maintenance_form") or {}
    default = _default_maintenance_settings()
    return {
        "image_format": str(stored.get("image_format", default["image_format"])).lower() if str(stored.get("image_format", default["image_format"])).lower() in {"webp", "jpg"} else default["image_format"],
        "image_quality": max(1, min(int(stored.get("image_quality", default["image_quality"])), 100)),
        "image_keep_exif": bool(stored.get("image_keep_exif", default["image_keep_exif"])),
        "image_skip_processed": bool(stored.get("image_skip_processed", default["image_skip_processed"])),
        "video_format": str(stored.get("video_format", default["video_format"])).lower() if str(stored.get("video_format", default["video_format"])).lower() in {"mp4", "webm"} else default["video_format"],
        "video_quality": str(stored.get("video_quality", default["video_quality"])).lower() if str(stored.get("video_quality", default["video_quality"])).lower() in {"high", "medium", "low"} else default["video_quality"],
        "video_resolution": str(stored.get("video_resolution", default["video_resolution"])).lower() if str(stored.get("video_resolution", default["video_resolution"])).lower() in {"original", "720", "512"} else default["video_resolution"],
        "video_keep_audio": bool(stored.get("video_keep_audio", default["video_keep_audio"])),
        "video_skip_processed": bool(stored.get("video_skip_processed", default["video_skip_processed"])),
    }


def _store_maintenance_settings(settings: dict) -> None:
    session["maintenance_form"] = {
        "image_format": settings["image_format"],
        "image_quality": settings["image_quality"],
        "image_keep_exif": bool(settings["image_keep_exif"]),
        "image_skip_processed": bool(settings["image_skip_processed"]),
        "video_format": settings["video_format"],
        "video_quality": settings["video_quality"],
        "video_resolution": settings["video_resolution"],
        "video_keep_audio": bool(settings["video_keep_audio"]),
        "video_skip_processed": bool(settings["video_skip_processed"]),
    }


def _available_model_types() -> list[str]:
    default_types = ["Checkpoint", "LORA"]
    if not os.path.exists(CACHE_FILE):
        return default_types
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        types = [str(value).strip() for value in (cache.get("types") or []) if str(value).strip()]
        types = [value for value in types if value != "All"]
        return types or default_types
    except Exception:
        return default_types


def _load_dropdown_cache_values() -> dict:
    if not os.path.exists(CACHE_FILE):
        return {"types": [], "bases": [], "known_model_ids": []}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "types": [str(value).strip() for value in (data.get("types") or []) if str(value).strip() and str(value).strip() != "All"],
            "bases": [str(value).strip() for value in (data.get("bases") or []) if str(value).strip() and str(value).strip() not in {"All", "Other"}],
            "known_model_ids": [str(value).strip() for value in (data.get("known_model_ids") or []) if str(value).strip()],
        }
    except Exception:
        return {"types": [], "bases": [], "known_model_ids": []}


def _parse_manual_dropdown_values(text: str) -> list[str]:
    values = []
    seen = set()
    for part in re.split(r"[\r\n,]+", text or ""):
        value = part.strip()
        if not value or value in seen:
            continue
        values.append(value)
        seen.add(value)
    return values


def _replace_dropdown_cache_values(*, edited_types: str, edited_bases: str) -> tuple[int, int]:
    cache = _load_dropdown_cache_values()
    new_types = _parse_manual_dropdown_values(edited_types)
    new_bases = _parse_manual_dropdown_values(edited_bases)

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "types": sorted(new_types),
                "bases": sorted(new_bases),
                "known_model_ids": cache["known_model_ids"][:100],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return len(new_types), len(new_bases)


def _load_ui_settings() -> dict:
    default = _default_ui_settings()
    if not os.path.exists(UI_SETTINGS_FILE):
        settings = dict(default)
        settings["download_type_dirs"] = dict(default["download_type_dirs"])
        for model_type in _available_model_types():
            settings["download_type_dirs"].setdefault(model_type, os.path.join(settings["download_root_dir"], model_type))
        return settings
    try:
        import json
        with open(UI_SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        settings = {**default, **data}
        download_type_dirs = dict(default["download_type_dirs"])
        download_type_dirs.update(data.get("download_type_dirs") or {})
        if data.get("download_lora_dir"):
            download_type_dirs["LORA"] = data["download_lora_dir"]
        if data.get("download_checkpoint_dir"):
            download_type_dirs["Checkpoint"] = data["download_checkpoint_dir"]
        for model_type in _available_model_types():
            download_type_dirs.setdefault(model_type, os.path.join(settings["download_root_dir"], model_type))
        settings["download_type_dirs"] = download_type_dirs
        return settings
    except Exception:
        settings = dict(default)
        settings["download_type_dirs"] = dict(default["download_type_dirs"])
        for model_type in _available_model_types():
            settings["download_type_dirs"].setdefault(model_type, os.path.join(settings["download_root_dir"], model_type))
        return settings


def _save_ui_settings(settings: dict) -> None:
    with open(UI_SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)


def _get_info_cache_file() -> str:
    ui_settings = _load_ui_settings()
    return ui_settings.get("info_cache_file") or _default_ui_settings()["info_cache_file"]


def _build_info_cache_key(params: dict) -> str:
    use_name_query = params.get("kw_mode") == "Name" and bool((params.get("keyword") or "").strip())
    cache_params = {
        "kw_mode": params.get("kw_mode", ""),
        "keyword": params.get("keyword", ""),
        "selected_type": params.get("selected_type", ""),
        "selected_base": params.get("selected_base", ""),
        "sort": params.get("sort", ""),
        "period": params.get("period", ""),
        "nsfw_enabled": bool(params.get("nsfw_enabled", False)),
        "include_video": bool(params.get("include_video", True)),
        "start_page": 1 if use_name_query else int(params.get("start_page", 1) or 1),
        "limit": int(params.get("limit", 20) or 20),
        "wait_time": float(params.get("wait_time", 0) or 0),
        "loop": bool(params.get("loop")),
    }
    raw = json.dumps(cache_params, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_info_cache() -> dict:
    cache_file = _get_info_cache_file()
    if not os.path.exists(cache_file):
        return {}
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_info_cache(cache_data: dict) -> None:
    cache_file = _get_info_cache_file()
    os.makedirs(os.path.dirname(cache_file) or APP_DIR, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)


def _clear_info_cache() -> int:
    removed = 0
    cache_file = _get_info_cache_file()
    if os.path.exists(cache_file):
        os.remove(cache_file)
        removed += 1
    return removed

def _get_cached_display_result(params: dict) -> dict | None:
    if not _load_ui_settings().get("info_cache_enabled", True):
        return None
    cache = _load_info_cache()
    entry = cache.get(_build_info_cache_key(params))
    if not isinstance(entry, dict):
        return None
    items = entry.get("items")
    if not isinstance(items, list):
        return None
    return {
        "items": items,
        "total": int(entry.get("total", len(items))),
        "cached_at": entry.get("cached_at", ""),
        "next_page_url": entry.get("next_page_url", ""),
        "uses_local_paging": bool(entry.get("uses_local_paging", False)),
    }


def _store_cached_display_result(params: dict, result: dict) -> None:
    if not _load_ui_settings().get("info_cache_enabled", True):
        return
    cache = _load_info_cache()
    cache[_build_info_cache_key(params)] = {
        "items": result["items"],
        "total": result["total"],
        "cached_at": datetime.now().isoformat(),
        "next_page_url": result.get("next_page_url", ""),
        "uses_local_paging": bool(result.get("uses_local_paging", False)),
    }
    _save_info_cache(cache)


def _is_video_item(item: dict) -> bool:
    media_type = (item.get("thumbnail_media_type") or "").lower()
    url = (item.get("thumbnail_url") or "").lower()
    return media_type == "video" or url.endswith((".mp4", ".webm", ".mov"))


def _guess_file_mimetype(path: str) -> str | None:
    image_kind = imghdr.what(path)
    image_mimetypes = {
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
    }
    if image_kind in image_mimetypes:
        return image_mimetypes[image_kind]
    guessed, _ = mimetypes.guess_type(path)
    return guessed


def _build_civitai_thumb_url(url: str, width: int = 300) -> str:
    source = (url or "").strip()
    if not source:
        return ""
    if "image.civitai.com/" not in source:
        return source
    if "/original=true/" in source:
        return source.replace("/original=true/", f"/width={width}/", 1)
    return source


def _media_db_model_id_from_path(path: Path) -> str:
    return path.stem


def _build_exif_user_comment(parameters: str) -> bytes:
    if not parameters:
        return b""
    try:
        import piexif
    except ImportError as exc:
        raise RuntimeError("parameters を保持するには piexif が必要です。venv にインストールしてください。") from exc
    prefix = b"UNICODE\x00"
    exif_dict = {"Exif": {piexif.ExifIFD.UserComment: prefix + parameters.encode("utf-16-be")}}
    return piexif.dump(exif_dict)


def _merge_exif_with_parameters(existing_exif: bytes | None, parameters: str) -> bytes | None:
    if not existing_exif and not parameters:
        return None
    if not parameters:
        return existing_exif or None
    try:
        import piexif
    except ImportError as exc:
        raise RuntimeError("parameters を保持するには piexif が必要です。venv にインストールしてください。") from exc

    if existing_exif:
        try:
            exif_dict = piexif.load(existing_exif)
        except Exception:
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}, "thumbnail": None}
    else:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}, "thumbnail": None}
    prefix = b"UNICODE\x00"
    exif_dict.setdefault("Exif", {})
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = prefix + parameters.encode("utf-16-be")
    return piexif.dump(exif_dict)


def _run_image_maintenance(
    media_dir: str,
    target_format: str,
    quality: int,
    keep_exif: bool,
    skip_processed: bool,
) -> dict:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("画像メンテナンスには Pillow が必要です。venv にインストールしてください。") from exc

    supported_exts = {".jpg", ".jpeg", ".png", ".webp"}
    format_name = "WEBP" if target_format == "webp" else "JPEG"
    output_ext = ".webp" if target_format == "webp" else ".jpg"
    current_params = json.dumps(
        {"format": target_format, "quality": quality, "keep_exif": bool(keep_exif)},
        ensure_ascii=False,
        sort_keys=True,
    )
    existing_params = get_media_maintenance_params(MAIN_DB_PATH)
    processed = 0
    skipped = 0
    failed = 0
    samples: list[str] = []

    for path in Path(media_dir).iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in supported_exts:
            skipped += 1
            continue
        model_id = _media_db_model_id_from_path(path)
        if skip_processed and existing_params.get(model_id, {}).get("image") == current_params:
            skipped += 1
            continue
        try:
            with Image.open(path) as img:
                parameters = str(img.info.get("parameters") or "").strip() if keep_exif else ""
                exif = None
                if keep_exif:
                    exif = _merge_exif_with_parameters(img.info.get("exif"), parameters)
                if target_format == "jpg":
                    if img.mode in ("RGBA", "LA"):
                        bg = Image.new("RGB", img.size, (255, 255, 255))
                        bg.paste(img, mask=img.split()[-1])
                        out = bg
                    else:
                        out = img.convert("RGB")
                else:
                    out = img if img.mode in ("RGB", "RGBA") else img.convert("RGBA" if "A" in img.mode else "RGB")

                tmp_path = path.with_name(f"{path.stem}.tmp{output_ext}")
                save_kwargs = {"quality": quality}
                if target_format == "jpg":
                    save_kwargs["optimize"] = True
                else:
                    save_kwargs["method"] = 6
                if exif and target_format in {"jpg", "webp"}:
                    save_kwargs["exif"] = exif
                out.save(tmp_path, format_name, **save_kwargs)

            final_path = path.with_suffix(output_ext)
            if final_path.exists() and final_path != path:
                final_path.unlink()
            if final_path == path:
                tmp_path.replace(path)
                saved_path = path
            else:
                tmp_path.replace(final_path)
                path.unlink(missing_ok=True)
                saved_path = final_path

            update_model_media_path(MAIN_DB_PATH, model_id, str(saved_path))
            update_model_maintenance_params(MAIN_DB_PATH, model_id, image_params=current_params)
            processed += 1
        except Exception as exc:
            failed += 1
            if len(samples) < 5:
                samples.append(f"{path.name}: {exc}")

    return {"processed": processed, "skipped": skipped, "failed": failed, "samples": samples}


def _video_quality_profile(target_format: str, quality: str) -> tuple[str, list[str]]:
    quality_key = quality if quality in {"high", "medium", "low"} else "medium"
    if target_format == "webm":
        crf_map = {"high": "30", "medium": "36", "low": "42"}
        return "webm", ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", crf_map[quality_key], "-row-mt", "1", "-deadline", "good"]
    crf_map = {"high": "23", "medium": "28", "low": "32"}
    return "mp4", ["-c:v", "libx264", "-preset", "medium", "-crf", crf_map[quality_key], "-pix_fmt", "yuv420p", "-movflags", "+faststart"]


def _video_scale_filter(resolution: str) -> list[str]:
    if resolution not in {"720", "512"}:
        return []
    size = resolution
    scale = f"scale='if(gt(iw,ih),min(iw,{size}),-2)':'if(gt(ih,iw),min(ih,{size}),-2)'"
    return ["-vf", scale]


def _run_video_maintenance(
    media_dir: str,
    target_format: str,
    quality: str,
    resolution: str,
    keep_audio: bool,
    skip_processed: bool,
) -> dict:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return {"processed": 0, "skipped": 0, "failed": 0, "samples": [], "ffmpeg_missing": True}

    supported_exts = {".mp4", ".webm", ".mov"}
    output_ext, video_args = _video_quality_profile(target_format, quality)
    file_ext = f".{output_ext}"
    current_params = json.dumps(
        {
            "format": target_format,
            "quality": quality,
            "resolution": resolution,
            "keep_audio": bool(keep_audio),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    existing_params = get_media_maintenance_params(MAIN_DB_PATH)
    processed = 0
    skipped = 0
    failed = 0
    samples: list[str] = []

    for path in Path(media_dir).iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in supported_exts:
            skipped += 1
            continue
        model_id = _media_db_model_id_from_path(path)
        if skip_processed and existing_params.get(model_id, {}).get("video") == current_params:
            skipped += 1
            continue
        tmp_path = path.with_name(f"{path.stem}.tmp{file_ext}")
        final_path = path.with_suffix(file_ext)
        command = [
            ffmpeg_path,
            "-y",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            *video_args,
            *_video_scale_filter(resolution),
        ]
        if keep_audio:
            if output_ext == "webm":
                command.extend(["-c:a", "libopus", "-b:a", "96k"])
            else:
                command.extend(["-c:a", "aac", "-b:a", "128k"])
        else:
            command.append("-an")
        command.append(str(tmp_path))
        try:
            result = subprocess.run(command, capture_output=True, text=True, cwd=APP_DIR)
            if result.returncode != 0:
                raise RuntimeError((result.stderr or result.stdout or "ffmpeg failed").strip())
            if final_path.exists() and final_path != path:
                final_path.unlink()
            if final_path == path:
                tmp_path.replace(path)
                saved_path = path
            else:
                tmp_path.replace(final_path)
                path.unlink(missing_ok=True)
                saved_path = final_path
            update_model_media_path(MAIN_DB_PATH, model_id, str(saved_path))
            update_model_maintenance_params(MAIN_DB_PATH, model_id, video_params=current_params)
            processed += 1
        except Exception as exc:
            failed += 1
            tmp_path.unlink(missing_ok=True)
            if len(samples) < 5:
                samples.append(f"{path.name}: {exc}")

    return {"processed": processed, "skipped": skipped, "failed": failed, "samples": samples, "ffmpeg_missing": False}


def _run_db_media_maintenance(
    *,
    image_format: str,
    image_quality: int,
    image_keep_exif: bool,
    video_format: str,
    video_quality: str,
    video_resolution: str,
    video_keep_audio: bool,
) -> dict:
    os.makedirs(DB_MEDIA_DIR, exist_ok=True)
    image_result = _run_image_maintenance(DB_MEDIA_DIR, image_format, image_quality, image_keep_exif)
    video_result = _run_video_maintenance(DB_MEDIA_DIR, video_format, video_quality, video_resolution, video_keep_audio)
    return {"image": image_result, "video": video_result}


def _decorate_media_item(item: dict, ui_settings: dict | None = None) -> dict:
    decorated = dict(item)
    decorated["is_video"] = _is_video_item(item)
    settings = ui_settings or _load_ui_settings()
    decorated["poster_only"] = bool(not settings.get("include_video", True) and decorated["is_video"])
    db_local_path = (item.get("thumbnail_local_path") or "").strip()
    has_db_local = bool(db_local_path and os.path.exists(db_local_path))
    full_media_url = item.get("thumbnail_url") or ""

    if has_db_local:
        display_media_url = url_for("model_media", row_id=item["id"])
    else:
        display_media_url = _build_civitai_thumb_url(full_media_url) if not decorated["is_video"] else full_media_url

    decorated["display_media_url"] = display_media_url
    decorated["full_media_url"] = url_for("model_media", row_id=item["id"]) if has_db_local else full_media_url

    decorated["has_local_media"] = has_db_local
    return decorated


def _base_fetch_form() -> dict:
    api_key = load_api_key(APP_DIR)
    ui_settings = _load_ui_settings()
    try:
        options = refresh_option_cache(CACHE_FILE, api_key, is_manual_refresh=False, nsfw_enabled=ui_settings.get("nsfw_enabled", False))
    except Exception:
        options = {"types": ["All", "Checkpoint", "LORA"], "bases": ["All", "Other", "SD 1.5", "SDXL 1.0"]}
    return {
        "kw_mode": "Tag",
        "keyword": "",
        "selected_type": "",
        "selected_base": "",
        "sort": SORT_OPTIONS[0],
        "period": PERIOD_OPTIONS[0],
        "start_page": 1,
        "limit": 20,
        "wait_time": 0,
        "mode": "single",
        "nsfw_enabled": ui_settings.get("nsfw_enabled", False),
        "types": options["types"],
        "bases": options["bases"],
    }


def _fetch_form_from_request_args() -> dict:
    fetch_form = _base_fetch_form()
    fetch_form.update({
        "kw_mode": request.args.get("fetch_kw_mode", fetch_form["kw_mode"]).strip() or fetch_form["kw_mode"],
        "keyword": request.args.get("fetch_keyword", fetch_form["keyword"]).strip(),
        "selected_type": request.args.get("fetch_selected_type", fetch_form["selected_type"]).strip() or fetch_form["selected_type"],
        "selected_base": request.args.get("fetch_selected_base", fetch_form["selected_base"]).strip() or fetch_form["selected_base"],
        "sort": request.args.get("fetch_sort", fetch_form["sort"]).strip() or fetch_form["sort"],
        "period": request.args.get("fetch_period", fetch_form["period"]).strip() or fetch_form["period"],
        "start_page": max(request.args.get("fetch_start_page", default=fetch_form["start_page"], type=int), 1),
        "limit": max(min(request.args.get("fetch_limit", default=fetch_form["limit"], type=int), 100), 1),
        "wait_time": max(request.args.get("fetch_wait_time", default=fetch_form["wait_time"], type=float), 0),
        "mode": request.args.get("fetch_mode", fetch_form["mode"]).strip() or fetch_form["mode"],
    })
    return fetch_form


def _job_snapshot(job_id: str) -> dict | None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return None
        return dict(job)


@app.get("/jobs/<job_id>.json")
def job_status(job_id: str):
    job = _job_snapshot(job_id)
    if not job:
        abort(404)
    return jsonify(job)


def _latest_jobs(limit: int = 10) -> list[dict]:
    with JOBS_LOCK:
        return list(reversed(list(JOBS.values())[-limit:]))


def _get_browse_result(browse_id: str | None) -> dict | None:
    if not browse_id:
        return None
    with BROWSE_LOCK:
        result = BROWSE_RESULTS.get(browse_id)
        return dict(result) if result else None


def _save_browse_result(browse_id: str, browse_result: dict) -> None:
    with BROWSE_LOCK:
        BROWSE_RESULTS[browse_id] = browse_result


def _store_browse_result(items: list[dict], params: dict, meta: dict | None = None) -> str:
    browse_id = uuid.uuid4().hex[:10]
    with BROWSE_LOCK:
        BROWSE_RESULTS[browse_id] = {
            "id": browse_id,
            "items": items,
            "raw_items": items,
            "params": params,
            "created_at": datetime.now().isoformat(),
            "meta": meta or {},
        }
    return browse_id


def _normalize_page_cache(meta: dict) -> dict:
    cache = meta.get("page_cache") or {}
    return cache if isinstance(cache, dict) else {}


def _normalize_table_sorts(fields: str, directions: str) -> list[tuple[str, str]]:
    raw_fields = [part.strip() for part in (fields or "").split(",") if part.strip()]
    raw_dirs = [part.strip().lower() for part in (directions or "").split(",")]
    sorts: list[tuple[str, str]] = []
    seen = set()
    for index, field in enumerate(raw_fields):
        if field not in TABLE_SORT_FIELDS or field in seen:
            continue
        direction = raw_dirs[index] if index < len(raw_dirs) else "asc"
        sorts.append((field, "desc" if direction == "desc" else "asc"))
        seen.add(field)
    return sorts


def _encode_table_sorts(sorts: list[tuple[str, str]]) -> tuple[str, str]:
    if not sorts:
        return "", ""
    return ",".join(field for field, _ in sorts), ",".join(direction for _, direction in sorts)


def _toggle_table_sort(sorts: list[tuple[str, str]], field: str) -> list[tuple[str, str]]:
    if field not in TABLE_SORT_FIELDS:
        return list(sorts)
    current = list(sorts)
    existing_dir = None
    remaining = []
    for sort_field, direction in current:
        if sort_field == field and existing_dir is None:
            existing_dir = direction
            continue
        remaining.append((sort_field, direction))
    if existing_dir is None:
        return [(field, "asc"), *remaining]
    if existing_dir == "asc":
        return [(field, "desc"), *remaining]
    return remaining


def _table_sort_map(sorts: list[tuple[str, str]]) -> dict[str, str]:
    return {field: direction for field, direction in sorts}


def _table_sort_value(item: dict, field: str):
    if field and field not in item and ("modelVersions" in item or "type" in item):
        item = model_to_row(item)
    if field.startswith("allow_"):
        return 0 if bool(item.get(field)) else 1
    return str(item.get(field) or "").casefold()


def _sort_model_items(items: list[dict], sorts: list[tuple[str, str]]) -> list[dict]:
    if not sorts:
        return list(items)
    sorted_items = list(items)
    for field, direction in reversed(sorts):
        reverse = direction == "desc"
        sorted_items = sorted(sorted_items, key=lambda item: _table_sort_value(item, field), reverse=reverse)
    return sorted_items


def _browse_redirect_params(browse_id: str, **extra) -> dict:
    params = {"view": "browse", "browse": browse_id}
    browse_result = _get_browse_result(browse_id)
    meta = browse_result.get("meta", {}) if browse_result else {}
    table_sorts = _normalize_table_sorts(
        str(meta.get("table_sort", "") or ""),
        str(meta.get("table_dir", "") or ""),
    )
    sort_fields, sort_dirs = _encode_table_sorts(table_sorts)
    if sort_fields:
        params["table_sort"] = sort_fields
        params["table_dir"] = sort_dirs
    params.update(extra)
    return params


@app.context_processor
def inject_template_helpers():
    def models_url(**updates):
        params = request.args.to_dict(flat=True)
        params.setdefault("view", session.get("last_view_mode", "browse"))
        for key, value in updates.items():
            if value is None:
                params.pop(key, None)
            else:
                params[key] = value
        return url_for("models", **params)

    return {
        "models_url": models_url,
        "table_sort_fields": TABLE_SORT_FIELDS,
    }


def _cache_current_browse_page(browse_result: dict) -> None:
    meta = dict(browse_result.get("meta") or {})
    page = int(meta.get("page", 1) or 1)
    page_cache = _normalize_page_cache(meta)
    page_cache[str(page)] = {
        "items": browse_result.get("raw_items") or browse_result.get("items", []),
        "page": page,
        "has_next": bool(meta.get("has_next", False)),
        "next_page_url": meta.get("next_page_url", ""),
    }
    meta["page_cache"] = page_cache
    browse_result["meta"] = meta


def _sanitize_filename(name: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", (name or "").strip())
    sanitized = sanitized.strip(" .")
    return sanitized or "model"


def _select_download_file(model: dict) -> dict | None:
    versions = model.get("modelVersions") or []
    if not versions:
        return None
    files = versions[0].get("files") or []
    if not files:
        return None
    for file_info in files:
        if file_info.get("primary") and file_info.get("downloadUrl"):
            return file_info
    for file_info in files:
        if file_info.get("downloadUrl"):
            return file_info
    return None


def _resolve_download_dir(model: dict) -> str:
    settings = _load_ui_settings()
    model_type = (model.get("type") or model.get("model_type") or "").strip()
    target_dir = (settings.get("download_type_dirs") or {}).get(model_type)
    if not target_dir:
        target_dir = settings.get("download_root_dir") or APP_DIR
    os.makedirs(target_dir, exist_ok=True)
    return target_dir


def _first_preview_info(model: dict) -> tuple[str, str]:
    include_video = _load_ui_settings().get("include_video", True)
    versions = model.get("modelVersions") or []
    if not versions:
        return "", ""
    images = versions[0].get("images") or []
    for image in images:
        media_type = (image.get("type") or "").strip().lower()
        if not include_video and media_type == "video":
            continue
        url = (image.get("url") or "").strip()
        if url:
            return url, media_type
    return "", ""


def _download_extra_file(url: str, save_path: str, api_key: str) -> None:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with requests.get(url, headers=headers, stream=True, timeout=(10, 600)) as response:
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)


def _build_civitai_info_payload(model: dict, file_info: dict) -> dict:
    version = (model.get("modelVersions") or [{}])[0]
    return {
        "id": version.get("id"),
        "modelId": model.get("id"),
        "name": version.get("name"),
        "baseModel": version.get("baseModel"),
        "description": version.get("description"),
        "trainedWords": version.get("trainedWords") or [],
        "model": {
            "name": model.get("name"),
            "type": model.get("type"),
            "nsfw": model.get("nsfw", False),
            "description": model.get("description"),
        },
        "files": version.get("files") or [],
        "images": version.get("images") or [],
        "selectedFile": file_info,
    }


def _save_download_extras(api_key: str, model: dict, file_info: dict, save_path: str) -> list[str]:
    settings = _load_ui_settings()
    saved = []
    base_path, _ = os.path.splitext(save_path)

    if settings.get("download_preview_enabled", True):
        preview_url, _preview_type = _first_preview_info(model)
        if preview_url:
            preview_ext = os.path.splitext(urlparse(preview_url).path)[1].lower() or ".jpeg"
            preview_path = f"{base_path}.preview{preview_ext}"
            if not os.path.exists(preview_path):
                _download_extra_file(preview_url, preview_path, api_key)
            saved.append(preview_path)

    if settings.get("download_civitai_info_enabled", True):
        info_path = f"{base_path}.civitai.info"
        payload = _build_civitai_info_payload(model, file_info)
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        saved.append(info_path)

    return saved


def _download_model_file(api_key: str, model: dict) -> tuple[str, str]:
    source_model = model
    file_info = _select_download_file(source_model)
    if not file_info:
        source_model = fetch_model_by_id(api_key, str(model.get("id") or model.get("civitai_model_id") or ""))
        file_info = _select_download_file(source_model)
    if not file_info:
        raise RuntimeError("ダウンロード可能なモデルファイルが見つかりません。")

    download_url = (file_info.get("downloadUrl") or "").strip()
    if not download_url:
        raise RuntimeError("ダウンロードURLが取得できません。")

    target_dir = _resolve_download_dir(source_model)
    filename = _sanitize_filename(file_info.get("name") or "")
    if not os.path.splitext(filename)[1]:
        parsed = urlparse(download_url)
        ext = os.path.splitext(parsed.path)[1]
        if ext:
            filename += ext
    save_path = os.path.join(target_dir, filename)
    if os.path.exists(save_path):
        _save_download_extras(api_key, source_model, file_info, save_path)
        return save_path, "exists"

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with requests.get(download_url, headers=headers, stream=True, timeout=(10, 600)) as response:
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    _save_download_extras(api_key, source_model, file_info, save_path)
    return save_path, "downloaded"


def _update_job(job_id: str, **fields) -> None:
    with JOBS_LOCK:
        JOBS[job_id].update(fields)


def _append_job_log(job_id: str, message: str) -> None:
    with JOBS_LOCK:
        logs = JOBS[job_id].setdefault("logs", [])
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        JOBS[job_id]["updated_at"] = datetime.now().isoformat()


def _run_fetch_job(job_id: str, params: dict) -> None:
    try:
        _update_job(job_id, status="running")

        def progress_cb(message: str) -> None:
            _append_job_log(job_id, message)

        result = fetch_and_store(APP_DIR, params, progress_cb)
        _update_job(
            job_id,
            status="completed",
            result=result,
            updated_at=datetime.now().isoformat(),
        )
        _append_job_log(job_id, f"完了: {result['total_saved']} 件を {result['db_name']} に保存しました。")
    except Exception as e:
        _update_job(
            job_id,
            status="failed",
            error=str(e),
            updated_at=datetime.now().isoformat(),
        )
        _append_job_log(job_id, f"失敗: {e}")


def _run_display_job(job_id: str, params: dict) -> None:
    try:
        _update_job(job_id, status="running")

        def progress_cb(message: str) -> None:
            _append_job_log(job_id, message)

        cached = _get_cached_display_result(params)
        if cached:
            _append_job_log(job_id, "一覧情報キャッシュを使用しました。")
            result = {
                "items": cached["items"],
                "total": cached["total"],
                "page": int(params.get("start_page", 1) or 1),
                "has_next": bool(cached.get("next_page_url")),
                "next_page_url": cached.get("next_page_url", ""),
                "uses_local_paging": bool(cached.get("uses_local_paging", False)),
            }
        else:
            result = fetch_for_display(params, progress_cb)
            _store_cached_display_result(params, result)
            _append_job_log(job_id, "一覧情報キャッシュを保存しました。")
        browse_id = _store_browse_result(
            result["items"],
            params,
            {
                "page": result.get("page", 1),
                "has_next": result.get("has_next", False),
                "next_page_url": result.get("next_page_url", ""),
                "uses_local_paging": bool(result.get("uses_local_paging", False)),
                "history": [],
                "page_cache": {
                    str(result.get("page", 1)): {
                        "items": result["items"],
                        "page": result.get("page", 1),
                        "has_next": result.get("has_next", False),
                        "next_page_url": result.get("next_page_url", ""),
                    }
                },
            },
        )
        _update_job(
            job_id,
            status="completed",
            result={"browse_id": browse_id, "total": result["total"]},
            updated_at=datetime.now().isoformat(),
        )
        _append_job_log(job_id, f"完了: 表示用に {result['total']} 件取得しました。")
    except Exception as e:
        _update_job(
            job_id,
            status="failed",
            error=str(e),
            updated_at=datetime.now().isoformat(),
        )
        _append_job_log(job_id, f"失敗: {e}")


def _run_download_job(job_id: str, api_key: str, models: list[dict]) -> None:
    try:
        _update_job(job_id, status="running")
        downloaded = 0
        skipped = 0
        for index, model in enumerate(models, start=1):
            name = model.get("name") or f"model_{index}"
            _append_job_log(job_id, f"{index}/{len(models)}: {name} を処理中...")
            save_path, state = _download_model_file(api_key, model)
            if state == "exists":
                skipped += 1
                _append_job_log(job_id, f"既存のためスキップ: {os.path.basename(save_path)}")
            else:
                downloaded += 1
                _append_job_log(job_id, f"保存: {save_path}")

        _update_job(
            job_id,
            status="completed",
            result={"downloaded": downloaded, "skipped": skipped, "total": len(models)},
            updated_at=datetime.now().isoformat(),
        )
        _append_job_log(job_id, f"完了: 新規 {downloaded} 件、既存スキップ {skipped} 件です。")
    except Exception as e:
        _update_job(
            job_id,
            status="failed",
            error=str(e),
            updated_at=datetime.now().isoformat(),
        )
        _append_job_log(job_id, f"失敗: {e}")


def _browse_models_from_request() -> tuple[str, list[dict]]:
    browse_id = request.form.get("browse_id", "").strip()
    browse_result = _get_browse_result(browse_id)
    if not browse_result:
        abort(404)

    selected = request.form.getlist("selected_indexes")
    indexes = []
    for value in selected:
        try:
            indexes.append(int(value))
        except ValueError:
            continue
    items = browse_result["items"]
    models = [items[i] for i in indexes if 0 <= i < len(items)]
    return browse_id, models


def _models_context() -> dict:
    db_path = _ensure_main_db()
    view_mode = request.args.get("view", session.get("last_view_mode", "browse")).strip() or "browse"
    keyword = request.args.get("q", "").strip()
    model_type = request.args.get("model_type", "").strip()
    base_model = request.args.get("base_model", "").strip()
    page = max(request.args.get("page", default=1, type=int), 1)
    db_per_page = max(min(request.args.get("db_per_page", default=DEFAULT_PER_PAGE, type=int), 200), 1)
    current_job_id = request.args.get("job", "").strip()
    browse_id = request.args.get("browse", session.get("last_browse_id", "")).strip()
    current_job = _job_snapshot(current_job_id) if current_job_id else None
    if not browse_id and current_job and current_job.get("result"):
        browse_id = (current_job["result"] or {}).get("browse_id", "") or ""
    elif current_job and current_job.get("status") in ("queued", "running") and not request.args.get("browse", "").strip():
        browse_id = ""

    fetch_form = _base_fetch_form()
    fetch_form.update({
        "kw_mode": request.args.get("fetch_kw_mode", fetch_form["kw_mode"]).strip() or fetch_form["kw_mode"],
        "keyword": request.args.get("fetch_keyword", fetch_form["keyword"]).strip(),
        "selected_type": request.args.get("fetch_selected_type", fetch_form["selected_type"]).strip() or fetch_form["selected_type"],
        "selected_base": request.args.get("fetch_selected_base", fetch_form["selected_base"]).strip() or fetch_form["selected_base"],
        "sort": request.args.get("fetch_sort", fetch_form["sort"]).strip() or fetch_form["sort"],
        "period": request.args.get("fetch_period", fetch_form["period"]).strip() or fetch_form["period"],
        "start_page": max(request.args.get("fetch_start_page", default=fetch_form["start_page"], type=int), 1),
        "limit": max(min(request.args.get("fetch_limit", default=fetch_form["limit"], type=int), 100), 1),
        "wait_time": max(request.args.get("fetch_wait_time", default=fetch_form["wait_time"], type=float), 0),
        "mode": request.args.get("fetch_mode", fetch_form["mode"]).strip() or fetch_form["mode"],
    })
    browse_local_page = max(request.args.get("browse_local_page", default=1, type=int), 1)
    browse_result = _get_browse_result(browse_id)
    meta_sort_fields = ""
    meta_sort_dirs = ""
    if browse_result:
        meta_sort_fields = str(browse_result.get("meta", {}).get("table_sort", "") or "")
        meta_sort_dirs = str(browse_result.get("meta", {}).get("table_dir", "") or "")
    table_sorts = _normalize_table_sorts(
        request.args.get("table_sort", meta_sort_fields).strip(),
        request.args.get("table_dir", meta_sort_dirs).strip(),
    )
    table_sort_fields_param, table_sort_dirs_param = _encode_table_sorts(table_sorts)
    table_sort_map = _table_sort_map(table_sorts)

    def toggle_sort_values(field: str) -> tuple[str, str]:
        return _encode_table_sorts(_toggle_table_sort(table_sorts, field))

    ui_settings = _load_ui_settings()
    if view_mode == "db":
        result = search_models(
            db_path,
            keyword=keyword,
            model_type=model_type,
            base_model=base_model,
            page=page,
            per_page=db_per_page,
            sorts=table_sorts,
        )
        result["items"] = [_decorate_media_item(item, ui_settings) for item in result["items"]]
        filters = get_filter_options(db_path)
    else:
        result = {"items": [], "total": 0, "page": 1, "pages": 1}
        filters = {"model_types": [], "base_models": []}
    browse_view = browse_result
    if browse_result:
        meta = dict(browse_result.get("meta") or {})
        if meta.get("table_sort") != table_sort_fields_param or meta.get("table_dir") != table_sort_dirs_param:
            meta["table_sort"] = table_sort_fields_param
            meta["table_dir"] = table_sort_dirs_param
            browse_result["meta"] = meta
            _save_browse_result(browse_id, browse_result)
        source_items = list(browse_result.get("raw_items") or browse_result.get("items", []))
        raw_items = _sort_model_items(source_items, table_sorts)
        uses_local_paging = bool(
            browse_result.get("meta", {}).get("uses_local_paging")
            or fetch_form["mode"] == "loop"
            or fetch_form["kw_mode"] == "Name"
        )
        browse_result["meta"]["uses_local_paging"] = uses_local_paging
        if uses_local_paging:
            per_page = max(fetch_form["limit"], 1)
            total_items = len(raw_items)
            total_pages = max((total_items + per_page - 1) // per_page, 1)
            browse_local_page = min(browse_local_page, total_pages)
            start = (browse_local_page - 1) * per_page
            end = start + per_page
            indexed_models = list(enumerate(raw_items[start:end], start=start))
            browse_result["meta"]["local_page"] = browse_local_page
            browse_result["meta"]["local_total_pages"] = total_pages
            browse_result["meta"]["local_total_items"] = total_items
        else:
            indexed_models = list(enumerate(raw_items))

        browse_items = []
        for index, model in indexed_models:
            item = model_to_row(model, include_video=ui_settings.get("include_video", True)) if model.get("modelVersions") is not None or model.get("type") is not None else dict(model)
            item["browse_index"] = index
            item["browse_id"] = browse_result["id"]
            item["source"] = "browse"
            browse_items.append(_decorate_media_item(item, ui_settings))
        browse_view = dict(browse_result)
        browse_view["items"] = browse_items

    session["last_view_mode"] = view_mode
    if browse_view:
        session["last_browse_id"] = browse_view["id"]

    return {
        "db_name": os.path.basename(db_path),
        "view_mode": view_mode,
        "view_mode_options": VIEW_MODE_OPTIONS,
        "keyword": keyword,
        "model_type": model_type,
        "base_model": base_model,
        "db_per_page": db_per_page,
        "filters": filters,
        "result": result,
        "fetch_form": fetch_form,
        "sort_options": SORT_OPTIONS,
        "period_options": PERIOD_OPTIONS,
        "mode_options": MODE_OPTIONS,
        "latest_jobs": _latest_jobs(),
        "current_job": current_job,
        "browse_result": browse_view,
        "ui_settings": ui_settings,
        "table_sorts": table_sorts,
        "table_sort_map": table_sort_map,
        "table_sort_fields_param": table_sort_fields_param,
        "table_sort_dirs_param": table_sort_dirs_param,
        "toggle_sort_values": toggle_sort_values,
    }


@app.route("/")
def index():
    return redirect(url_for("models"))


@app.post("/app/restart")
def app_restart():
    restart_app_process()
    return """
    <!doctype html>
    <html lang="ja">
    <head>
      <meta charset="utf-8">
      <meta http-equiv="refresh" content="4;url=http://127.0.0.1:5000/models">
      <title>GUI再起動中</title>
      <style>
        body { font-family: "Yu Gothic UI", sans-serif; background:#f3f0e8; color:#2d241c; padding:40px; }
        .box { max-width:720px; margin:0 auto; background:#fffdf8; border:1px solid #d8cfbf; border-radius:18px; padding:28px; }
      </style>
    </head>
    <body>
      <div class="box">
        <h1 style="margin-top:0;">GUIを再起動しています</h1>
        <p>数秒後に一覧ページへ戻ります。</p>
      </div>
    </body>
    </html>
    """


@app.route("/settings", methods=["GET", "POST"])
def settings():
    api_key = load_api_key(APP_DIR)
    ui_settings = _load_ui_settings()
    maintenance_settings = _load_maintenance_settings()
    dropdown_cache_values = _load_dropdown_cache_values()
    model_types = _available_model_types()
    return_view = request.args.get("return_view", session.get("last_view_mode", "browse")).strip() or "browse"
    return_browse = request.args.get("return_browse", session.get("last_browse_id", "")).strip()
    if request.method == "POST":
        return_view = request.form.get("return_view", return_view).strip() or return_view
        return_browse = request.form.get("return_browse", return_browse).strip()
        action = request.form.get("settings_action", "save").strip()
        if action == "clear_info_cache":
            removed = _clear_info_cache()
            flash(f"一覧情報キャッシュを削除しました。{removed} 件です。", "success")
            return redirect(url_for("settings", return_view=return_view, return_browse=return_browse))
        if action in {"run_image_maintenance", "run_video_maintenance"}:
            run_image = action == "run_image_maintenance"
            run_video = action == "run_video_maintenance"
            image_format = request.form.get("maintenance_image_format", maintenance_settings["image_format"]).strip().lower()
            if image_format not in {"webp", "jpg"}:
                image_format = maintenance_settings["image_format"]
            try:
                image_quality = int(request.form.get("maintenance_image_quality", str(maintenance_settings["image_quality"])))
            except ValueError:
                image_quality = maintenance_settings["image_quality"]
            image_quality = max(1, min(image_quality, 100))
            image_keep_exif = request.form.get("maintenance_image_keep_exif") == "1"
            image_skip_processed = request.form.get("maintenance_image_skip_processed") == "1"

            video_format = request.form.get("maintenance_video_format", maintenance_settings["video_format"]).strip().lower()
            if video_format not in {"mp4", "webm"}:
                video_format = maintenance_settings["video_format"]
            video_quality = request.form.get("maintenance_video_quality", maintenance_settings["video_quality"]).strip().lower()
            if video_quality not in {"high", "medium", "low"}:
                video_quality = maintenance_settings["video_quality"]
            video_resolution = request.form.get("maintenance_video_resolution", maintenance_settings["video_resolution"]).strip().lower()
            if video_resolution not in {"original", "720", "512"}:
                video_resolution = maintenance_settings["video_resolution"]
            video_keep_audio = request.form.get("maintenance_video_keep_audio", "1") == "1"
            video_skip_processed = request.form.get("maintenance_video_skip_processed") == "1"
            _store_maintenance_settings({
                "image_format": image_format,
                "image_quality": image_quality,
                "image_keep_exif": image_keep_exif,
                "image_skip_processed": image_skip_processed,
                "video_format": video_format,
                "video_quality": video_quality,
                "video_resolution": video_resolution,
                "video_keep_audio": video_keep_audio,
                "video_skip_processed": video_skip_processed,
            })

            try:
                if run_image:
                    image_result = _run_image_maintenance(DB_MEDIA_DIR, image_format, image_quality, image_keep_exif, image_skip_processed)
                    flash(
                        f"画像メンテナンスを実行しました。 {image_result['processed']}件処理 / {image_result['skipped']}件スキップ / {image_result['failed']}件失敗。",
                        "success",
                    )
                    for line in image_result.get("samples", []):
                        flash(f"画像処理エラー: {line}", "error")
                if run_video:
                    video_result = _run_video_maintenance(DB_MEDIA_DIR, video_format, video_quality, video_resolution, video_keep_audio, video_skip_processed)
                    flash(
                        f"動画メンテナンスを実行しました。 {video_result['processed']}件処理 / {video_result['skipped']}件スキップ / {video_result['failed']}件失敗。",
                        "success",
                    )
                    if video_result.get("ffmpeg_missing"):
                        flash("動画処理は FFmpeg が見つからないためスキップしました。", "error")
                    for line in video_result.get("samples", []):
                        flash(f"動画処理エラー: {line}", "error")
            except Exception as exc:
                flash(f"メンテナンスの実行に失敗しました: {exc}", "error")
            return redirect(url_for("settings", return_view=return_view, return_browse=return_browse))
        api_key = request.form.get("api_key", "").strip()
        manual_types = request.form.get("manual_types", "").strip()
        manual_bases = request.form.get("manual_bases", "").strip()
        nsfw_enabled = request.form.get("nsfw_enabled") == "1"
        include_video = request.form.get("include_video") == "1"
        info_cache_enabled = request.form.get("info_cache_enabled") == "1"
        download_preview_enabled = request.form.get("download_preview_enabled") == "1"
        download_civitai_info_enabled = request.form.get("download_civitai_info_enabled") == "1"
        info_cache_file = request.form.get("info_cache_file", "").strip() or ui_settings["info_cache_file"]
        download_root_dir = request.form.get("download_root_dir", "").strip() or ui_settings["download_root_dir"]
        download_type_dirs = {}
        for model_type in model_types:
            field_name = f"download_type_dir__{model_type}"
            current_value = (ui_settings.get("download_type_dirs") or {}).get(model_type) or os.path.join(download_root_dir, model_type)
            download_type_dirs[model_type] = request.form.get(field_name, "").strip() or current_value

        os.makedirs(os.path.dirname(info_cache_file) or APP_DIR, exist_ok=True)
        if download_root_dir:
            os.makedirs(download_root_dir, exist_ok=True)
        for target_dir in download_type_dirs.values():
            if target_dir:
                os.makedirs(target_dir, exist_ok=True)

        saved_types, saved_bases = _replace_dropdown_cache_values(edited_types=manual_types, edited_bases=manual_bases)
        save_api_key(APP_DIR, api_key)
        _save_ui_settings({
            "nsfw_enabled": nsfw_enabled,
            "include_video": include_video,
            "info_cache_enabled": info_cache_enabled,
            "download_preview_enabled": download_preview_enabled,
            "download_civitai_info_enabled": download_civitai_info_enabled,
            "info_cache_file": info_cache_file,
            "download_root_dir": download_root_dir,
            "download_type_dirs": download_type_dirs,
        })
        flash(f"設定を保存しました。 Types {saved_types} 件、Base models {saved_bases} 件です。", "success")
        if return_view == "browse" and return_browse:
            return redirect(url_for("models", view="browse", browse=return_browse))
        if return_view == "bulk":
            return redirect(url_for("db_fetch"))
        if return_view == "settings":
            return redirect(url_for("settings"))
        return redirect(url_for("models", view=return_view))
    model_type_downloads = [
        {
            "type": model_type,
            "field_name": f"download_type_dir__{model_type}",
            "value": (ui_settings.get("download_type_dirs") or {}).get(model_type) or os.path.join(ui_settings["download_root_dir"], model_type),
        }
        for model_type in model_types
    ]
    return render_template(
        "settings.html",
        api_key=api_key,
        ui_settings=ui_settings,
        maintenance_settings=maintenance_settings,
        dropdown_cache_values=dropdown_cache_values,
        dropdown_cache_text={
            "types": "\n".join(dropdown_cache_values["types"]),
            "bases": "\n".join(dropdown_cache_values["bases"]),
        },
        return_view=return_view,
        return_browse=return_browse,
        model_type_downloads=model_type_downloads,
    )




@app.route("/models")
def models():
    return render_template("models.html", **_models_context())


@app.route("/db-fetch")
def db_fetch():
    current_job_id = request.args.get("job", "").strip()
    current_job = _job_snapshot(current_job_id) if current_job_id else None
    fetch_form = _fetch_form_from_request_args()
    session["last_view_mode"] = "bulk"
    return render_template(
        "db_fetch.html",
        fetch_form=fetch_form,
        sort_options=SORT_OPTIONS,
        period_options=PERIOD_OPTIONS,
        mode_options=MODE_OPTIONS,
        current_job=current_job,
    )


@app.post("/models/fetch")
def start_fetch():
    api_key = load_api_key(APP_DIR)
    if not api_key:
        flash("先に設定ページでAPIキーを保存してください。", "error")
        return redirect(url_for("settings"))

    fetch_form = _base_fetch_form()
    view_mode = request.form.get("view_mode", "browse").strip() or "browse"
    fetch_submit = request.form.get("fetch_submit", "display").strip() or "display"
    if view_mode == "bulk":
        fetch_submit = "save"
    fetch_form.update({
        "kw_mode": request.form.get("kw_mode", fetch_form["kw_mode"]).strip() or fetch_form["kw_mode"],
        "keyword": request.form.get("fetch_keyword", fetch_form["keyword"]).strip(),
        "selected_type": request.form.get("fetch_selected_type", fetch_form["selected_type"]).strip() or fetch_form["selected_type"],
        "selected_base": request.form.get("fetch_selected_base", fetch_form["selected_base"]).strip() or fetch_form["selected_base"],
        "sort": request.form.get("fetch_sort", fetch_form["sort"]).strip() or fetch_form["sort"],
        "period": request.form.get("fetch_period", fetch_form["period"]).strip() or fetch_form["period"],
        "start_page": max(request.form.get("fetch_start_page", type=int, default=fetch_form["start_page"]), 1),
        "limit": max(min(request.form.get("fetch_limit", type=int, default=fetch_form["limit"]), 100), 1),
        "wait_time": max(request.form.get("fetch_wait_time", type=float, default=fetch_form["wait_time"]), 0),
        "mode": request.form.get("fetch_mode", fetch_form["mode"]).strip() or fetch_form["mode"],
    })

    job_id = uuid.uuid4().hex[:10]
    job = {
        "id": job_id,
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "params": dict(fetch_form),
        "logs": ["ジョブを作成しました。"],
    }
    with JOBS_LOCK:
        JOBS[job_id] = job

    job_params = {
        "api_key": api_key,
        "kw_mode": fetch_form["kw_mode"],
        "keyword": fetch_form["keyword"],
        "selected_type": fetch_form["selected_type"],
        "selected_base": fetch_form["selected_base"],
        "sort": fetch_form["sort"],
        "period": fetch_form["period"],
        "nsfw_enabled": _load_ui_settings().get("nsfw_enabled", False),
        "include_video": _load_ui_settings().get("include_video", True),
        "start_page": fetch_form["start_page"],
        "limit": fetch_form["limit"],
        "wait_time": fetch_form["wait_time"],
        "loop": fetch_form["mode"] == "loop",
    }

    if view_mode == "browse" and fetch_submit == "display":
        cached = _get_cached_display_result(job_params)
        if cached:
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "completed"
                JOBS[job_id]["result"] = {
                    "total": cached["total"],
                }
                JOBS[job_id]["updated_at"] = datetime.now().isoformat()
            _append_job_log(job_id, "キャッシュを表示しました。")
            _append_job_log(job_id, f"完了: 表示用に {cached['total']} 件取得しました。")
            browse_id = _store_browse_result(
                cached["items"],
                job_params,
                {
                    "page": int(job_params.get("start_page", 1) or 1),
                    "has_next": bool(cached.get("next_page_url")),
                    "next_page_url": cached.get("next_page_url", ""),
                    "uses_local_paging": bool(cached.get("uses_local_paging", False)),
                    "history": [],
                    "page_cache": {
                        str(int(job_params.get("start_page", 1) or 1)): {
                            "items": cached["items"],
                            "page": int(job_params.get("start_page", 1) or 1),
                            "has_next": bool(cached.get("next_page_url")),
                            "next_page_url": cached.get("next_page_url", ""),
                        }
                    },
                },
            )
            return redirect(url_for(
                "models",
                view=view_mode,
                job=job_id,
                browse=browse_id,
                fetch_kw_mode=fetch_form["kw_mode"],
                fetch_keyword=fetch_form["keyword"],
                fetch_selected_type=fetch_form["selected_type"],
                fetch_selected_base=fetch_form["selected_base"],
                fetch_sort=fetch_form["sort"],
                fetch_period=fetch_form["period"],
                fetch_start_page=fetch_form["start_page"],
                fetch_limit=fetch_form["limit"],
                fetch_wait_time=fetch_form["wait_time"],
                fetch_mode=fetch_form["mode"],
            ))

    target = _run_display_job if fetch_submit == "display" else _run_fetch_job
    threading.Thread(target=target, args=(job_id, job_params), daemon=True).start()

    redirect_endpoint = "db_fetch" if view_mode == "bulk" else "models"
    redirect_params = {
        "job": job_id,
        "fetch_kw_mode": fetch_form["kw_mode"],
        "fetch_keyword": fetch_form["keyword"],
        "fetch_selected_type": fetch_form["selected_type"],
        "fetch_selected_base": fetch_form["selected_base"],
        "fetch_sort": fetch_form["sort"],
        "fetch_period": fetch_form["period"],
        "fetch_start_page": fetch_form["start_page"],
        "fetch_limit": fetch_form["limit"],
        "fetch_wait_time": fetch_form["wait_time"],
        "fetch_mode": fetch_form["mode"],
    }
    if redirect_endpoint == "models":
        redirect_params["view"] = view_mode
    return redirect(url_for(redirect_endpoint, **redirect_params))


@app.post("/models/browse-page")
def browse_page():
    browse_id = request.form.get("browse_id", "").strip()
    direction = request.form.get("direction", "").strip()
    browse_result = _get_browse_result(browse_id)
    if not browse_result:
        abort(404)

    meta = dict(browse_result.get("meta") or {})
    params = dict(browse_result.get("params") or {})
    current_page = int(meta.get("page", 1) or 1)
    page_cache = _normalize_page_cache(meta)
    table_sorts = _normalize_table_sorts(
        str(meta.get("table_sort", "") or ""),
        str(meta.get("table_dir", "") or ""),
    )
    table_sort_fields, table_sort_dirs = _encode_table_sorts(table_sorts)

    if direction == "prev":
        target_page = current_page - 1
        cached_page = page_cache.get(str(target_page))
        if not cached_page:
            return redirect(url_for("models", view="browse", browse=browse_id, table_sort=table_sort_fields, table_dir=table_sort_dirs))
        browse_result["items"] = cached_page.get("items", [])
        browse_result["raw_items"] = cached_page.get("items", [])
        meta["page"] = cached_page.get("page", target_page)
        meta["has_next"] = cached_page.get("has_next", False)
        meta["next_page_url"] = cached_page.get("next_page_url", "")
        browse_result["meta"] = meta
        _save_browse_result(browse_id, browse_result)
        return redirect(url_for("models", view="browse", browse=browse_id, table_sort=table_sort_fields, table_dir=table_sort_dirs))

    if direction != "next":
        abort(400)

    target_page = current_page + 1
    cached_page = page_cache.get(str(target_page))
    if cached_page:
        browse_result["items"] = cached_page.get("items", [])
        browse_result["raw_items"] = cached_page.get("items", [])
        meta["page"] = cached_page.get("page", target_page)
        meta["has_next"] = cached_page.get("has_next", False)
        meta["next_page_url"] = cached_page.get("next_page_url", "")
        browse_result["meta"] = meta
        _save_browse_result(browse_id, browse_result)
        return redirect(url_for("models", view="browse", browse=browse_id, table_sort=table_sort_fields, table_dir=table_sort_dirs))

    next_page_url = (meta.get("next_page_url") or "").strip()
    if not next_page_url:
        return redirect(url_for("models", view="browse", browse=browse_id, table_sort=table_sort_fields, table_dir=table_sort_dirs))

    api_key = load_api_key(APP_DIR)
    items, further_next_page_url = fetch_one_page(
        api_key=api_key,
        limit=params["limit"],
        page=target_page,
        base_model=params["selected_base"],
        keyword=params["keyword"],
        kw_mode=params["kw_mode"],
        model_type=params["selected_type"],
        sort=params["sort"],
        period=params["period"],
        nsfw_enabled=bool(params.get("nsfw_enabled", False)),
        next_page_url=next_page_url,
        allow_query=params["kw_mode"] == "Name",
    )

    browse_result["items"] = items
    browse_result["raw_items"] = items
    meta["page"] = target_page
    meta["has_next"] = bool(further_next_page_url)
    meta["next_page_url"] = further_next_page_url or ""
    page_cache[str(target_page)] = {
        "items": items,
        "page": target_page,
        "has_next": bool(further_next_page_url),
        "next_page_url": further_next_page_url or "",
    }
    meta["page_cache"] = page_cache
    browse_result["meta"] = meta
    _save_browse_result(browse_id, browse_result)
    return redirect(url_for("models", view="browse", browse=browse_id, table_sort=table_sort_fields, table_dir=table_sort_dirs))


@app.post("/models/refresh-options")
def refresh_options():
    api_key = load_api_key(APP_DIR)
    if not api_key:
        flash("先に設定ページでAPIキーを保存してください。", "error")
        return redirect(url_for("settings"))
    refresh_option_cache(CACHE_FILE, api_key, is_manual_refresh=True, nsfw_enabled=_load_ui_settings().get("nsfw_enabled", False))
    flash("タイプとベースモデルの候補を更新しました。", "success")
    target_view = request.form.get("view_mode", "browse").strip() or "browse"
    if target_view == "bulk":
        return redirect(url_for("db_fetch"))
    return redirect(url_for("models", view=target_view))


@app.post("/models/<int:row_id>/refresh")
def refresh_model(row_id: int):
    db_path = _ensure_main_db()
    model = get_model_by_row_id(db_path, row_id)
    if not model:
        abort(404)

    api_key = load_api_key(APP_DIR)
    if not api_key:
        flash("先に設定ページでAPIキーを保存してください。", "error")
        return redirect(url_for("settings"))

    latest = fetch_model_by_id(api_key, model["civitai_model_id"])
    include_video = _load_ui_settings().get("include_video", True)
    save_models(db_path, [latest], include_video=include_video)
    save_media_for_models(db_path, DB_MEDIA_DIR, [latest], include_video=include_video)
    flash(f"{model['name']} を更新しました。", "success")

    redirect_params = {
        "view": "db",
        "q": request.form.get("q", "").strip(),
        "model_type": request.form.get("model_type", "").strip(),
        "base_model": request.form.get("base_model", "").strip(),
        "db_per_page": request.form.get("db_per_page", type=int, default=DEFAULT_PER_PAGE),
        "page": request.form.get("page", type=int, default=1),
        "table_sort": request.form.get("table_sort", "").strip(),
        "table_dir": request.form.get("table_dir", "asc").strip() or "asc",
    }
    return redirect(url_for("models", **redirect_params))


@app.post("/models/save-one")
def save_one_from_browse():
    browse_id = request.form.get("browse_id", "").strip()
    browse_index = request.form.get("browse_index", type=int)
    browse_result = _get_browse_result(browse_id)
    if not browse_result or browse_index is None:
        abort(404)
    items = browse_result["items"]
    if browse_index < 0 or browse_index >= len(items):
        abort(404)
    model = items[browse_index]
    db_path = _ensure_main_db()
    include_video = _load_ui_settings().get("include_video", True)
    save_models(db_path, [model], include_video=include_video)
    save_media_for_models(db_path, DB_MEDIA_DIR, [model], include_video=include_video)
    flash(f"{model.get('name', 'モデル')} をDBへ保存しました。", "success")
    return redirect(url_for("models", **_browse_redirect_params(browse_id)))


@app.post("/models/save-selected")
def save_selected_from_browse():
    browse_id, models = _browse_models_from_request()
    if not models:
        flash("保存対象が選択されていません。", "error")
        return redirect(url_for("models", **_browse_redirect_params(browse_id)))
    db_path = _ensure_main_db()
    include_video = _load_ui_settings().get("include_video", True)
    save_models(db_path, models, include_video=include_video)
    save_media_for_models(db_path, DB_MEDIA_DIR, models, include_video=include_video)
    flash(f"{len(models)} 件をDBへ保存しました。", "success")
    return redirect(url_for("models", **_browse_redirect_params(browse_id)))


@app.post("/models/download-one")
def download_one_from_browse():
    api_key = load_api_key(APP_DIR)
    browse_id = request.form.get("browse_id", "").strip()
    browse_index = request.form.get("browse_index", type=int)
    browse_result = _get_browse_result(browse_id)
    if not browse_result or browse_index is None:
        abort(404)
    items = browse_result["items"]
    if browse_index < 0 or browse_index >= len(items):
        abort(404)

    job_id = uuid.uuid4().hex[:10]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "logs": ["ダウンロードジョブを作成しました。"],
        }
    threading.Thread(target=_run_download_job, args=(job_id, api_key, [items[browse_index]]), daemon=True).start()
    return redirect(url_for("models", **_browse_redirect_params(browse_id, job=job_id)))


@app.post("/models/download-selected")
def download_selected_from_browse():
    api_key = load_api_key(APP_DIR)
    browse_id, models = _browse_models_from_request()
    if not models:
        flash("ダウンロード対象が選択されていません。", "error")
        return redirect(url_for("models", **_browse_redirect_params(browse_id)))

    job_id = uuid.uuid4().hex[:10]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "logs": ["ダウンロードジョブを作成しました。"],
        }
    threading.Thread(target=_run_download_job, args=(job_id, api_key, models), daemon=True).start()
    return redirect(url_for("models", **_browse_redirect_params(browse_id, job=job_id)))


@app.route("/media/<int:row_id>")
def model_media(row_id: int):
    db_path = _ensure_main_db()
    item = get_model_by_row_id(db_path, row_id)
    if not item:
        abort(404)
    local_path = (item.get("thumbnail_local_path") or "").strip()
    if local_path and os.path.exists(local_path):
        return send_file(local_path, mimetype=_guess_file_mimetype(local_path))
    abort(404)


def main():
    parser = argparse.ArgumentParser(description="Run the Civitai Flask UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--skip-browser", action="store_true")
    args = parser.parse_args()
    app.config["APP_HOST"] = args.host
    app.config["APP_PORT"] = args.port
    if not args.skip_browser:
        threading.Thread(target=_open_browser_later, args=(args.host, args.port), daemon=True).start()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
