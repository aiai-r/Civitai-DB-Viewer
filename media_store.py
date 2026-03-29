import os

import requests

from sqlite_store import model_to_row, update_model_media_path


def _content_type_to_ext(content_type: str, fallback_ext: str) -> str:
    content_type = (content_type or "").split(";")[0].strip().lower()
    mapping = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "video/mp4": ".mp4",
        "video/webm": ".webm",
        "video/quicktime": ".mov",
    }
    return mapping.get(content_type, fallback_ext)


def _guess_fallback_ext(item: dict) -> str:
    media_type = (item.get("thumbnail_media_type") or "").lower()
    url = (item.get("thumbnail_url") or "").lower()
    if media_type == "video" or url.endswith((".mp4", ".webm", ".mov")):
        return ".mp4"
    if url.endswith(".png"):
        return ".png"
    if url.endswith(".webp"):
        return ".webp"
    if url.endswith(".gif"):
        return ".gif"
    return ".jpg"


def _find_existing_media(media_dir: str, model_id: str) -> str | None:
    if not os.path.isdir(media_dir):
        return None
    prefix = f"{model_id}."
    for name in os.listdir(media_dir):
        if name.startswith(prefix):
            path = os.path.join(media_dir, name)
            if os.path.isfile(path):
                return path
    return None


def save_media_for_models(db_path: str, media_dir: str, models: list[dict], progress_cb=None, include_video: bool = True) -> int:
    os.makedirs(media_dir, exist_ok=True)
    saved = 0
    for model in models:
        item = model_to_row(model, include_video=include_video)
        model_id = str(item.get("civitai_model_id") or "")
        url = (item.get("thumbnail_url") or "").strip()
        if not model_id or not url:
            continue

        existing_path = _find_existing_media(media_dir, model_id)
        if existing_path:
            update_model_media_path(db_path, model_id, existing_path)
            continue

        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        ext = _content_type_to_ext(response.headers.get("content-type", ""), _guess_fallback_ext(item))
        save_path = os.path.join(media_dir, f"{model_id}{ext}")
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
        update_model_media_path(db_path, model_id, save_path)
        saved += 1
        if progress_cb:
            progress_cb(f"メディア保存: {model.get('name', model_id)}")
    return saved
