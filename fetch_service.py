import csv
import json
import os
import time

from civitai_api import fetch_one_page
from media_store import save_media_for_models
from sqlite_store import init_db, load_existing_model_ids, save_models


SORT_OPTIONS = ["Highest Rated", "Most Downloaded", "Newest"]
PERIOD_OPTIONS = ["AllTime", "Year", "Month", "Week", "Day"]
MODE_OPTIONS = [("single", "シングルページ"), ("loop", "ループ取得（全件取得）")]
TAG_SUGGEST_MAX_RESULTS = 50


def load_tag_suggestions(tags_file: str) -> list[str]:
    if not os.path.exists(tags_file):
        return []
    try:
        tags = []
        for enc in ("utf-8-sig", "utf-8", "cp932"):
            try:
                with open(tags_file, "r", encoding=enc, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        name = (row.get("name") or "").strip()
                        if not name:
                            continue
                        try:
                            count_val = int(row.get("modelCount") or 0)
                        except Exception:
                            count_val = 0
                        tags.append((count_val, name))
                break
            except UnicodeDecodeError:
                tags = []
                continue
        tags.sort(key=lambda x: x[0], reverse=True)
        return [name for _, name in tags[:TAG_SUGGEST_MAX_RESULTS]]
    except Exception:
        return []


def refresh_option_cache(cache_file: str, api_key: str, is_manual_refresh=False, nsfw_enabled=False) -> dict:
    existing_types = set()
    existing_bases = set()
    known_model_ids = []
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
            existing_types = set(cache.get("types", []))
            existing_bases = set(cache.get("bases", []))
            known_model_ids = [str(v) for v in (cache.get("known_model_ids", []) or []) if str(v).strip()]
        except Exception:
            existing_types = set()
            existing_bases = set()
            known_model_ids = []

    if not is_manual_refresh and os.path.exists(cache_file) and (existing_types or existing_bases):
        final_types = sorted(existing_types)
        final_bases = sorted(existing_bases)
    else:
        items, _next_page = fetch_one_page(
            api_key=api_key,
            limit=100,
            page=1,
            base_model="All",
            keyword="",
            kw_mode="Tag",
            model_type="All",
            sort="Newest",
            period="AllTime",
            nsfw_enabled=nsfw_enabled,
        )
        known_id_set = set(known_model_ids)
        latest_known_ids = []
        new_types = set(existing_types)
        new_bases = set(existing_bases)
        for item in items:
            item_id = str(item.get("id") or "").strip()
            if item_id:
                latest_known_ids.append(item_id)
                if item_id in known_id_set:
                    break
            model_type = (item.get("type") or "").strip()
            if model_type:
                new_types.add(model_type)
            for model_version in item.get("modelVersions", []):
                base = (model_version.get("baseModel") or "").strip()
                if base:
                    new_bases.add(base)
        final_types = sorted(new_types)
        final_bases = sorted(new_bases)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "types": final_types,
                    "bases": final_bases,
                    "known_model_ids": latest_known_ids[:100],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    if not final_types:
        final_types = ["Checkpoint", "LORA"]
    if not final_bases:
        final_bases = ["SD 1.5", "SDXL 1.0"]
    return {"types": ["All", *final_types], "bases": ["All", "Other", *final_bases]}


def build_db_path(base_dir: str) -> str:
    return os.path.join(base_dir, "civitai_models.db")


def _extract_base_model(model: dict) -> str:
    if not model.get("modelVersions"):
        return ""
    return (model["modelVersions"][0].get("baseModel") or "").strip()


def _matches_filters(model: dict, *, selected_type: str, selected_base: str) -> bool:
    if selected_type and selected_type != "All":
        if (model.get("type") or "").strip() != selected_type:
            return False

    if selected_base and selected_base != "All":
        model_base = _extract_base_model(model)
        if selected_base == "Other":
            return model_base == "Other"
        if model_base != selected_base:
            return False

    return True


def fetch_and_store(base_dir: str, params: dict, progress_cb) -> dict:
    save_path = build_db_path(
        base_dir,
    )
    media_dir = os.path.join(base_dir, "db_media")
    init_db(save_path)
    existing_model_ids = load_existing_model_ids(save_path)
    progress_cb(f"保存先DBを準備しました: {os.path.relpath(save_path, base_dir)}")
    if existing_model_ids:
        progress_cb(f"既存 {len(existing_model_ids)} 件をスキップ対象として読み込みました。")

    current_page = params["start_page"]
    next_page_url = None
    total_saved = 0
    fetch_all_name_results = params["kw_mode"] == "Name" and bool((params.get("keyword") or "").strip())

    while True:
        progress_cb(f"ページ {current_page} を取得中...")
        items, next_page_url = fetch_one_page(
            api_key=params["api_key"],
            limit=params["limit"],
            page=current_page,
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

        original_count = len(items)
        filtered_items = [
            item for item in items
            if _matches_filters(
                item,
                selected_type=params["selected_type"],
                selected_base=params["selected_base"],
            )
        ]
        filtered_out = original_count - len(filtered_items)
        if filtered_out > 0:
            progress_cb(f"ページ {current_page}: 条件外 {filtered_out} 件を除外しました。")

        items = filtered_items

        if existing_model_ids:
            items = [m for m in items if str(m.get("id")) not in existing_model_ids]
            skipped = len(filtered_items) - len(items)
            if skipped > 0:
                progress_cb(f"ページ {current_page}: 既存 {skipped} 件をスキップしました。")

        if not items:
            progress_cb(f"ページ {current_page}: 新規データはありません。")
            if not (params["loop"] or fetch_all_name_results) or not next_page_url:
                break
            current_page += 1
            if params["wait_time"] > 0:
                time.sleep(params["wait_time"])
            continue

        saved_count = save_models(save_path, items, include_video=bool(params.get("include_video", True)))
        media_saved_count = save_media_for_models(
            save_path,
            media_dir,
            items,
            progress_cb=progress_cb,
            include_video=bool(params.get("include_video", True)),
        )
        total_saved += saved_count
        existing_model_ids.update(str(item.get("id")) for item in items)
        progress_cb(f"ページ {current_page}: {saved_count} 件保存、メディア {media_saved_count} 件保存。合計 {total_saved} 件です。")

        if not (params["loop"] or fetch_all_name_results) or not next_page_url:
            break
        if params["wait_time"] > 0:
            progress_cb(f"{params['wait_time']} 秒待機します。")
            time.sleep(params["wait_time"])
        current_page += 1

    return {
        "db_path": save_path,
        "db_name": os.path.basename(save_path),
        "total_saved": total_saved,
    }


def fetch_for_display(params: dict, progress_cb) -> dict:
    current_page = params["start_page"]
    next_page_url = None
    collected = []
    last_page = current_page
    fetch_all_name_results = params["kw_mode"] == "Name" and bool((params.get("keyword") or "").strip())

    if fetch_all_name_results:
        progress_cb("Name検索のため、ヒット結果を最後まで取得します。")

    while True:
        progress_cb(f"ページ {current_page} を取得中...")
        items, next_page_url = fetch_one_page(
            api_key=params["api_key"],
            limit=params["limit"],
            page=current_page,
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

        original_count = len(items)
        filtered_items = [
            item for item in items
            if _matches_filters(
                item,
                selected_type=params["selected_type"],
                selected_base=params["selected_base"],
            )
        ]
        filtered_out = original_count - len(filtered_items)
        if filtered_out > 0:
            progress_cb(f"ページ {current_page}: 条件外 {filtered_out} 件を除外しました。")

        collected.extend(filtered_items)
        progress_cb(f"ページ {current_page}: 表示対象 {len(filtered_items)} 件、合計 {len(collected)} 件です。")
        last_page = current_page

        if not (params["loop"] or fetch_all_name_results) or not next_page_url:
            break
        if params["wait_time"] > 0:
            progress_cb(f"{params['wait_time']} 秒待機します。")
            time.sleep(params["wait_time"])
        current_page += 1

    return {
        "items": collected,
        "total": len(collected),
        "page": last_page,
        "has_next": bool(next_page_url) and not fetch_all_name_results,
        "next_page_url": "" if fetch_all_name_results else (next_page_url or ""),
        "uses_local_paging": fetch_all_name_results or params["loop"],
    }
