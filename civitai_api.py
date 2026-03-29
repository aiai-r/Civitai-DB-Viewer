import json
import os
import time
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError, ConnectTimeout, HTTPError, ReadTimeout


API_CONNECT_TIMEOUT = 10
API_READ_TIMEOUT = 60
API_RETRIES = 3
HTTP_POOL_SIZE = min(12, max(4, (os.cpu_count() or 4) * 2))

GLOBAL_ADAPTER = HTTPAdapter(pool_connections=HTTP_POOL_SIZE * 2,
                             pool_maxsize=HTTP_POOL_SIZE * 4)
SESSION = requests.Session()
SESSION.mount("https://", GLOBAL_ADAPTER)
SESSION.mount("http://", GLOBAL_ADAPTER)


def load_api_key(app_dir: str) -> str:
    config_path = os.path.join(app_dir, "api_config.json")
    if not os.path.exists(config_path):
        return ""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f).get("api_key", "").strip()


def save_api_key(app_dir: str, api_key: str) -> None:
    config_path = os.path.join(app_dir, "api_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({"api_key": api_key.strip()}, f, ensure_ascii=False, indent=2)


def robust_get_json(url: str, *, headers=None, params=None, label="api") -> dict:
    last_err = None
    for i in range(1, API_RETRIES + 1):
        try:
            response = SESSION.get(
                url,
                headers=headers,
                params=params,
                timeout=(API_CONNECT_TIMEOUT, API_READ_TIMEOUT),
            )
            response.raise_for_status()
            return response.json()
        except (ReadTimeout, ConnectTimeout, ConnectionError, HTTPError) as e:
            last_err = e
            if i == API_RETRIES:
                break
    raise last_err if last_err else RuntimeError(f"{label} error")


def fetch_model_by_id(api_key: str, model_id: str) -> dict:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    return robust_get_json(
        f"https://civitai.com/api/v1/models/{model_id}",
        headers=headers,
        label=f"model_{model_id}",
    )


def fetch_one_page(api_key, *, limit, page, base_model,
                   keyword, kw_mode, model_type, sort, period, nsfw_enabled=False,
                   next_page_url=None, allow_query=False):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    if next_page_url:
        url, params = next_page_url, None
    else:
        url = "https://civitai.com/api/v1/models"
        params = dict(limit=limit)

        if not (kw_mode == "Name" and keyword and allow_query):
            params["page"] = page

        if model_type and model_type != "All":
            params["types"] = model_type

        if base_model and base_model != "All":
            params["baseModels"] = base_model

        if kw_mode == "Name" and keyword and allow_query:
            params["query"] = keyword
        elif kw_mode == "Tag" and keyword:
            params["tag"] = keyword

        params["sort"] = sort
        params["period"] = period
        params["nsfw"] = "true" if nsfw_enabled else "false"

    data = robust_get_json(url, headers=headers, params=params, label=f"page_{page}")
    items = data.get("items", [])
    next_page = data.get("metadata", {}).get("nextPage")
    return items, next_page


def scan_models_for_discovery(api_key, sort_method, pages=1, limit=100, nsfw_enabled=False):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    url = "https://civitai.com/api/v1/models"
    params = {"limit": limit, "page": 1, "sort": sort_method, "nsfw": "true" if nsfw_enabled else "false"}
    seen_types, seen_bases, fetched = set(), set(), 0
    while url and fetched < pages:
        data = robust_get_json(url, headers=headers, params=params if fetched == 0 else None,
                               label=f"discover_{sort_method}_{fetched+1}")
        items = data.get("items", [])
        for item in items:
            model_type = item.get("type", "")
            if model_type:
                seen_types.add(model_type)
            for model_version in item.get("modelVersions", []):
                base = model_version.get("baseModel", "")
                if base:
                    seen_bases.add(base)
        url = data.get("metadata", {}).get("nextPage")
        fetched += 1
    return seen_types, seen_bases


def update_tags_file(tags_file, api_key, progress_cb=None, max_pages=5000):
    limit = 200
    existing = set()
    if os.path.exists(tags_file):
        try:
            read_ok = False
            for enc in ("utf-8-sig", "utf-8", "cp932"):
                try:
                    with open(tags_file, "r", encoding=enc, newline="") as f:
                        import csv
                        reader = csv.DictReader(f)
                        for row in reader:
                            name = (row.get("name") or "").strip()
                            if name:
                                existing.add(name)
                    read_ok = True
                    break
                except UnicodeDecodeError:
                    continue
            if not read_ok:
                raise UnicodeDecodeError("utf-8", b"", 0, 1, "failed to decode")
        except Exception:
            pass

    new_rows = 0
    page = 1
    fetched_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    file_exists = os.path.exists(tags_file)
    with open(tags_file, "a", encoding="utf-8", newline="") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=["name", "modelCount", "link", "fetched_at"])
        if not file_exists:
            writer.writeheader()
        auth_modes = ["header", "token", "none"] if api_key else ["none"]
        chosen_mode = None

        def _tags_get_json(page_num, mode):
            params = {"limit": limit, "page": page_num}
            if mode == "header" and api_key:
                return robust_get_json(
                    "https://civitai.com/api/v1/tags",
                    headers={"Authorization": f"Bearer {api_key}"},
                    params=params,
                    label=f"tags_header_{page_num}"
                )
            if mode == "token" and api_key:
                return robust_get_json(
                    f"https://civitai.com/api/v1/tags?token={api_key}",
                    headers=None,
                    params=params,
                    label=f"tags_token_{page_num}"
                )
            return robust_get_json(
                "https://civitai.com/api/v1/tags",
                headers=None,
                params=params,
                label=f"tags_none_{page_num}"
            )

        while True:
            if progress_cb:
                progress_cb(f"タグ取得中... page {page}")
            if chosen_mode is None:
                last_err = None
                for mode in auth_modes:
                    try:
                        data = _tags_get_json(page, mode)
                        chosen_mode = mode
                        break
                    except Exception as e:
                        last_err = e
                if chosen_mode is None:
                    raise last_err
            else:
                data = _tags_get_json(page, chosen_mode)

            items = data.get("items", []) or []
            for item in items:
                name = item.get("name")
                if not name or str(name) in existing:
                    continue
                existing.add(str(name))
                writer.writerow({
                    "name": name,
                    "modelCount": item.get("modelCount", ""),
                    "link": item.get("link", ""),
                    "fetched_at": fetched_at,
                })
                new_rows += 1

            next_page = data.get("metadata", {}).get("nextPage")
            if next_page:
                page += 1
                if page > max_pages:
                    break
                continue
            if len(items) == limit and page < max_pages:
                page += 1
                continue
            break
    return new_rows
