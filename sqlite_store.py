import os
import sqlite3
from datetime import datetime, timezone


DB_SORT_FIELDS = {
    "name",
    "model_type",
    "base_model",
    "model_url",
    "allow_no_credit",
    "allow_sell_images",
    "allow_run_paid_service",
    "allow_share_merges",
    "allow_sell_model",
    "allow_different_licenses",
}


def _connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                civitai_model_id TEXT NOT NULL UNIQUE,
                name TEXT,
                model_type TEXT,
                base_model TEXT,
                model_url TEXT,
                thumbnail_url TEXT,
                thumbnail_local_path TEXT NOT NULL DEFAULT '',
                thumbnail_media_type TEXT,
                allow_no_credit INTEGER NOT NULL DEFAULT 0,
                allow_sell_images INTEGER NOT NULL DEFAULT 0,
                allow_run_paid_service INTEGER NOT NULL DEFAULT 0,
                allow_share_merges INTEGER NOT NULL DEFAULT 0,
                allow_sell_model INTEGER NOT NULL DEFAULT 0,
                allow_different_licenses INTEGER NOT NULL DEFAULT 0,
                fetched_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(models)").fetchall()
        }
        if "thumbnail_local_path" not in columns:
            conn.execute("ALTER TABLE models ADD COLUMN thumbnail_local_path TEXT NOT NULL DEFAULT ''")
        if "image_maintenance_params" not in columns:
            conn.execute("ALTER TABLE models ADD COLUMN image_maintenance_params TEXT NOT NULL DEFAULT ''")
        if "video_maintenance_params" not in columns:
            conn.execute("ALTER TABLE models ADD COLUMN video_maintenance_params TEXT NOT NULL DEFAULT ''")


def load_existing_model_ids(db_path: str) -> set[str]:
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT civitai_model_id FROM models").fetchall()
    return {str(row["civitai_model_id"]) for row in rows}


def _normalize_allow_commercial_use(value):
    if not value:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("{") and s.endswith("}"):
            s = s[1:-1]
        if not s or s.lower() in ("none", "null"):
            return []
        return [part.strip() for part in s.split(",") if part.strip()]
    return []


def _extract_thumb_info(model: dict, include_video: bool = True) -> tuple[str, str]:
    if not model.get("modelVersions"):
        return "", ""
    for model_version in model["modelVersions"]:
        images = model_version.get("images", [])
        if not images:
            continue
        for img in images[:5]:
            url = img.get("url", "")
            media_type = (img.get("type", "") or "").lower()
            if not include_video and media_type == "video":
                continue
            if url:
                return url, img.get("type", "")
    return "", ""


def _model_to_row(model: dict, include_video: bool = True) -> dict:
    lic = model if "allowNoCredit" in model else model.get("modelVersions", [{}])[0].get("license", {})
    comm = _normalize_allow_commercial_use(lic.get("allowCommercialUse"))
    thumb_url, thumb_type = _extract_thumb_info(model, include_video=include_video)
    now = datetime.now(timezone.utc).isoformat()
    base_model = ""
    if model.get("modelVersions"):
        base_model = model["modelVersions"][0].get("baseModel", "") or ""

    return {
        "civitai_model_id": str(model.get("id", "")),
        "name": model.get("name", ""),
        "model_type": model.get("type", ""),
        "base_model": base_model,
        "model_url": f"https://civitai.com/models/{model.get('id', '')}",
        "thumbnail_url": thumb_url,
        "thumbnail_local_path": "",
        "thumbnail_media_type": thumb_type,
        "allow_no_credit": 1 if lic.get("allowNoCredit", False) else 0,
        "allow_sell_images": 1 if "Image" in comm else 0,
        "allow_run_paid_service": 1 if "Rent" in comm else 0,
        "allow_share_merges": 1 if lic.get("allowDerivatives", False) else 0,
        "allow_sell_model": 1 if "Sell" in comm else 0,
        "allow_different_licenses": 1 if lic.get("allowDifferentLicenses", lic.get("allowDifferentLicense", False)) else 0,
        "fetched_at": now,
        "created_at": now,
        "updated_at": now,
    }


def model_to_row(model: dict, include_video: bool = True) -> dict:
    return dict(_model_to_row(model, include_video=include_video))


def save_models(db_path: str, models: list[dict], include_video: bool = True) -> int:
    if not models:
        return 0

    init_db(db_path)
    rows = [_model_to_row(model, include_video=include_video) for model in models]
    with _connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO models (
                civitai_model_id,
                name,
                model_type,
                base_model,
                model_url,
                thumbnail_url,
                thumbnail_local_path,
                thumbnail_media_type,
                allow_no_credit,
                allow_sell_images,
                allow_run_paid_service,
                allow_share_merges,
                allow_sell_model,
                allow_different_licenses,
                fetched_at,
                created_at,
                updated_at
            ) VALUES (
                :civitai_model_id,
                :name,
                :model_type,
                :base_model,
                :model_url,
                :thumbnail_url,
                :thumbnail_local_path,
                :thumbnail_media_type,
                :allow_no_credit,
                :allow_sell_images,
                :allow_run_paid_service,
                :allow_share_merges,
                :allow_sell_model,
                :allow_different_licenses,
                :fetched_at,
                :created_at,
                :updated_at
            )
            ON CONFLICT(civitai_model_id) DO UPDATE SET
                name=excluded.name,
                model_type=excluded.model_type,
                base_model=excluded.base_model,
                model_url=excluded.model_url,
                thumbnail_url=excluded.thumbnail_url,
                thumbnail_local_path=CASE
                    WHEN excluded.thumbnail_local_path <> '' THEN excluded.thumbnail_local_path
                    ELSE models.thumbnail_local_path
                END,
                thumbnail_media_type=excluded.thumbnail_media_type,
                allow_no_credit=excluded.allow_no_credit,
                allow_sell_images=excluded.allow_sell_images,
                allow_run_paid_service=excluded.allow_run_paid_service,
                allow_share_merges=excluded.allow_share_merges,
                allow_sell_model=excluded.allow_sell_model,
                allow_different_licenses=excluded.allow_different_licenses,
                fetched_at=excluded.fetched_at,
                updated_at=excluded.updated_at
            """
            ,
            rows,
        )
    return len(rows)


def find_db_files(base_dir: str) -> list[str]:
    if not os.path.exists(base_dir):
        return []
    db_files = []
    for root, _, files in os.walk(base_dir):
        for name in files:
            if name.lower().endswith(".db"):
                db_files.append(os.path.join(root, name))
    return sorted(db_files)


def get_filter_options(db_path: str) -> dict:
    init_db(db_path)
    with _connect(db_path) as conn:
        types = [
            row["model_type"]
            for row in conn.execute(
                "SELECT DISTINCT model_type FROM models WHERE model_type <> '' ORDER BY model_type"
            ).fetchall()
        ]
        bases = [
            row["base_model"]
            for row in conn.execute(
                "SELECT DISTINCT base_model FROM models WHERE base_model <> '' ORDER BY base_model"
            ).fetchall()
        ]
    return {"model_types": types, "base_models": bases}


def _build_order_by_sql(sorts: list[tuple[str, str]] | None) -> str:
    order_terms = []
    for field, direction in (sorts or []):
        if field not in DB_SORT_FIELDS:
            continue
        normalized_direction = "DESC" if direction == "desc" else "ASC"
        if field.startswith("allow_"):
            order_terms.append(f"CASE WHEN {field} THEN 0 ELSE 1 END {normalized_direction}")
        else:
            order_terms.append(f"COALESCE({field}, '') COLLATE NOCASE {normalized_direction}")
    order_terms.extend(["updated_at DESC", "id DESC"])
    return ", ".join(order_terms)


def search_models(
    db_path: str,
    *,
    keyword: str = "",
    model_type: str = "",
    base_model: str = "",
    page: int = 1,
    per_page: int = 50,
    sorts: list[tuple[str, str]] | None = None,
) -> dict:
    init_db(db_path)
    clauses = []
    params = []

    if keyword:
        clauses.append("(name LIKE ? OR civitai_model_id LIKE ?)")
        like = f"%{keyword}%"
        params.extend([like, like])
    if model_type:
        clauses.append("model_type = ?")
        params.append(model_type)
    if base_model:
        clauses.append("base_model = ?")
        params.append(base_model)

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    order_by_sql = _build_order_by_sql(sorts)
    offset = max(page - 1, 0) * per_page
    with _connect(db_path) as conn:
        total = conn.execute(
            f"SELECT COUNT(*) AS cnt FROM models {where_sql}",
            params,
        ).fetchone()["cnt"]
        rows = conn.execute(
            f"""
            SELECT *
            FROM models
            {where_sql}
            ORDER BY {order_by_sql}
            LIMIT ? OFFSET ?
            """,
            [*params, per_page, offset],
        ).fetchall()

    return {
        "items": [dict(row) for row in rows],
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": max((total + per_page - 1) // per_page, 1),
    }


def get_model_by_row_id(db_path: str, row_id: int) -> dict | None:
    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM models WHERE id = ?",
            (row_id,),
        ).fetchone()
    return dict(row) if row else None


def update_model_media_path(db_path: str, civitai_model_id: str, local_path: str) -> None:
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE models
            SET thumbnail_local_path = ?, updated_at = ?
            WHERE civitai_model_id = ?
            """,
            (local_path, datetime.now(timezone.utc).isoformat(), str(civitai_model_id)),
        )


def get_media_maintenance_params(db_path: str) -> dict[str, dict[str, str]]:
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT civitai_model_id, image_maintenance_params, video_maintenance_params
            FROM models
            """
        ).fetchall()
    return {
        str(row["civitai_model_id"]): {
            "image": str(row["image_maintenance_params"] or ""),
            "video": str(row["video_maintenance_params"] or ""),
        }
        for row in rows
    }


def update_model_maintenance_params(db_path: str, civitai_model_id: str, *, image_params: str | None = None, video_params: str | None = None) -> None:
    init_db(db_path)
    updates = []
    params: list[str] = []
    if image_params is not None:
        updates.append("image_maintenance_params = ?")
        params.append(str(image_params))
    if video_params is not None:
        updates.append("video_maintenance_params = ?")
        params.append(str(video_params))
    if not updates:
        return
    updates.append("updated_at = ?")
    params.append(datetime.now(timezone.utc).isoformat())
    params.append(str(civitai_model_id))
    with _connect(db_path) as conn:
        conn.execute(
            f"""
            UPDATE models
            SET {", ".join(updates)}
            WHERE civitai_model_id = ?
            """,
            params,
        )
