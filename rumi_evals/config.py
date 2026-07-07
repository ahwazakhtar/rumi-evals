from __future__ import annotations

from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else PACKAGE_ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sql_text(query_name: str) -> str:
    return (PACKAGE_ROOT / "sql" / f"{query_name}.sql").read_text(encoding="utf-8")
