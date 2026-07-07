"""Data access layer.

Two backends:
- "bigquery": live reads via google-cloud-bigquery using Application Default
  Credentials (`gcloud auth application-default login`).
- "csv": reads extracts from <extracts_dir>/<query_name>.csv. Extracts can be
  produced by running the .sql files through the taleemabad-data plugin
  (save_query_results) or the BigQuery console. This is the default so the
  pipeline runs without cloud credentials.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PACKAGE_ROOT, sql_text


class DataBackend:
    def fetch(self, query_name: str) -> pd.DataFrame:
        raise NotImplementedError


class CsvBackend(DataBackend):
    def __init__(self, extracts_dir: str | Path):
        self.extracts_dir = (PACKAGE_ROOT / extracts_dir) if not Path(extracts_dir).is_absolute() else Path(extracts_dir)

    def fetch(self, query_name: str) -> pd.DataFrame:
        path = self.extracts_dir / f"{query_name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"No extract at {path}. Run sql/{query_name}.sql against BigQuery "
                f"and save the result there, or switch data.backend to 'bigquery'."
            )
        return pd.read_csv(path)


class BigQueryBackend(DataBackend):
    def __init__(self, project: str | None):
        from google.cloud import bigquery  # lazy import

        self.client = bigquery.Client(project=project)

    def fetch(self, query_name: str) -> pd.DataFrame:
        return self.client.query(sql_text(query_name)).to_dataframe()


def get_backend(cfg: dict) -> DataBackend:
    data_cfg = cfg["data"]
    if data_cfg["backend"] == "bigquery":
        return BigQueryBackend(data_cfg.get("bigquery_project"))
    return CsvBackend(data_cfg.get("extracts_dir", "data/extracts"))
