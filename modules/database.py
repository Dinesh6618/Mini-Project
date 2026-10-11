"""Optional local SQLite storage for prepared student data.

Concept (Principles of Data Science): *data storage and retrieval*. The
database is a single local file (``data/students.db``); nothing is sent to
any server. It is optional: the app works entirely from CSV files.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from modules.config import NAME_COLUMN

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "students.db"
TABLE = "students"


def _connect(path: Path):
    """Open a connection that is committed *and closed* on exit.

    ``with sqlite3.connect()`` alone only commits; on Windows an unclosed
    connection keeps the file locked so it cannot be deleted.
    """
    return closing(sqlite3.connect(path))


def database_exists(path: Path = DB_PATH) -> bool:
    if not path.exists():
        return False
    try:
        with _connect(path) as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (TABLE,)
            ).fetchone()
        return row is not None
    except sqlite3.Error:
        return False


def save_dataset(frame: pd.DataFrame, path: Path = DB_PATH, include_names: bool = False) -> int:
    """Replace the stored table with ``frame``. Returns the number of rows saved.

    Student names are dropped unless ``include_names`` is True.
    """
    data = frame.copy()
    if not include_names and NAME_COLUMN in data.columns:
        data = data.drop(columns=NAME_COLUMN)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as conn:
        data.to_sql(TABLE, conn, if_exists="replace", index=False)
        conn.commit()
    return len(data)


def load_dataset(path: Path = DB_PATH) -> pd.DataFrame:
    """Read the stored table back into a DataFrame."""
    if not database_exists(path):
        raise FileNotFoundError("No saved dataset was found. Save one from the Student Records page.")
    with _connect(path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {TABLE}", conn)


def delete_database(path: Path = DB_PATH) -> None:
    if path.exists():
        path.unlink()
