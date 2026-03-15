from __future__ import annotations

import sqlite3
from typing import List, Tuple

import pandas as pd

def connect_db(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(db_path)

def get_tables(conn: sqlite3.Connection) -> List[str]:
    query = '''
    SELECT name
    FROM sqlite_master
    WHERE type='table' AND name NOT LIKE 'sqlite_%';
    '''
    return [row[0] for row in conn.execute(query).fetchall()]

def get_table_info(conn: sqlite3.Connection, table: str) -> List[Tuple]:
    return conn.execute(f"PRAGMA table_info({table});").fetchall()

def schema_text(conn: sqlite3.Connection) -> str:
    tables = get_tables(conn)
    lines = []
    for table in tables:
        cols = get_table_info(conn, table)
        col_str = ", ".join(f"{c[1]} ({c[2]})" for c in cols)
        lines.append(f"- {table}: {col_str}")
    return "\n".join(lines)

def load_product_tables(conn: sqlite3.Connection) -> tuple[pd.DataFrame, pd.DataFrame]:
    products_df = pd.read_sql_query("SELECT * FROM products;", conn)
    keywords_df = pd.read_sql_query("SELECT * FROM product_keywords;", conn)
    return products_df, keywords_df
