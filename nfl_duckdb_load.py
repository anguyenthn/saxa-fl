#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 21:39:09 2025

@author: an
"""

# =========================
# How to open the DuckDB file I sent you
# =========================
# 1) Save the email attachment named `nfl.duckdb` to a folder you can find.
#    Example locations:
#       - macOS:   ~/Desktop/NFL/nfl.duckdb
#       - Windows: C:\Users\<you>\Desktop\NFL\nfl.duckdb
#
# 2) (One time) Install the DuckDB Python package:
#       pip install duckdb
#    Optional (for prettier table output): also install pandas
#       pip install pandas
#
# 3) Update DB_PATH below to where you saved the file.
#
# 4) Run this script (python) or paste into a Jupyter/VS Code cell.
#
# 5) You should see:
#       - a list of tables (look for "nfl")
#       - a small preview (first 5 rows)
#       - a simple row count
#
# Notes:
# - If you don’t install pandas, replace any “.df()” calls with “.fetchall()”.
# - The database is read directly from the .duckdb file; no “import” step needed.

from pathlib import Path
import duckdb

# --- UPDATE THIS PATH to wherever you saved the attachment ---
DB_PATH = Path.home() / "Desktop" / "NFL" / "nfl.duckdb"   # e.g., /Users/you/Desktop/NFL/nfl.duckdb
# Example Windows path if you prefer a literal string:
# DB_PATH = r"C:\Users\you\Desktop\NFL\nfl.duckdb"

# 6) Connect to the DuckDB file (it’s just a normal file on disk)
conn = duckdb.connect(str(DB_PATH))

# 7) Sanity check: list tables inside the database (you should see "nfl")
print("Tables in DB:")
print(conn.execute("PRAGMA show_tables;").df())  # use .fetchall() if you didn't install pandas

# 8) Quick preview of the data
print("\nPreview (first 5 rows):")
print(conn.execute("SELECT * FROM nfl LIMIT 5;").df())     # use .fetchall() if no pandas

# 9) Simple row count
print("\nRow count:")
print(conn.execute("SELECT COUNT(*) AS rows FROM nfl;").df())  # use .fetchall() if no pandas

# 10) Done — close the connection
conn.close()
