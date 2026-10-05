"""Delete ALL rows from ALL tables of bench.db (the tables stay, the file stays).
Stop Streamlit first.  Run:  python -m tests.db_clear   (--yes skips the question)"""

import os
import sqlite3
import sys

from dotenv import load_dotenv

load_dotenv()


def db_path() -> str:
    url = os.getenv("BENCH_DATABASE_URL", "sqlite:///bench.db")
    if not url.startswith("sqlite:///"):
        sys.exit(f"Only a SQLite file is supported here (BENCH_DATABASE_URL={url}).")
    return os.path.abspath(url[len("sqlite:///"):])


def table_names(con) -> list:
    return [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]


def show(title: str, path: str) -> None:
    con = sqlite3.connect(path)
    try:
        print(title)
        for name in table_names(con):
            try:
                n = con.execute('SELECT COUNT(*) FROM "%s"' % name).fetchone()[0]
            except sqlite3.Error:
                n = "?"
            print(f"   {name:<22}{n:>7}")
    finally:
        con.close()


def main() -> None:
    path = db_path()
    if not os.path.isfile(path):
        sys.exit(f"No database at {path}. Nothing to clear (the app creates it on start).")
    print(f"Database : {path}")
    show("Current contents:", path)
    print("ALL rows in ALL tables will be deleted. The tables stay.")
    print("Tip: run  python -m tests.db_backup  first.")
    if "--yes" not in sys.argv and input("Type YES to clear: ").strip() != "YES":
        sys.exit("Cancelled.")
    con = sqlite3.connect(path, timeout=5)
    try:
        con.execute("PRAGMA foreign_keys=OFF")
        for name in table_names(con):
            try:
                con.execute('DELETE FROM "%s"' % name)
            except sqlite3.Error as exc:
                print(f"   skipped {name}: {exc}")
        try:
            con.execute("DELETE FROM sqlite_sequence")
        except sqlite3.Error:
            pass  # only exists when a table uses AUTOINCREMENT
        con.commit()
        con.execute("VACUUM")
    except sqlite3.OperationalError as exc:
        sys.exit(f"Could not clear ({exc}). Stop Streamlit and try again.")
    finally:
        con.close()
    show("Cleared. Database now contains:", path)


if __name__ == "__main__":
    main()