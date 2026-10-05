"""Replace bench.db with bench_db_backup.db.
Stop Streamlit first.  Run:  python -m tests.db_restore   (--yes skips the question)"""

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


def show(title: str, path: str) -> None:
    con = sqlite3.connect(path)
    try:
        names = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        print(title)
        for name in names:
            try:
                n = con.execute('SELECT COUNT(*) FROM "%s"' % name).fetchone()[0]
            except sqlite3.Error:
                n = "?"
            print(f"   {name:<22}{n:>7}")
    finally:
        con.close()


def main() -> None:
    target = db_path()
    backup = os.path.join(os.path.dirname(target), "bench_db_backup.db")
    if not os.path.isfile(backup):
        sys.exit(f"No backup found at {backup}")
    print(f"Backup   : {backup}\nDatabase : {target}")
    show("Backup contains:", backup)
    if os.path.isfile(target):
        show("Current database (will be REPLACED):", target)
    if "--yes" not in sys.argv and input("Type YES to restore: ").strip() != "YES":
        sys.exit("Cancelled.")
    src = sqlite3.connect(backup)
    dst = sqlite3.connect(target, timeout=5)
    try:
        dst.execute("PRAGMA journal_mode=DELETE")
        src.backup(dst)
    except sqlite3.OperationalError as exc:
        sys.exit(f"Could not restore ({exc}). Stop Streamlit and try again.")
    finally:
        dst.close()
        src.close()
    show("Restored. Database now contains:", target)


if __name__ == "__main__":
    main()