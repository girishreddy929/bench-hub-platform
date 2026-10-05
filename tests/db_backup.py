"""Copy bench.db to bench_db_backup.db (an existing backup is replaced).
Stop Streamlit first.  Run:  python -m tests.db_backup   (--yes skips the question)"""

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
    source = db_path()
    target = os.path.join(os.path.dirname(source), "bench_db_backup.db")
    if not os.path.isfile(source):
        sys.exit(f"No database found at {source}")
    print(f"Database : {source}\nBackup   : {target}")
    show("Database contains:", source)
    if os.path.isfile(target):
        show("Existing backup (will be REPLACED):", target)
        if "--yes" not in sys.argv and input("Type YES to replace it: ").strip() != "YES":
            sys.exit("Cancelled.")
    temp = target + ".tmp"  # written first, so a failure never destroys the old backup
    if os.path.exists(temp):
        os.remove(temp)
    src = sqlite3.connect(source)
    dst = sqlite3.connect(temp)
    try:
        src.backup(dst)
        dst.execute("PRAGMA journal_mode=DELETE")  # one standalone file, no -wal/-shm
    finally:
        dst.close()
        src.close()
    os.replace(temp, target)
    show("Backup done. Backup contains:", target)


if __name__ == "__main__":
    main()