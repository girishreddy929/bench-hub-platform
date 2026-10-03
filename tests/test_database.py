import os

# Must be set BEFORE importing database.
os.environ["BENCH_DATABASE_URL"] = "sqlite:///test_bench.db"
for suffix in ("", "-wal", "-shm"):
    if os.path.exists("test_bench.db" + suffix):
        os.remove("test_bench.db" + suffix)

import time

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

import src.db.database as db

# Safety: never touch the real database.
assert "test_bench" in db.DATABASE_URL, f"Refusing to run against {db.DATABASE_URL}"


def expect_error(exc_type, func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except exc_type:
        return
    raise AssertionError(f"{func.__name__} should have raised {exc_type.__name__}")


db.init_db()

# Employees
eid = db.save_employee("T-1", "Test Person", "Test@Example.com", "Python, SQL", "Available")
assert db.get_employee(eid).email == "test@example.com"
assert db.get_employee_by_code("T-1").id == eid
expect_error(ValueError, db.save_employee, "T-2", "Other", "test@example.com")  # dup email
expect_error(ValueError, db.save_employee, "T-1", "Other", "other@example.com")  # dup code
db.save_employee("T-1", "Test Person", "test@example.com", "Python, Spark", "Interviewing", employee_id=eid)
assert db.get_employee(eid).availability_status == "Interviewing"
assert len(db.get_all_employees()) == 1

# Documents: upsert and separation of types
db.save_document(eid, "resume", "a.pdf", b"one", "text one")
db.save_document(eid, "resume", "b.pdf", b"two", "text two")
docs = db.get_documents(eid)
assert len(docs) == 1 and docs["resume"].file_data == b"two"
db.save_document(eid, "one_slider", "s.pptx", b"slide", "slide text")
docs = db.get_documents(eid)
assert len(docs) == 2 and docs["resume"].file_name == "b.pdf"
expect_error(ValueError, db.save_document, 9999, "resume", "x.pdf", b"x", "")

# Foreign keys are really enforced
with Session(db.engine) as s:
    s.add(db.Document(employee_id=9999, doc_type="resume", file_name="x", file_data=b"x"))
    expect_error(IntegrityError, s.commit)

# Feedback: history kept, newest first, rating validated
for i, mgr in enumerate(["M1", "M2", "M3"]):
    assert db.add_feedback(eid, mgr, "Proj A", 3 + (i % 3), pros="good", cons="late", comments="ok") is False
    time.sleep(0.01)
fb = db.get_feedback(eid)
assert len(fb) == 3 and fb[0].manager_name == "M3"
expect_error(ValueError, db.add_feedback, eid, "M4", "Proj A", 6)
expect_error(ValueError, db.add_feedback, eid, "M4", "", 3)  # project required

# Same manager + same project replaces; a different project is a new entry
assert db.add_feedback(eid, " m1 ", "proj a", 5, pros="updated") is True
assert len(db.get_feedback(eid)) == 3 and db.get_feedback(eid)[0].pros == "updated"
assert db.add_feedback(eid, "M1", "Proj B", 4) is False
assert len(db.get_feedback(eid)) == 4

# Summary
db.save_summary(eid, "first", ["a"], ["python"])
db.save_summary(eid, "second", ["a", "b"], ["python", "sql"])
summary = db.get_summary(eid)
assert summary["organization_summary"] == "second"
assert summary["top_strengths"] == ["a", "b"]
assert summary["skills"] == ["python", "sql"]

print("All database checks passed.")
db.engine.dispose()
for suffix in ("", "-wal", "-shm"):
    if os.path.exists("test_bench.db" + suffix):
        os.remove("test_bench.db" + suffix)