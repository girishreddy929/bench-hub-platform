"""Background jobs: AI summaries on one slow worker, search indexing on its own fast worker.
No Streamlit here."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from src.ai import ai_engine as ai
from src.ai import models
from src.db import database as db
from src.search import indexer

_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bench-jobs")        # summaries (slow)
_embed_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bench-embed")  # search index (seconds)
_pending: dict = {}   # (kind, employee_id) -> queued + running count
_errors: dict = {}    # (kind, employee_id) -> last failure message
_force: set = set()   # employee ids whose next summary run must ignore the input-hash cache


def _do_summary(employee_id: int) -> None:
    try:
        with _lock:
            force = employee_id in _force
            _force.discard(employee_id)

        docs = db.get_documents(employee_id)
        resume = docs["resume"].extracted_text if "resume" in docs else ""
        slider = docs["one_slider"].extracted_text if "one_slider" in docs else ""

        # 1. Certifications are read by code: instant, exact wording, HR-entered ones are kept.
        db.save_ai_certifications(employee_id, ai.extract_certifications(resume, slider))

        # 2. Skip the AI when the files (and model) are unchanged since the last good run.
        new_hash = ai.input_hash(resume, slider, models.get_summary_model())
        stored = db.get_summary(employee_id)
        if not force and stored and stored["organization_summary"] and stored["input_hash"] == new_hash:
            return

        # 3. Two small AI calls. Whatever succeeds is saved; a failure in one keeps the other.
        problems, summary_ok = [], False
        try:
            part = ai.generate_summary_part(resume, slider)
            db.save_summary(
                employee_id, part["organization_summary"], [], part["skills"],
                ai_designation=part["designation"],
            )
            summary_ok = True
        except ai.SummaryError as exc:
            problems.append(f"summary: {exc}")
        try:
            projects = ai.generate_projects(resume, slider)
            # the hash is stored only when both calls worked, so a half-failed run is retried
            db.save_projects(employee_id, projects, input_hash=new_hash if summary_ok else None)
        except ai.SummaryError as exc:
            problems.append(f"projects: {exc}")
        if problems:
            raise ai.SummaryError("; ".join(problems))
    finally:
        queue("embed", employee_id)  # vectors follow the summary, even if it failed

def _do_embed(employee_id: int) -> None:
    indexer.index_employee(employee_id)


TASKS = {"summary": _do_summary, "embed": _do_embed}


def _run(kind: str, employee_id: int) -> None:
    key = (kind, employee_id)
    error = None
    try:
        TASKS[kind](employee_id)
    except Exception as exc:  # SummaryError or anything unexpected
        error = str(exc) or exc.__class__.__name__
    with _lock:
        left = _pending.get(key, 1) - 1
        if left <= 0:
            _pending.pop(key, None)
        else:
            _pending[key] = left
        if error:
            _errors[key] = error
        else:
            _errors.pop(key, None)


def queue(kind: str, employee_id: int) -> None:
    """Start a task in the background and return immediately."""
    key = (kind, employee_id)
    with _lock:
        if _pending.get(key, 0) >= 2:  # one running + one queued already covers new data
            return
        _pending[key] = _pending.get(key, 0) + 1
        _errors.pop(key, None)
    (_embed_pool if kind == "embed" else _pool).submit(_run, kind, employee_id)


def queue_summary(employee_id: int, force: bool = False) -> None:
    """force=True ignores the input-hash cache (used by the Regenerate button)."""
    if force:
        with _lock:
            _force.add(employee_id)
    queue("summary", employee_id)

_last_check = 0.0
CHECK_EVERY_SECONDS = 5


def queue_stale_embeddings() -> int:
    """Queue an embed job for every employee whose vectors (for the selected model) are
    missing or outdated. Skips employees already queued, or whose last embed attempt failed
    (a Regenerate click, a new review or a model switch clears that block)."""
    global _last_check
    now = time.monotonic()
    with _lock:
        if now - _last_check < CHECK_EVERY_SECONDS:
            return 0
        _last_check = now
    try:
        stale = indexer.stale_ids()
    except Exception:
        return 0
    count = 0
    for employee_id in stale:
        key = ("embed", employee_id)
        with _lock:
            blocked = key in _errors or _pending.get(key, 0) > 0
        if blocked:
            continue
        queue("embed", employee_id)
        count += 1
    return count

def clear_errors() -> None:
    """Forget earlier failures (called when the user switches a model)."""
    with _lock:
        _errors.clear()

def pending_ids() -> set:
    with _lock:
        return {emp_id for (_, emp_id) in _pending}


def error_for(employee_id: int):
    with _lock:
        for (kind, emp_id), message in _errors.items():
            if emp_id == employee_id:
                return f"{kind}: {message}"
    return None