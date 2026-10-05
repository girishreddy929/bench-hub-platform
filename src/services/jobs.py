"""Background jobs: AI summaries on one slow worker, search indexing on its own fast worker.
No Streamlit here."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from src.ai import ai_engine as ai
from src.ai import models
from src.db import database as db
from src.search import indexer
from src.services import importer, timing

_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bench-jobs")        # summaries (slow)
_embed_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bench-embed")  # search index (seconds)
_pending: dict = {}   # (kind, employee_id) -> queued + running count
_errors: dict = {}    # (kind, employee_id) -> last failure message
_force: set = set()   # employee ids whose next summary run must ignore the input-hash cache
_batch = {"start": None, "src": "auto", "ran": 0, "skipped": 0, "failed": 0}  # for the BATCH log line
_read_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bench-read")  # folder-import reading
_reads: dict = {}  # uid -> {"state": queued|running|done|failed, "draft": Draft or None, "error": str}


def _who(employee_id: int) -> str:
    """Employee ID and name for the timing log (only looked up when the log is on)."""
    if not timing.enabled():
        return str(employee_id)
    try:
        emp = db.get_employee(employee_id)
    except Exception:
        emp = None
    return f"{emp.employee_code} ({emp.name})" if emp else str(employee_id)


def _do_summary(employee_id: int) -> str:
    """Returns 'ran' or 'skipped' (files unchanged)."""
    try:
        with _lock:
            force = employee_id in _force
            _force.discard(employee_id)

        with timing.span("read documents (db)"):
            docs = db.get_documents(employee_id)
        resume = docs["resume"].extracted_text if "resume" in docs else ""
        slider = docs["one_slider"].extracted_text if "one_slider" in docs else ""
        timing.log(
            f"  input: resume {len(resume or '')} chars, one-slider {len(slider or '')} chars"
            + (", forced" if force else "")
        )

        # 1. Certifications are read by code: instant, exact wording, HR-entered ones are kept.
        with timing.span("certifications (code)"):
            db.save_ai_certifications(employee_id, ai.extract_certifications(resume, slider))

        # 2. Skip the AI when the files (and model) are unchanged since the last good run.
        with timing.span("hash check"):
            new_hash = ai.input_hash(resume, slider, models.get_summary_model())
            stored = db.get_summary(employee_id)
        if not force and stored and stored["organization_summary"] and stored["input_hash"] == new_hash:
            timing.set_result("skipped (same files)")
            return "skipped"

        # 3. Two small AI calls. Whatever succeeds is saved; a failure in one keeps the other.
        problems, summary_ok = [], False
        try:
            part = ai.generate_summary_part(resume, slider)
            with timing.span("save summary"):
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
            with timing.span("save projects"):
                db.save_projects(employee_id, projects, input_hash=new_hash if summary_ok else None)
        except ai.SummaryError as exc:
            problems.append(f"projects: {exc}")
        if problems:
            raise ai.SummaryError("; ".join(problems))
        return "ran"
    finally:
        queue("embed", employee_id, src=timing.current_src())  # vectors follow the summary, even if it failed


def _do_embed(employee_id: int) -> None:
    indexer.index_employee(employee_id)


TASKS = {"summary": _do_summary, "embed": _do_embed}


def _run(kind: str, employee_id: int, src: str = "auto", queued_at=None) -> None:
    key = (kind, employee_id)
    error, result = None, None
    waited = (time.monotonic() - queued_at) if queued_at else 0.0
    started = time.perf_counter()
    try:
        if kind == "summary":
            with timing.profile(
                _who(employee_id), src, model=models.get_summary_model(), queue_wait=waited
            ):
                result = TASKS[kind](employee_id)
        else:
            result = TASKS[kind](employee_id)
    except Exception as exc:  # SummaryError or anything unexpected
        error = str(exc) or exc.__class__.__name__

    if kind == "embed" and timing.enabled():
        timing.log(
            f"EMBED emp={_who(employee_id)} src={src} model={models.get_embed_model()} "
            f"wait={waited:.1f}s run={time.perf_counter() - started:.1f}s"
            + (f" FAILED: {error}" if error else "")
        )

    batch_line = None
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
        if kind == "summary":
            _batch["failed" if error else ("skipped" if result == "skipped" else "ran")] += 1
            if not any(k[0] == "summary" for k in _pending):  # the summary queue is now empty
                count = _batch["ran"] + _batch["skipped"] + _batch["failed"]
                if count >= 2 and timing.enabled():
                    wall = time.monotonic() - (_batch["start"] or time.monotonic())
                    batch_line = (
                        f"BATCH src={_batch['src']}  {count} profiles: {_batch['ran']} ran, "
                        f"{_batch['skipped']} skipped, {_batch['failed']} failed, "
                        f"wall {timing.fmt_duration(wall)}"
                    )
                _batch.update(start=None, ran=0, skipped=0, failed=0)
    if batch_line:
        timing.log(batch_line)


def queue(kind: str, employee_id: int, src: str = "auto") -> None:
    """Start a task in the background and return immediately. `src` only labels the timing log."""
    key = (kind, employee_id)
    with _lock:
        if _pending.get(key, 0) >= 2:  # one running + one queued already covers new data
            return
        _pending[key] = _pending.get(key, 0) + 1
        _errors.pop(key, None)
        if kind == "summary" and _batch["start"] is None:
            _batch.update(start=time.monotonic(), src=src, ran=0, skipped=0, failed=0)
    (_embed_pool if kind == "embed" else _pool).submit(_run, kind, employee_id, src, time.monotonic())


def queue_summary(employee_id: int, force: bool = False, src: str = "auto") -> None:
    """force=True ignores the input-hash cache (used by the Regenerate button)."""
    if force:
        with _lock:
            _force.add(employee_id)
    queue("summary", employee_id, src)


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
        queue("embed", employee_id, src="stale_check")
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

# ---------------------- folder-import reading (background) ----------------------
def _read_job(uid: str, slot: dict) -> None:
    with _lock:
        _reads[uid]["state"] = "running"
    try:
        draft = importer.build_draft(uid, slot)  # AI problems become notes on the draft
        state, error = "done", ""
    except Exception as exc:  # unexpected file or database problem
        draft, state, error = None, "failed", str(exc) or exc.__class__.__name__
    with _lock:
        _reads[uid].update(state=state, draft=draft, error=error)


def queue_read(uid: str, slot: dict) -> bool:
    """Read one scanned profile with the AI in the background.
    Returns False if it is already queued, running, or waiting for review (no duplicates)."""
    with _lock:
        if uid in _reads:
            return False
        _reads[uid] = {"state": "queued", "draft": None, "error": ""}
    _read_pool.submit(_read_job, uid, slot)
    return True


def read_counts() -> dict:
    with _lock:
        counts = {"queued": 0, "running": 0, "done": 0, "failed": 0}
        for item in _reads.values():
            counts[item["state"]] += 1
        return counts


def take_reads() -> tuple:
    """Returns (finished drafts, {uid: error} for failed ones) and removes them from the store."""
    drafts, failed = [], {}
    with _lock:
        for uid in list(_reads):
            item = _reads[uid]
            if item["state"] == "done":
                drafts.append(item["draft"])
                del _reads[uid]
            elif item["state"] == "failed":
                failed[uid] = item["error"]
                del _reads[uid]
    return drafts, failed