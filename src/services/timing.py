"""Optional timing log for the whole pipeline. No Streamlit here.

.env:  TIMING_LOG=1        turn the log on (default: off)
       TIMING_LOG_FILE=    optional file too, e.g. logs/timing.log
"""

import logging
import os
import sys
import threading
import time
from contextlib import contextmanager

from dotenv import load_dotenv

load_dotenv()

ENABLED = os.getenv("TIMING_LOG", "0").strip().lower() in ("1", "true", "yes", "on")
LOG_FILE = os.getenv("TIMING_LOG_FILE", "").strip()

_logger = logging.getLogger("bench.timing")
_logger.propagate = False  # never reaches Streamlit's root logger (no double lines)
_logger.setLevel(logging.INFO)
if ENABLED and not _logger.handlers:  # this guard survives Streamlit reloading the module
    _console = logging.StreamHandler(sys.stdout)
    _console.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_console)
    if LOG_FILE:
        os.makedirs(os.path.dirname(LOG_FILE) or ".", exist_ok=True)
        _file = logging.FileHandler(LOG_FILE, encoding="utf-8")
        _file.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d"))
        _logger.addHandler(_file)

_local = threading.local()  # the profile being timed on this thread


def enabled() -> bool:
    return ENABLED


def log(text: str) -> None:
    if ENABLED:
        _logger.info(f"[time] {time.strftime('%H:%M:%S')} {text}")


def fmt_duration(seconds: float) -> str:
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes}m {secs:02d}s" if minutes else f"{secs}s"


class _Profile:
    def __init__(self, emp, src: str):
        self.emp, self.src = emp, src
        self.start = time.perf_counter()
        self.read = self.write = 0.0  # AI seconds, from Ollama's own numbers
        self.calls = 0
        self.result = "ok"


def current():
    return getattr(_local, "profile", None)


def current_src(default: str = "auto") -> str:
    prof = current()
    return prof.src if prof else default


def set_result(text: str) -> None:
    """Shown at the end of the DONE line, e.g. 'skipped (same files)'."""
    prof = current()
    if prof:
        prof.result = text


@contextmanager
def profile(emp, src: str, model: str = "", queue_wait=None, note: str = ""):
    """START line, then DONE line with the total. One per employee and action."""
    if not ENABLED:
        yield None
        return
    prof = _Profile(emp, src)
    previous = current()
    _local.profile = prof
    head = f"START emp={emp} src={src}"
    if model:
        head += f" model={model}"
    if queue_wait is not None:
        head += f" queue_wait={queue_wait:.1f}s"
    log(head + (f" {note}" if note else ""))
    try:
        yield prof
    except Exception as exc:
        prof.result = "FAILED: " + (str(exc) or exc.__class__.__name__)[:150]
        raise
    finally:
        _local.profile = previous
        total = time.perf_counter() - prof.start
        log(
            f"DONE  emp={emp} src={src} total={total:.1f}s "
            f"(AI read {prof.read:.1f}s, AI write {prof.write:.1f}s, {prof.calls} calls) {prof.result}"
        )


@contextmanager
def span(name: str):
    """Times one step. Indented inside a profile, a plain line outside one."""
    if not ENABLED:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        secs = time.perf_counter() - start
        log(f"{'  ' if current() else ''}{name:<30}{secs:7.2f}s")


def ai_call(label: str, response, attempt: int = 1, num_ctx: int = 0) -> None:
    """One line per model call, taken from Ollama's own timing fields."""
    if not ENABLED:
        return

    def num(name: str) -> float:
        return float(getattr(response, name, 0) or 0)

    load, read, write = (num(n) / 1e9 for n in ("load_duration", "prompt_eval_duration", "eval_duration"))
    read_tok, write_tok = int(num("prompt_eval_count")), int(num("eval_count"))
    rate = write_tok / write if write else 0.0
    prof = current()
    if prof:
        prof.read += read
        prof.write += write
        prof.calls += 1
    line = (
        f"  AI {label:<9} load {load:.1f}s | read {read:.1f}s ({read_tok} tok) | "
        f"write {write:.1f}s ({write_tok} tok, {rate:.1f} tok/s)"
    )
    if attempt > 1:
        line += f" | attempt {attempt}"
    if getattr(response, "done_reason", "") == "length":
        line += " | HIT OUTPUT CAP (answer cut)"
    if num_ctx and read_tok + write_tok >= num_ctx - 50:
        line += " | CONTEXT NEARLY FULL"
    log(line)