"""Folder import: group files by Employee ID, read them, build draft profiles.

File naming it understands (the Employee ID is the longest number in the name):
    1226_girish reddy resume.pdf
    girish one slider 1226.pptx
"""

import os
import re
from dataclasses import dataclass, field
from typing import Optional

import src.ai.ai_engine as ai
import src.db.database as db

SUPPORTED = {".pdf", ".docx", ".pptx"}
LEGACY = {".ppt", ".doc"}
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_NOISE = re.compile(r"one\s*slider|slider|resume|\bcv\b", re.I)


@dataclass
class Draft:
    uid: str  # stable row key (the Employee ID, or "noid:<name>")
    code: str = ""  # Employee ID; may be empty until the user types it
    name: str = ""
    email: str = ""
    skills: str = ""  # comma-separated, editable
    status: str = "Available"
    resume_path: Optional[str] = None
    slider_path: Optional[str] = None
    resume_text: str = ""
    slider_text: str = ""
    summary: Optional[dict] = None
    designation: str = ""  # job title, editable
    certs: str = ""  # one "Name | Issuer | Year" per line, editable
    projects: list = field(default_factory=list)  # from the AI, saved as is
    notes: list = field(default_factory=list)
    exists: bool = False  # Employee ID already in the database
    error: str = ""


def parse_employee_code(file_name: str) -> Optional[str]:
    numbers = re.findall(r"\d+", os.path.splitext(file_name)[0])
    return max(numbers, key=len) if numbers else None


def classify(file_name: str) -> str:
    stem, ext = os.path.splitext(file_name.lower())
    if "slider" in stem:
        return "one_slider"
    if re.search(r"(^|[\s_\-])(resume|cv)($|[\s_\-])", stem):
        return "resume"
    return "one_slider" if ext == ".pptx" else "resume"


def guess_name(file_name: str) -> str:
    stem = os.path.splitext(os.path.basename(file_name))[0]
    stem = re.sub(r"[_\-.]+", " ", stem)
    stem = re.sub(r"\d+", " ", stem)
    stem = _NOISE.sub(" ", stem)
    return " ".join(stem.split()).title()


def display_name(slot: dict) -> str:
    names = [guess_name(p) for p in (slot["resume"], slot["one_slider"]) if p]
    return max(names, key=len) if names else ""


def find_email(text: str) -> str:
    match = EMAIL_RE.search(text or "")
    return match.group(0).lower() if match else ""


def scan_folder(folder: str):
    """Return ({key: {'resume', 'one_slider', 'code'}}, [messages]).

    Files with an Employee ID are grouped by it. Files without one are grouped
    by the name in the file name, and need an ID typed in before upload.
    """
    if not folder or not os.path.isdir(folder):
        raise ValueError(f"Folder not found: {folder!r}")
    groups: dict = {}
    skipped: list[str] = []
    for entry in sorted(os.listdir(folder)):
        path = os.path.join(folder, entry)
        ext = os.path.splitext(entry)[1].lower()
        if not os.path.isfile(path) or entry.startswith("~$"):
            continue
        if ext in LEGACY:
            skipped.append(f"{entry}: old {ext} format not supported (save as .docx/.pptx)")
            continue
        if ext not in SUPPORTED:
            continue
        code = parse_employee_code(entry) or ""
        key = code or "noid:" + (" ".join(guess_name(entry).lower().split()) or entry.lower())
        kind = classify(entry)
        slot = groups.setdefault(key, {"resume": None, "one_slider": None, "code": code})
        if slot[kind] is None:
            slot[kind] = path
        else:
            skipped.append(f"{entry}: extra {kind.replace('_', '-')} for the same person, ignored")
    return groups, skipped



def _read(path: str, notes: list) -> str:
    name = os.path.basename(path)
    try:
        with open(path, "rb") as fh:
            text = ai.extract_text(name, fh.read())
    except Exception as exc:
        notes.append(f"Could not read {name}: {exc}")
        return ""
    if not text:
        notes.append(f"No text found in {name} (scanned or image-only file?)")
    return text

def _certs_text(certs: list) -> str:
    """AI certification dicts -> the editable text form (Name | Issuer | Year per line)."""
    lines = []
    for c in certs or []:
        parts = [c.get("name", ""), c.get("issuer", ""), c.get("year", "")]
        while parts and not parts[-1]:
            parts.pop()
        if parts and parts[0]:
            lines.append(" | ".join(parts))
    return "\n".join(lines)

def build_draft(uid: str, slot: dict) -> Draft:
    """Read the files, pull the email, and let the local AI fill summary + skills."""
    d = Draft(uid=uid, code=slot["code"], resume_path=slot["resume"], slider_path=slot["one_slider"])
    d.name = display_name(slot)
    known = db.get_employee_by_code(d.code) if d.code else None
    d.exists = known is not None
    if known is not None:
        d.designation = known.designation or ""  # HR's title wins over the AI's
    if d.resume_path:
        d.resume_text = _read(d.resume_path, d.notes)
    if d.slider_path:
        d.slider_text = _read(d.slider_path, d.notes)
    else:
        d.notes.append("No one-slider found: using the resume only")
    d.email = find_email(d.resume_text) or find_email(d.slider_text)
    try:
        d.summary = ai.generate_summary(d.resume_text, d.slider_text, [])
        d.skills = ", ".join(d.summary["skills"])
        d.designation = d.designation or d.summary.get("designation", "")
        d.certs = _certs_text(d.summary.get("certifications", []))
        d.projects = d.summary.get("projects", [])
    except ai.SummaryError as exc:
        d.notes.append(f"AI could not fill the profile: {exc}")
    return d


def problems(d: Draft) -> list[str]:
    """Blocking issues: a profile with any of these cannot be uploaded."""
    out = []
    if not d.code.strip():
        out.append("Employee ID is missing: click the edit and enter it")
    if not d.name.strip():
        out.append("Name is missing")
    if not EMAIL_RE.fullmatch(d.email.strip()):
        out.append("Email is missing or invalid (not found in the files, please edit)")
    return out


def upload_draft(d: Draft) -> tuple[int, bool]:
    """Save a reviewed draft. Returns (database id, needs_regen).

    needs_regen is True for an existing employee: their old feedback and files
    must be part of the new summary, so the caller regenerates it.
    Certifications HR has reviewed are saved as manual ones (the AI never overwrites them).
    """
    existing = db.get_employee_by_code(d.code)
    ai_title = (d.summary or {}).get("designation", "")
    typed = d.designation.strip()
    hr_title = None if (not typed or typed == ai_title) else typed  # the AI's title is not HR's
    emp_id = db.save_employee(
        d.code, d.name, d.email, d.skills, d.status,
        employee_id=existing.id if existing else None,
        designation=hr_title,  # None = keep what is stored
    )
    for kind, path, text in (
        ("resume", d.resume_path, d.resume_text),
        ("one_slider", d.slider_path, d.slider_text),
    ):
        if path:
            with open(path, "rb") as fh:
                db.save_document(emp_id, kind, os.path.basename(path), fh.read(), text)
    if d.certs.strip():  # an empty box never wipes existing certifications
        db.save_manual_certifications(emp_id, db.parse_certifications(d.certs))
    if existing is not None:
        return emp_id, True
    if d.summary:
        skills = [s.strip() for s in d.skills.split(",") if s.strip()]  # user-edited
        db.save_summary(
            emp_id, d.summary["organization_summary"], d.summary["top_strengths"], skills,
            projects=d.projects,
            ai_designation=ai_title,
            # the "files unchanged" fingerprint is stored only if both AI calls worked
            input_hash=ai.input_hash(d.resume_text, d.slider_text) if "projects" in d.summary else None,
        )
    return emp_id, False