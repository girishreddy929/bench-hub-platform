"""Text extraction + local LLM summary (Ollama). No cloud, no API keys."""

import io
import json
import os

import httpx
import ollama
import pdfplumber
from docx import Document as DocxDocument
from dotenv import load_dotenv
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from src.search.skills import merge_skills
from src.ai import models

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
OLLAMA_MAX_OUTPUT = int(os.getenv("OLLAMA_MAX_OUTPUT", "600"))
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "180"))

MAX_FEEDBACK_CHARS = 3000
MAX_SLIDER_CHARS = 3500
MAX_RESUME_CHARS = 3500

# OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
# OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "180"))

# # Keep the prompt inside the context window (roughly 4 chars per token).
# MAX_FEEDBACK_CHARS = 5000
# MAX_SLIDER_CHARS = 5000
# MAX_RESUME_CHARS = 6000


class SummaryError(Exception):
    """Raised when a summary cannot be produced (message is user-friendly)."""


# --------------------------------------------------------------------------
# Text extraction
# --------------------------------------------------------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = DocxDocument(io.BytesIO(file_bytes))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts).strip()


def _shape_texts(shape):
    """Yield text from a shape, recursing into groups and reading tables."""
    if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
        for child in shape.shapes:
            yield from _shape_texts(child)
        return
    if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
        text = shape.text_frame.text.strip()
        if text:
            yield text
    if getattr(shape, "has_table", False) and shape.has_table:
        for row in shape.table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                yield " | ".join(cells)


def extract_text_from_pptx(file_bytes: bytes) -> str:
    presentation = Presentation(io.BytesIO(file_bytes))
    parts = []
    for slide in presentation.slides:
        for shape in slide.shapes:
            parts.extend(_shape_texts(shape))
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                parts.append(f"Notes: {notes}")
    return "\n".join(parts).strip()


def extract_text(file_name: str, file_bytes: bytes) -> str:
    ext = os.path.splitext(file_name)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    if ext == ".docx":
        return extract_text_from_docx(file_bytes)
    if ext == ".pptx":
        return extract_text_from_pptx(file_bytes)
    if ext in (".ppt", ".doc"):
        raise ValueError(f"Old '{ext}' files are not supported. Save as .pptx/.docx.")
    raise ValueError(f"Unsupported file type '{ext}'. Use PDF, DOCX or PPTX.")


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an HR assistant writing profiles of bench employees for project "
    "managers. Use ONLY the information provided. Never invent skills, "
    "employers, projects, dates, or facts. If information is missing, say so. "
    "Source priority when sources overlap or disagree: 1) MANAGER FEEDBACK "
    "(most trusted), 2) ORGANIZATION ONE-SLIDER, 3) RESUME (least weight). "
    "Respond with JSON only, using exactly these keys: "
    "organization_summary, top_strengths, skills."
)

USER_TEMPLATE = """Write (be concise):
1. organization_summary: 3-4 sentences, as a string, on what this person has done INSIDE OUR ORGANIZATION (roles, projects, contributions, how managers see them). Use the one-slider first, confirmed or adjusted by manager feedback. Use the resume only for one short sentence of background. If there is no one-slider, say so and rely on manager feedback.
2. top_strengths: 3-5 bullets, at most 12 words each, as a list of strings. Prefer strengths backed by manager pros/comments, then the one-slider, then the resume.
3. skills: up to 20 short skill names (tools, technologies, domains), 1-4 words each, from ALL three sources. No sentences, no duplicates.

Do not invent anything not present in the text.

### MANAGER FEEDBACK (highest priority)
{feedback}

### ORGANIZATION ONE-SLIDER (second priority)
{one_slider}

### RESUME (career background, lowest priority)
{resume}
"""

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "organization_summary": {"type": "string"},
        "top_strengths": {"type": "array", "items": {"type": "string"}},
        "skills": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["organization_summary", "top_strengths", "skills"],
}


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n[...truncated]"


def _field(item, name: str) -> str:
    value = item.get(name, "") if isinstance(item, dict) else getattr(item, name, "")
    return "" if value is None else str(value)


def _format_feedback(feedback_list) -> str:
    """Newest-first list in, trimmed to the character budget."""
    blocks, used = [], 0
    for item in feedback_list:
        block = (
            f"- Manager: {_field(item, 'manager_name')} | "
            f"Project: {_field(item, 'project_name') or 'n/a'} | "
            f"Rating: {_field(item, 'rating')}/5\n"
            f"  Pros: {_field(item, 'pros') or 'None'}\n"
            f"  Cons: {_field(item, 'cons') or 'None'}\n"
            f"  Comments: {_field(item, 'comments') or 'None'}"
        )
        if used + len(block) > MAX_FEEDBACK_CHARS and blocks:
            break
        blocks.append(block)
        used += len(block)
    return "\n".join(blocks)


# --------------------------------------------------------------------------
# Summary generation
# --------------------------------------------------------------------------
def _chat(client: ollama.Client, user_prompt: str, model: str) -> str:
    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            format=SUMMARY_SCHEMA,
            keep_alive=OLLAMA_KEEP_ALIVE,
            options={"temperature": 0.2, "num_ctx": OLLAMA_NUM_CTX, "num_predict": OLLAMA_MAX_OUTPUT},
        )
    except ollama.ResponseError as exc:
        if exc.status_code == 404:
            raise SummaryError(
                f"Model '{model}' is not installed. "
                f"Run: ollama pull {model}"
            )
        raise SummaryError(f"Ollama error: {exc.error}")
    except (ConnectionError, httpx.ConnectError):
        raise SummaryError(
            "Cannot reach Ollama. Make sure it is running (try: ollama serve)."
        )
    except httpx.TimeoutException:
        raise SummaryError(
            "The model took too long to respond. Try again, or raise OLLAMA_TIMEOUT."
        )
    return response["message"]["content"]


def _as_list(value) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def _normalize(data: dict) -> dict:
    overview = str(data.get("organization_summary", "")).strip()
    if not overview:
        raise ValueError("empty organization summary")
    return {
        "organization_summary": overview,
        "top_strengths": _as_list(data.get("top_strengths"))[:5],
        "skills": merge_skills(_as_list(data.get("skills")))[:25],
    }


def generate_summary(resume_text: str, one_slider_text: str, feedback_list, model=None) -> dict:
    """Return {'organization_summary': str, 'top_strengths': [...], 'skills': [...]}.

    feedback_list: Feedback objects (or dicts), newest first.
    Raises SummaryError with a user-friendly message on any failure.
    """
    resume_text = (resume_text or "").strip()
    one_slider_text = (one_slider_text or "").strip()
    feedback_text = _format_feedback(feedback_list or [])

    if not (resume_text or one_slider_text or feedback_text):
        raise SummaryError(
            "Nothing to summarize yet. Upload a resume/one-slider or add feedback."
        )

    user_prompt = USER_TEMPLATE.format(
        feedback=feedback_text or "None",
        one_slider=_truncate(one_slider_text, MAX_SLIDER_CHARS) or "Not provided",
        resume=_truncate(resume_text, MAX_RESUME_CHARS) or "Not provided",
    )

    model = model or models.get_summary_model()  # current dropdown choice unless overridden
    client = ollama.Client(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT)
    for _attempt in range(2):  # one retry if the JSON is unusable
        content = _chat(client, user_prompt, model)
        try:
            return _normalize(json.loads(content))
        except (json.JSONDecodeError, ValueError, AttributeError):
            continue
    raise SummaryError("The model returned an unusable answer twice. Try again.")


def check_ollama() -> tuple[bool, str]:
    """Quick health check for the UI sidebar (uses the currently selected summary model)."""
    installed = models.installed_models()
    if installed is None:
        return False, "Ollama is not reachable. Start it (ollama serve)."
    model = models.get_summary_model()
    if models.is_installed(model, installed):
        return True, f"Ollama ready ({model})"
    return False, f"Model not installed. Run: ollama pull {model}"

