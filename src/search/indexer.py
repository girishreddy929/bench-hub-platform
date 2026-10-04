"""Builds evidence chunks and the skill vocabulary, and embeds whatever changed,
for the currently selected embedding model. No Streamlit here."""

from src.ai import embeddings as emb
from src.ai import models
from src.db import database as db
from src.search.skills import merge_skills, normalize_skill

CHUNK_CHARS = 500
MAX_CHUNKS_PER_SOURCE = 40


def chunk_text(text: str, max_chars: int = CHUNK_CHARS) -> list:
    """Group lines into chunks of about max_chars; the last line repeats as overlap."""
    lines = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        while len(line) > max_chars:  # hard-split very long lines
            lines.append(line[:max_chars])
            line = line[max_chars:]
        if line:
            lines.append(line)

    chunks, current, size = [], [], 0
    for line in lines:
        if current and size + len(line) + 1 > max_chars:
            chunks.append(" ".join(current))
            current = current[-1:]  # one-line overlap keeps context across chunks
            size = len(current[0]) + 1
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks[:MAX_CHUNKS_PER_SOURCE]


def employee_skills(employee, summary) -> dict:
    """{normalized_key: label} for HR-typed plus AI-extracted skills."""
    typed = [s for s in (employee.primary_skills or "").split(",") if s.strip()]
    labels = merge_skills(typed, (summary or {}).get("skills", []))
    return {normalize_skill(s): s for s in labels if normalize_skill(s)}

def _cert_names(certs) -> list:
    """Certification rows or dicts -> ['AWS Solutions Architect (Amazon)', ...]."""
    names = []
    for c in certs or []:
        name = c.get("name", "") if isinstance(c, dict) else getattr(c, "name", "")
        issuer = c.get("issuer", "") if isinstance(c, dict) else getattr(c, "issuer", "")
        if name:
            names.append(f"{name} ({issuer})" if issuer else name)
    return names


def build_sources(summary, feedback_list, texts, designation="", certs=None) -> dict:

    """{source: [chunk texts]} for one employee. An empty list means 'nothing for this source'.

    review  = pros, plus comments from reviews rated 3+.
    concern = cons, plus comments from reviews rated 1-2.
    """
    summary = summary or {}
    parts = []
    if designation:
        parts.append(f"Designation: {designation}")
    if summary.get("organization_summary"):
        parts.append(summary["organization_summary"])
    if summary.get("top_strengths"):
        parts.append("Strengths: " + "; ".join(summary["top_strengths"]))
    names = _cert_names(certs)
    if names:
        parts.append("Certifications: " + "; ".join(names))

    reviews, concerns = [], []
    for f in feedback_list:
        head = f"Project {f.project_name or 'n/a'} (rating {f.rating}/5)."
        good = []
        if f.pros:
            good.append(f"Pros: {f.pros}")
        if f.comments and f.rating >= 3:
            good.append(f"Comments: {f.comments}")
        if good:
            reviews.append(f"{head} " + " ".join(good))
        bad = []
        if f.cons:
            bad.append(f"Cons: {f.cons}")
        if f.comments and f.rating <= 2:
            bad.append(f"Comments: {f.comments}")
        if bad:
            concerns.append(f"{head} " + " ".join(bad))

    return {
        "resume": chunk_text(texts.get("resume", "")),
        "one_slider": chunk_text(texts.get("one_slider", "")),
        "summary": chunk_text("\n".join(parts)),
        "review": reviews[:MAX_CHUNKS_PER_SOURCE],
        "concern": concerns[:MAX_CHUNKS_PER_SOURCE],
    }


def _source_hash(chunks: list, model: str) -> str:
    return emb.make_hash("\n".join(chunks), model)


def _index_skills(skill_map: dict, model: str) -> int:
    """Embed skills that have no vector yet for this model. Returns how many were added."""
    have = db.get_skill_keys(model)
    missing = [(key, label) for key, label in skill_map.items() if key not in have]
    if not missing:
        return 0
    vectors = emb.embed_documents([label for _, label in missing], model=model)
    db.save_skill_vectors(model, missing, vectors)
    return len(missing)


def index_employee(employee_id: int) -> int:
    """Embed whatever is new or changed for one employee. Returns how many vectors were written.
    Raises embeddings.EmbeddingError if Ollama or the model is unavailable."""
    employee = db.get_employee(employee_id)
    if employee is None:
        return 0
    model = models.get_embed_model()
    summary = db.get_summary(employee_id)
    texts = {t: (d.extracted_text or "") for t, d in db.get_documents(employee_id).items()}
    sources = build_sources(
        summary, db.get_feedback(employee_id), texts,
        designation=employee.designation or (summary or {}).get("designation", ""),
        certs=db.get_certifications(employee_id),
    )
    meta = db.get_chunk_meta()

    written = 0
    for source in db.CHUNK_SOURCES:
        chunks = sources.get(source, [])
        have = meta.get((employee_id, source, model))
        if not chunks:
            if have is not None:  # source disappeared: drop its old chunks
                db.replace_chunks(employee_id, source, model, "", [], [])
            continue
        source_hash = _source_hash(chunks, model)
        if have == source_hash:
            continue  # unchanged
        vectors = emb.embed_documents(chunks, model=model)
        db.replace_chunks(employee_id, source, model, source_hash, chunks, vectors)
        written += len(chunks)

    written += _index_skills(employee_skills(employee, summary), model)
    return written


def stale_ids() -> list:
    """Employee ids whose vectors (for the selected model) are missing, outdated or orphaned."""
    model = models.get_embed_model()
    meta = db.get_chunk_meta()
    skill_keys = db.get_skill_keys(model)
    summaries = db.get_all_summaries()
    feedback = db.get_all_feedback()
    all_texts = db.get_all_document_texts()
    all_certs = db.get_all_certifications()

    stale = []
    for employee in db.get_all_employees():
        summary = summaries.get(employee.id)
        sources = build_sources(
            summary, feedback.get(employee.id, []), all_texts.get(employee.id, {}),
            designation=employee.designation or (summary or {}).get("designation", ""),
            certs=all_certs.get(employee.id, []),
        )
        is_stale = False
        for source in db.CHUNK_SOURCES:
            chunks = sources.get(source, [])
            have = meta.get((employee.id, source, model))
            if chunks and have != _source_hash(chunks, model):
                is_stale = True
                break
            if not chunks and have is not None:
                is_stale = True
                break
        if not is_stale:
            is_stale = any(key not in skill_keys for key in employee_skills(employee, summary))
        if is_stale:
            stale.append(employee.id)
    return stale


def reindex_all() -> tuple:
    """Bring every employee up to date. Returns (vectors_written, error_message_or_None).
    Stops at the first embedding error instead of failing once per employee."""
    written = 0
    for employee_id in stale_ids():
        try:
            written += index_employee(employee_id)
        except emb.EmbeddingError as exc:
            return written, str(exc)
    return written, None