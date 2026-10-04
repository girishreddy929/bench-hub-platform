"""Database layer for the Bench Platform (SQLModel + SQLite).

Portable by design: the connection string comes from DATABASE_URL, and the
SQLite-only settings are applied only when the URL starts with "sqlite".
Swapping to Postgres later is a config change.

Lists (top_strengths, manager_caveats) are stored as JSON text in the DB and
returned as Python lists by get_summary().
"""

import json
import os
import re
from datetime import datetime, timezone
from typing import Optional

import sqlite_vec

from dotenv import load_dotenv
from sqlalchemy import Column, LargeBinary, UniqueConstraint, event, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlmodel import Field, Session, SQLModel, col, create_engine, select

load_dotenv()

DATABASE_URL = os.getenv("BENCH_DATABASE_URL", "sqlite:///bench.db")

STATUSES = ["Available", "Interviewing", "Allocated"]
DOC_TYPES = ["resume", "one_slider"]
EMBED_KINDS = ["profile", "concerns"]  # two vectors per employee
CHUNK_SOURCES = ["resume", "one_slider", "summary", "review", "concern"]

def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class Employee(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    employee_code: str = Field(unique=True, index=True)  # business ID, e.g. "1226"
    name: str
    email: str = Field(unique=True, index=True)
    primary_skills: str = ""  # comma-separated
    availability_status: str = "Available"
    designation: str = ""  # job title, e.g. "Senior Data Scientist"
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

class Document(SQLModel, table=True):
    # Only the latest file of each type is kept per employee.
    __table_args__ = (UniqueConstraint("employee_id", "doc_type"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    employee_id: int = Field(foreign_key="employee.id", index=True)
    doc_type: str  # "resume" or "one_slider"
    file_name: str
    file_data: bytes = Field(sa_column=Column(LargeBinary, nullable=False))
    extracted_text: str = ""
    uploaded_at: datetime = Field(default_factory=utcnow)


class Feedback(SQLModel, table=True):
    # Entries are only added, never replaced: full history is kept.
    id: Optional[int] = Field(default=None, primary_key=True)
    employee_id: int = Field(foreign_key="employee.id", index=True)
    manager_name: str
    project_name: str = ""
    rating: int
    pros: str = ""
    cons: str = ""
    comments: str = ""
    created_at: datetime = Field(default_factory=utcnow)


class Summary(SQLModel, table=True):
    # One summary per employee, overwritten on every regeneration.
    id: Optional[int] = Field(default=None, primary_key=True)
    employee_id: int = Field(foreign_key="employee.id", unique=True, index=True)
    organization_summary: str = ""
    top_strengths: str = "[]"  # JSON list
    skills: str = "[]"  # JSON list
    projects: str = "[]"  # JSON list of {name, role, description, technologies, source}
    input_hash: str = ""  # hash of resume + one-slider + model: unchanged = the AI is skipped
    ai_designation: str = ""  # title read by the AI; HR's own title (Employee.designation) wins
    generated_at: datetime = Field(default_factory=utcnow)

class Certification(SQLModel, table=True):
    # source "ai"     = found automatically in the files (by the parser), replaced on every refresh
    # source "manual" = typed or confirmed by HR, never touched by the AI
    __table_args__ = (UniqueConstraint("employee_id", "cert_key"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    employee_id: int = Field(foreign_key="employee.id", index=True)
    cert_key: str  # normalized name, used to skip duplicates
    name: str
    issuer: str = ""
    year: str = ""
    source: str = "manual"
    created_at: datetime = Field(default_factory=utcnow)

class EmployeeEmbedding(SQLModel, table=True):
    # One vector per employee per kind. doc_hash + model tell us when it is stale.
    __table_args__ = (UniqueConstraint("employee_id", "kind"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    employee_id: int = Field(foreign_key="employee.id", index=True)
    kind: str  # "profile" or "concerns"
    model: str
    dim: int
    doc_hash: str
    vector: bytes = Field(sa_column=Column(LargeBinary, nullable=False))
    updated_at: datetime = Field(default_factory=utcnow)

class SkillVector(SQLModel, table=True):
    # One vector per distinct skill term per embedding model (the skill vocabulary).
    __table_args__ = (UniqueConstraint("skill_key", "model"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    skill_key: str = Field(index=True)  # normalized skill, e.g. "azure ml"
    label: str
    model: str
    dim: int
    vector: bytes = Field(sa_column=Column(LargeBinary, nullable=False))


class ChunkVector(SQLModel, table=True):
    # Small pieces of evidence (resume, one-slider, summary, each review, each concern).
    __table_args__ = (UniqueConstraint("employee_id", "source", "chunk_no", "model"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    employee_id: int = Field(foreign_key="employee.id", index=True)
    source: str  # one of CHUNK_SOURCES
    chunk_no: int
    chunk_text: str
    source_hash: str  # hash of the whole source text: unchanged source = skip re-embedding
    model: str
    dim: int
    vector: bytes = Field(sa_column=Column(LargeBinary, nullable=False))
    updated_at: datetime = Field(default_factory=utcnow)

# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
_is_sqlite = DATABASE_URL.startswith("sqlite")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
    pool_pre_ping=True,
)

if _is_sqlite:

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")  # readers + a writer at once
        cursor.execute("PRAGMA foreign_keys=ON")  # SQLite skips FKs by default
        cursor.close()
        if sqlite_vec is not None:  # vector functions, e.g. vec_distance_cosine
            try:
                dbapi_connection.enable_load_extension(True)
                sqlite_vec.load(dbapi_connection)
                dbapi_connection.enable_load_extension(False)
            except Exception:
                pass  # search falls back to a clear error, the rest of the app still works
            


def _session() -> Session:
    # expire_on_commit=False lets us return objects after the session closes.
    return Session(engine, expire_on_commit=False)


def init_db() -> None:
    """Create tables, then add any columns introduced after first creation."""
    SQLModel.metadata.create_all(engine)
    wanted = {
        "feedback": {"project_name": "VARCHAR DEFAULT ''"},
        "employee": {"designation": "VARCHAR DEFAULT ''"},
        "summary": {
            "projects": "VARCHAR DEFAULT '[]'",
            "input_hash": "VARCHAR DEFAULT ''",
            "ai_designation": "VARCHAR DEFAULT ''",
        },
    }
    inspector = inspect(engine)
    with engine.begin() as conn:
        for table, columns in wanted.items():
            present = {c["name"] for c in inspector.get_columns(table)}
            for column, ddl in columns.items():
                if column not in present:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))


# --------------------------------------------------------------------------
# Employees
# --------------------------------------------------------------------------
def save_employee(
    employee_code: str,
    name: str,
    email: str,
    primary_skills: str = "",
    availability_status: str = "Available",
    employee_id: Optional[int] = None,
    designation: Optional[str] = None,
) -> int:
    """Insert a new employee, or update one if employee_id (database id) is given.

    Returns the database id. Raises ValueError with a friendly message.
    """
    employee_code = (employee_code or "").strip()
    name = (name or "").strip()
    email = (email or "").strip().lower()
    primary_skills = ", ".join(
        s.strip() for s in (primary_skills or "").split(",") if s.strip()
    )

    if designation is not None:
        designation = " ".join(designation.split())[:120]
    if not employee_code:
        raise ValueError("Employee ID is required.")
    if not name:
        raise ValueError("Name is required.")
    if not email or "@" not in email:
        raise ValueError("A valid email is required.")
    if availability_status not in STATUSES:
        raise ValueError(f"Status must be one of: {', '.join(STATUSES)}.")

    with _session() as session:
        clash = session.exec(
            select(Employee).where(Employee.employee_code == employee_code)
        ).first()
        if clash and clash.id != employee_id:
            raise ValueError(f"Employee ID '{employee_code}' already belongs to {clash.name}.")
        clash = session.exec(select(Employee).where(Employee.email == email)).first()
        if clash and clash.id != employee_id:
            raise ValueError(
                f"The email '{email}' already belongs to {clash.name} (ID {clash.employee_code})."
            )

        if employee_id is not None:
            employee = session.get(Employee, employee_id)
            if employee is None:
                raise ValueError("Employee not found.")
            employee.employee_code = employee_code
            employee.name = name
            employee.email = email
            employee.primary_skills = primary_skills
            employee.availability_status = availability_status
            if designation is not None:
                employee.designation = designation
            employee.updated_at = utcnow()
        else:
            employee = Employee(
                employee_code=employee_code,
                name=name,
                email=email,
                primary_skills=primary_skills,
                availability_status=availability_status,
                designation=designation or "",
            )
        session.add(employee)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            raise ValueError("Employee ID or email already exists.")
        return employee.id


def get_employee_by_code(employee_code: str) -> Optional[Employee]:
    with _session() as session:
        return session.exec(
            select(Employee).where(Employee.employee_code == (employee_code or "").strip())
        ).first()

def get_employee(employee_id: int) -> Optional[Employee]:
    with _session() as session:
        return session.get(Employee, employee_id)


def get_all_employees() -> list[Employee]:
    with _session() as session:
        return list(session.exec(select(Employee).order_by(col(Employee.name))).all())


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------
def save_document(
    employee_id: int,
    doc_type: str,
    file_name: str,
    file_data: bytes,
    extracted_text: str = "",
) -> None:
    """Upsert: a new upload replaces the old file of the same type."""
    if doc_type not in DOC_TYPES:
        raise ValueError(f"doc_type must be one of: {', '.join(DOC_TYPES)}.")
    if not file_data:
        raise ValueError("The file is empty.")

    with _session() as session:
        if session.get(Employee, employee_id) is None:
            raise ValueError("Employee not found.")

        existing = session.exec(
            select(Document).where(
                Document.employee_id == employee_id,
                Document.doc_type == doc_type,
            )
        ).first()

        if existing:
            existing.file_name = file_name
            existing.file_data = file_data
            existing.extracted_text = extracted_text
            existing.uploaded_at = utcnow()
            session.add(existing)
        else:
            session.add(
                Document(
                    employee_id=employee_id,
                    doc_type=doc_type,
                    file_name=file_name,
                    file_data=file_data,
                    extracted_text=extracted_text,
                )
            )
        session.commit()


def get_documents(employee_id: int) -> dict[str, Document]:
    """Return {doc_type: Document} (at most one per type)."""
    with _session() as session:
        rows = session.exec(
            select(Document).where(Document.employee_id == employee_id)
        ).all()
        return {doc.doc_type: doc for doc in rows}


# --------------------------------------------------------------------------
# Feedback
# --------------------------------------------------------------------------
def add_feedback(
    employee_id: int,
    manager_name: str,
    project_name: str,
    rating: int,
    pros: str = "",
    cons: str = "",
    comments: str = "",
) -> bool:
    """Save a review. Same manager + same project replaces the old review.

    Returns True if an existing review was replaced.
    """
    manager_name = (manager_name or "").strip()
    project_name = (project_name or "").strip()
    if not manager_name:
        raise ValueError("Manager name is required.")
    if not project_name:
        raise ValueError("Project name is required.")
    if not isinstance(rating, int) or not 1 <= rating <= 5:
        raise ValueError("Rating must be a whole number from 1 to 5.")

    with _session() as session:
        if session.get(Employee, employee_id) is None:
            raise ValueError("Employee not found.")
        rows = session.exec(select(Feedback).where(Feedback.employee_id == employee_id)).all()
        existing = next(
            (
                f for f in rows
                if f.manager_name.strip().lower() == manager_name.lower()
                and (f.project_name or "").strip().lower() == project_name.lower()
            ),
            None,
        )
        feedback = existing or Feedback(employee_id=employee_id, manager_name=manager_name)
        feedback.manager_name = manager_name
        feedback.project_name = project_name
        feedback.rating = rating
        feedback.pros = (pros or "").strip()
        feedback.cons = (cons or "").strip()
        feedback.comments = (comments or "").strip()
        feedback.created_at = utcnow()
        session.add(feedback)
        session.commit()
        return existing is not None



def get_feedback(employee_id: int) -> list[Feedback]:
    """All feedback for an employee, newest first."""
    with _session() as session:
        return list(
            session.exec(
                select(Feedback)
                .where(Feedback.employee_id == employee_id)
                .order_by(col(Feedback.created_at).desc(), col(Feedback.id).desc())
            ).all()
        )


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
def save_summary(
    employee_id: int,
    organization_summary: str,
    top_strengths: list[str],
    skills: list[str],
    projects: Optional[list] = None,
    input_hash: Optional[str] = None,
    ai_designation: Optional[str] = None,
) -> None:
    """Upsert: one summary per employee, overwritten on every regeneration.

    projects=None and input_hash=None leave the stored values unchanged.
    """
    with _session() as session:
        if session.get(Employee, employee_id) is None:
            raise ValueError("Employee not found.")

        summary = session.exec(
            select(Summary).where(Summary.employee_id == employee_id)
        ).first()
        if summary is None:
            summary = Summary(employee_id=employee_id)

        summary.organization_summary = organization_summary
        summary.top_strengths = json.dumps(top_strengths)
        summary.skills = json.dumps(skills)
        if projects is not None:
            summary.projects = json.dumps(projects)
        if input_hash is not None:
            summary.input_hash = input_hash
        if ai_designation is not None:
            summary.ai_designation = " ".join(ai_designation.split())[:120]
        summary.generated_at = utcnow()
        session.add(summary)
        session.commit()


def save_projects(employee_id: int, projects: list, input_hash: Optional[str] = None) -> None:
    """Save only the projects (the other AI call may have failed). Creates the row if needed."""
    with _session() as session:
        if session.get(Employee, employee_id) is None:
            raise ValueError("Employee not found.")
        summary = session.exec(
            select(Summary).where(Summary.employee_id == employee_id)
        ).first()
        if summary is None:
            summary = Summary(employee_id=employee_id)
        summary.projects = json.dumps(projects)
        if input_hash is not None:
            summary.input_hash = input_hash
        session.add(summary)
        session.commit()


def _loads_list(text: str) -> list[str]:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return [text] if text else []
    return value if isinstance(value, list) else [str(value)]

def _loads_dicts(text: str) -> list[dict]:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def get_summary(employee_id: int) -> Optional[dict]:
    """Return the summary as a dict with real lists, or None."""
    with _session() as session:
        summary = session.exec(
            select(Summary).where(Summary.employee_id == employee_id)
        ).first()
        if summary is None:
            return None
        return {
            "organization_summary": summary.organization_summary,
            "top_strengths": _loads_list(summary.top_strengths),
            "skills": _loads_list(summary.skills),
            "projects": _loads_dicts(summary.projects),
            "input_hash": summary.input_hash or "",
            "designation": summary.ai_designation or "",
            "generated_at": summary.generated_at,
        }

def get_all_skills_map() -> dict[int, list[str]]:
    """{employee_id: AI-extracted skills} for every summary (used by search)."""
    with _session() as session:
        return {s.employee_id: _loads_list(s.skills) for s in session.exec(select(Summary)).all()}
    
# --------------------------------------------------------------------------
# Embeddings and bulk reads for search
# --------------------------------------------------------------------------
def save_embedding(employee_id: int, kind: str, model: str, vector: list, doc_hash: str) -> None:
    """Upsert one vector (kind = 'profile' or 'concerns')."""
    if kind not in EMBED_KINDS:
        raise ValueError(f"kind must be one of: {', '.join(EMBED_KINDS)}.")
    if sqlite_vec is None:
        raise ValueError("Vector search needs the sqlite-vec package (pip install sqlite-vec).")
    blob = sqlite_vec.serialize_float32(list(vector))

    with _session() as session:
        if session.get(Employee, employee_id) is None:
            raise ValueError("Employee not found.")
        row = session.exec(
            select(EmployeeEmbedding).where(
                EmployeeEmbedding.employee_id == employee_id,
                EmployeeEmbedding.kind == kind,
            )
        ).first()
        if row is None:
            row = EmployeeEmbedding(
                employee_id=employee_id, kind=kind, model=model,
                dim=len(vector), doc_hash=doc_hash, vector=blob,
            )
        else:
            row.model, row.dim, row.doc_hash, row.vector = model, len(vector), doc_hash, blob
            row.updated_at = utcnow()
        session.add(row)
        session.commit()


def delete_embedding(employee_id: int, kind: str) -> None:
    """Remove a vector, e.g. the 'concerns' one when an employee has no cons left."""
    with _session() as session:
        row = session.exec(
            select(EmployeeEmbedding).where(
                EmployeeEmbedding.employee_id == employee_id,
                EmployeeEmbedding.kind == kind,
            )
        ).first()
        if row is not None:
            session.delete(row)
            session.commit()


def get_embedding_meta() -> dict:
    """{(employee_id, kind): (model, doc_hash)} used to find stale or missing vectors."""
    with _session() as session:
        rows = session.exec(
            select(
                EmployeeEmbedding.employee_id, EmployeeEmbedding.kind,
                EmployeeEmbedding.model, EmployeeEmbedding.doc_hash,
            )
        ).all()
        return {(r[0], r[1]): (r[2], r[3]) for r in rows}


def get_similarities(kind: str, query_vector: list) -> dict:
    """{employee_id: cosine similarity} between the query and every stored vector of this kind."""
    if sqlite_vec is None:
        raise ValueError("Vector search needs the sqlite-vec package (pip install sqlite-vec).")
    blob = sqlite_vec.serialize_float32(list(query_vector))
    with engine.connect() as conn:
        rows = conn.exec_driver_sql(
            "SELECT employee_id, 1.0 - vec_distance_cosine(vector, ?) "
            "FROM employeeembedding WHERE kind = ?",
            (blob, kind),
        ).fetchall()
    return {r[0]: float(r[1]) for r in rows}


def get_all_summaries() -> dict:
    """{employee_id: {organization_summary, top_strengths, skills}} for every summary."""
    with _session() as session:
        return {
            s.employee_id: {
                "organization_summary": s.organization_summary,
                "top_strengths": _loads_list(s.top_strengths),
                "skills": _loads_list(s.skills),
                "designation": s.ai_designation or "",
            }
            for s in session.exec(select(Summary)).all()
        }


def get_all_documents_text() -> dict:
    """{employee_id: resume + one-slider text joined} for keyword evidence."""
    with _session() as session:
        rows = session.exec(select(Document.employee_id, Document.extracted_text)).all()
    out: dict = {}
    for employee_id, text_value in rows:
        out[employee_id] = (out.get(employee_id, "") + "\n" + (text_value or "")).strip()
    return out


def get_all_feedback() -> dict:
    """{employee_id: [Feedback, newest first]} for every employee."""
    with _session() as session:
        rows = session.exec(
            select(Feedback).order_by(col(Feedback.created_at).desc(), col(Feedback.id).desc())
        ).all()
    out: dict = {}
    for feedback in rows:
        out.setdefault(feedback.employee_id, []).append(feedback)
    return out

  
# --------------------------------------------------------------------------
# Skill vectors and evidence chunks (model-aware)
# --------------------------------------------------------------------------
def _need_vec() -> None:
    if sqlite_vec is None:
        raise ValueError("Vector search needs the sqlite-vec package (pip install sqlite-vec).")


def get_skill_keys(model: str) -> set:
    """Skill keys that already have a vector for this embedding model."""
    with _session() as session:
        rows = session.exec(select(SkillVector.skill_key).where(SkillVector.model == model)).all()
        return set(rows)


def save_skill_vectors(model: str, items: list, vectors: list) -> None:
    """Upsert vectors. items = [(skill_key, label), ...], vectors aligned with items."""
    _need_vec()
    if len(items) != len(vectors):
        raise ValueError("Skills and vectors must have the same length.")
    with _session() as session:
        existing = {
            row.skill_key: row
            for row in session.exec(select(SkillVector).where(SkillVector.model == model)).all()
        }
        for (key, label), vector in zip(items, vectors):
            blob = sqlite_vec.serialize_float32(list(vector))
            row = existing.get(key)
            if row is None:
                row = SkillVector(skill_key=key, label=label, model=model, dim=len(vector), vector=blob)
                existing[key] = row
            else:
                row.label, row.dim, row.vector = label, len(vector), blob
            session.add(row)
        session.commit()


def get_skill_similarities(model: str, query_vector: list) -> dict:
    """{skill_key: cosine similarity} between the query and every skill vector of this model."""
    _need_vec()
    blob = sqlite_vec.serialize_float32(list(query_vector))
    with engine.connect() as conn:
        rows = conn.exec_driver_sql(
            "SELECT skill_key, 1.0 - vec_distance_cosine(vector, ?) "
            "FROM skillvector WHERE model = ?",
            (blob, model),
        ).fetchall()
    return {r[0]: float(r[1]) for r in rows}


def get_chunk_meta() -> dict:
    """{(employee_id, source, model): source_hash} to find missing or outdated chunks."""
    with _session() as session:
        rows = session.exec(
            select(
                ChunkVector.employee_id, ChunkVector.source,
                ChunkVector.model, ChunkVector.source_hash,
            ).distinct()
        ).all()
        return {(r[0], r[1], r[2]): r[3] for r in rows}


def replace_chunks(
    employee_id: int, source: str, model: str, source_hash: str, chunks: list, vectors: list
) -> None:
    """Replace all chunks of one source for one employee and model. Empty chunks = delete only."""
    if source not in CHUNK_SOURCES:
        raise ValueError(f"source must be one of: {', '.join(CHUNK_SOURCES)}.")
    if len(chunks) != len(vectors):
        raise ValueError("Chunks and vectors must have the same length.")
    if chunks:
        _need_vec()
    with _session() as session:
        if session.get(Employee, employee_id) is None:
            raise ValueError("Employee not found.")
        for row in session.exec(
            select(ChunkVector).where(
                ChunkVector.employee_id == employee_id,
                ChunkVector.source == source,
                ChunkVector.model == model,
            )
        ).all():
            session.delete(row)
        session.flush()  # deletes must reach the DB before the inserts below
        for number, (text_value, vector) in enumerate(zip(chunks, vectors)):
            session.add(
                ChunkVector(
                    employee_id=employee_id, source=source, chunk_no=number,
                    chunk_text=text_value, source_hash=source_hash, model=model,
                    dim=len(vector),
                    vector=sqlite_vec.serialize_float32(list(vector)),
                )
            )
        session.commit()


def get_chunk_similarities(model: str, query_vector: list, sources: list) -> dict:
    """{employee_id: {source: best cosine similarity}} over the chosen sources."""
    _need_vec()
    if not sources:
        return {}
    blob = sqlite_vec.serialize_float32(list(query_vector))
    marks = ",".join("?" for _ in sources)
    with engine.connect() as conn:
        rows = conn.exec_driver_sql(
            "SELECT employee_id, source, MAX(1.0 - vec_distance_cosine(vector, ?)) "
            f"FROM chunkvector WHERE model = ? AND source IN ({marks}) "
            "GROUP BY employee_id, source",
            (blob, model, *sources),
        ).fetchall()
    out: dict = {}
    for employee_id, source, similarity in rows:
        out.setdefault(employee_id, {})[source] = float(similarity)
    return out

def get_all_document_texts() -> dict:
    """{employee_id: {doc_type: extracted_text}} for every employee (no file bytes)."""
    with _session() as session:
        rows = session.exec(
            select(Document.employee_id, Document.doc_type, Document.extracted_text)
        ).all()
    out: dict = {}
    for employee_id, doc_type, text_value in rows:
        out.setdefault(employee_id, {})[doc_type] = text_value or ""
    return out

# --------------------------------------------------------------------------
# Designation and certifications
# --------------------------------------------------------------------------
MAX_CERTS = 20


def set_designation_if_empty(employee_id: int, designation: str) -> bool:
    """Fill the designation from the AI only when HR has not set one. True if it was set."""
    designation = " ".join((designation or "").split())[:120]
    if not designation:
        return False
    with _session() as session:
        employee = session.get(Employee, employee_id)
        if employee is None or (employee.designation or "").strip():
            return False
        employee.designation = designation
        employee.updated_at = utcnow()
        session.add(employee)
        session.commit()
        return True


def _cert_key(name: str) -> str:
    return " ".join(re.findall(r"[a-z0-9+#.]+", (name or "").lower()))


def _clean_cert(item) -> Optional[dict]:
    if isinstance(item, str):
        item = {"name": item}
    if not isinstance(item, dict):
        return None
    name = " ".join(str(item.get("name") or "").split())[:150]
    if not _cert_key(name):
        return None
    return {
        "name": name,
        "issuer": " ".join(str(item.get("issuer") or "").split())[:80],
        "year": " ".join(str(item.get("year") or "").split())[:10],
    }


def _replace_certs(employee_id: int, certs: list, source: str, only_ai: bool) -> int:
    """Delete old rows (only the 'ai' ones, or all), then insert the new list. Returns rows added."""
    with _session() as session:
        if session.get(Employee, employee_id) is None:
            raise ValueError("Employee not found.")
        query = select(Certification).where(Certification.employee_id == employee_id)
        if only_ai:
            query = query.where(Certification.source == "ai")
        for row in session.exec(query).all():
            session.delete(row)
        session.flush()  # deletes must reach the DB before the inserts below
        seen = {
            row.cert_key
            for row in session.exec(
                select(Certification).where(Certification.employee_id == employee_id)
            ).all()
        }
        added = 0
        for item in certs or []:
            clean = _clean_cert(item)
            if clean is None:
                continue
            key = _cert_key(clean["name"])
            if key in seen or added >= MAX_CERTS:
                continue
            seen.add(key)
            session.add(Certification(employee_id=employee_id, cert_key=key, source=source, **clean))
            added += 1
        session.commit()
        return added


def save_ai_certifications(employee_id: int, certs: list) -> int:
    """Replace the AI-read certifications. HR-entered ones are kept and win on duplicates."""
    return _replace_certs(employee_id, certs, "ai", only_ai=True)


def save_manual_certifications(employee_id: int, certs: list) -> int:
    """HR's list is the full truth: replaces every certification (AI and manual)."""
    return _replace_certs(employee_id, certs, "manual", only_ai=False)


def get_certifications(employee_id: int) -> list:
    """Certifications of one employee, in the order they were added."""
    with _session() as session:
        return list(
            session.exec(
                select(Certification)
                .where(Certification.employee_id == employee_id)
                .order_by(col(Certification.id))
            ).all()
        )


def get_all_certifications() -> dict:
    """{employee_id: [Certification, ...]} for every employee (used by the search index)."""
    with _session() as session:
        rows = session.exec(select(Certification).order_by(col(Certification.id))).all()
    out: dict = {}
    for cert in rows:
        out.setdefault(cert.employee_id, []).append(cert)
    return out


def format_certifications(certs: list) -> str:
    """Certification rows -> the HR text box: one 'Name | Issuer | Year' per line."""
    lines = []
    for c in certs:
        parts = [c.name, c.issuer, c.year] if (c.issuer or c.year) else [c.name]
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def parse_certifications(text: str) -> list:
    """The HR text box -> [{name, issuer, year}]. Blank lines and bullet marks are ignored."""
    out = []
    for raw in (text or "").splitlines():
        line = raw.strip().lstrip("-*•").strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        out.append(
            {
                "name": parts[0],
                "issuer": parts[1] if len(parts) > 1 else "",
                "year": parts[2] if len(parts) > 2 else "",
            }
        )
    return out


def save_ai_result(employee_id: int, result: dict, input_hash: Optional[str] = None) -> None:
    """Save what generate_summary() returned.

    Optional parts are saved only if they are in the result: a missing 'projects' keeps the stored
    ones, a missing 'certifications' keeps the stored ones. The AI's title is stored separately
    from HR's (HR's title wins on display). input_hash marks these inputs as processed.
    """
    save_summary(
        employee_id,
        result["organization_summary"],
        result.get("top_strengths", []),
        result.get("skills", []),
        projects=result.get("projects"),
        input_hash=input_hash,
        ai_designation=result.get("designation"),
    )
    if "certifications" in result:
        save_ai_certifications(employee_id, result["certifications"])