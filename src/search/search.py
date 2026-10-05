"""Hybrid search: query concepts -> exact skills + keywords + vectors -> 0-100 score -> stars.
No Streamlit here."""

import re
from dataclasses import dataclass, field

from src.ai import embeddings as emb
from src.ai import models
from src.db import database as db
from src.search import indexer
from src.search.skills import compact_skill, normalize_skill
from src.services import timing

POSITIVE_SOURCES = ["review", "summary", "one_slider", "resume"]

# Explicit words outrank fuzzy vector matches; feedback > one-slider > resume.
KEYWORD_WEIGHT = {"review": 0.90, "summary": 0.90, "one_slider": 0.85, "resume": 0.75, "concern": 0.90}
SEMANTIC_WEIGHT = {"review": 0.85, "summary": 0.85, "one_slider": 0.80, "resume": 0.70, "concern": 0.85}
SKILL_SEMANTIC_CAP = 0.95  # a related skill never beats an exact skill match (1.0)

FILLER = {
    "a", "an", "the", "of", "in", "on", "at", "for", "to", "is", "are", "was", "who", "whom",
    "that", "which", "has", "have", "having", "had", "do", "does", "can", "know", "knows",
    "knowledge", "knowing", "expert", "experts", "expertise", "experience", "experienced",
    "skill", "skills", "skilled", "proficiency", "proficient", "strong", "good", "great",
    "hands", "work", "worked", "working", "works", "project", "projects", "person", "people",
    "employee", "employees", "someone", "anyone", "candidate", "candidates", "resource",
    "resources", "profile", "profiles", "show", "find", "list", "me", "need", "needs", "want",
    "looking", "look", "familiar", "familiarity", "background", "exposure", "ability", "able",
    "using", "use", "used", "about", "related", "like", "similar",
    "certified", "certification", "certifications", "certificate", "certificates",
}
NEGATIVE_RE = re.compile(
    r"\b(bad|weak|weakness(?:es)?|poor|poorly|lacks?|lacking|struggl\w*|worst|"
    r"needs? improvement|not good|not strong|concerns?|cons|improve\w*)\b"
)
SPLIT_RE = re.compile(r"[,;&]|\band\b|\bwith\b|\bplus\b|\balso\b")


@dataclass
class Hit:
    employee_id: int
    score: int  # 0-100
    stars: int  # 1-5


@dataclass
class Outcome:
    hits: list = field(default_factory=list)
    concepts: list = field(default_factory=list)
    concern_mode: bool = False
    notes: list = field(default_factory=list)


def stars_for(score: float) -> int:
    """80-100 = 5 stars, 60-79 = 4, 40-59 = 3, 20-39 = 2, below 20 = 1."""
    for stars, minimum in ((5, 80), (4, 60), (3, 40), (2, 20)):
        if score >= minimum:
            return stars
    return 1


def star_text(stars: int) -> str:
    return "★" * stars + "☆" * (5 - stars)


def _split_known(words: list, vocab: set):
    """Split 'azure ml sql python' into known skills; None if not fully covered."""
    parts, i = [], 0
    while i < len(words):
        for j in range(len(words), i, -1):
            phrase = " ".join(words[i:j])
            if phrase in vocab:
                parts.append(phrase)
                i = j
                break
        else:
            return None
    return parts if len(parts) > 1 else None


def parse_query(query: str, vocab: set) -> tuple:
    """Return (concern_mode, [concepts]).

    "Azure ML knowledge, with sql expertise" -> (False, ["azure ml", "sql"])
    "who is bad at azure ml"                 -> (True,  ["azure ml"])
    """
    text = (query or "").lower()
    concern_mode = bool(NEGATIVE_RE.search(text))
    text = NEGATIVE_RE.sub(" ", text)

    concepts = []
    for segment in SPLIT_RE.split(text):
        words = normalize_skill(segment).split()
        while words and words[0] in FILLER:
            words.pop(0)
        while words and words[-1] in FILLER:
            words.pop()
        if not words:
            continue
        phrase = " ".join(words)
        parts = None if phrase in vocab else _split_known(words, vocab)
        concepts.extend(parts or [phrase])

    unique, seen = [], set()
    for concept in concepts:
        key = compact_skill(concept)
        if key and key not in seen:
            seen.add(key)
            unique.append(concept)
    if not unique:  # only filler words were typed: use the text as it is
        fallback = normalize_skill(NEGATIVE_RE.sub(" ", (query or "").lower()))
        if fallback:
            unique = [fallback]
    return concern_mode, unique


# ---- Part 2 (data loading, scoring, run_search)  ----

def _scale(similarity: float, floor: float, ceiling: float) -> float:
    if ceiling <= floor:
        return 0.0
    return max(0.0, min(1.0, (similarity - floor) / (ceiling - floor)))


def _keyword_regex(concept: str):
    """Whole-word match that tolerates spaces, hyphens, slashes or none ('azure ml' ~ 'azureml')."""
    if len(compact_skill(concept)) < 2:
        return None
    tokens = [re.escape(t) for t in concept.split() if t]
    if not tokens:
        return None
    return re.compile(r"(?<![a-z0-9])" + r"[\s\-_/]*".join(tokens) + r"(?![a-z0-9])")


def _load_bundle(employees: list) -> dict:
    """Per-employee skills and searchable text, loaded with a few bulk queries."""
    summaries = db.get_all_summaries()
    feedback = db.get_all_feedback()
    texts = db.get_all_document_texts()
    certs = db.get_all_certifications()
    bundle = {}
    for employee in employees:
        summary = summaries.get(employee.id)
        doc_texts = texts.get(employee.id, {})
        sources = indexer.build_sources(
            summary, feedback.get(employee.id, []), {},
            designation=employee.designation or (summary or {}).get("designation", ""),
            certs=certs.get(employee.id, []),
        )
        skills = indexer.employee_skills(employee, summary)
        bundle[employee.id] = {
            "skills": list(skills),
            "compacts": {key.replace(" ", "") for key in skills},
            "corpus": {
                "review": "\n".join(sources["review"]).lower(),
                "summary": "\n".join(sources["summary"]).lower(),
                "one_slider": doc_texts.get("one_slider", "").lower(),
                "resume": doc_texts.get("resume", "").lower(),
                "concern": "\n".join(sources["concern"]).lower(),
            },
        }
    return bundle


def _semantic(concept: str, model: str, concern_mode: bool, notes: list) -> tuple:
    """(skill_sims, chunk_sims, anchors) for one concept. Empty sims if vectors are unavailable."""
    anchors = models.anchors(model)
    try:
        query_vector = emb.embed_query(concept, model=model)
        skill_sims = {} if concern_mode else db.get_skill_similarities(model, query_vector)
        sources = ["concern"] if concern_mode else POSITIVE_SOURCES
        chunk_sims = db.get_chunk_similarities(model, query_vector, sources)
    except Exception as exc:
        note = f"Semantic matching unavailable ({exc}). Showing keyword matches only."
        if note not in notes:
            notes.append(note)
        return {}, {}, anchors
    return skill_sims, chunk_sims, anchors


def _concept_score(employee_id, info, concern_mode, regex, concept_key, skill_sims, chunk_sims, anchors) -> float:
    """0.0-1.0: how well one employee covers one concept."""
    best = 0.0
    sources = ["concern"] if concern_mode else POSITIVE_SOURCES

    if not concern_mode:
        if concept_key in info["compacts"]:
            best = 1.0  # exact skill
        else:
            sims = [skill_sims[k] for k in info["skills"] if k in skill_sims]
            if sims:
                scaled = _scale(max(sims), anchors["skill_floor"], anchors["skill_ceiling"])
                best = max(best, SKILL_SEMANTIC_CAP * scaled)  # related skill

    if regex is not None:
        for source in sources:
            if regex.search(info["corpus"][source]):
                best = max(best, KEYWORD_WEIGHT[source])  # the word appears in the text

    for source, similarity in chunk_sims.get(employee_id, {}).items():
        scaled = _scale(similarity, anchors["chunk_floor"], anchors["chunk_ceiling"])
        best = max(best, SEMANTIC_WEIGHT[source] * scaled)  # a passage means the same thing
    return best


def _combine(scores: list) -> float:
    """Covering every concept beats covering one strongly."""
    if len(scores) == 1:
        return scores[0]
    return 0.75 * (sum(scores) / len(scores)) + 0.25 * min(scores)

def _explain(concept, skill_sims, chunk_sims, anchors, labels) -> None:
    """Timing log only: raw similarity statistics, used to tune the star anchors. Changes no result."""
    if not timing.enabled():
        return

    def stats(values) -> str:
        values = sorted(values)
        if not values:
            return "no values"
        pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]
        return f"min {values[0]:.2f} | median {pick(0.5):.2f} | p90 {pick(0.9):.2f} | max {values[-1]:.2f}"

    timing.log(
        f"  SEARCH '{concept}'  anchors: skill {anchors['skill_floor']}-{anchors['skill_ceiling']}, "
        f"chunk {anchors['chunk_floor']}-{anchors['chunk_ceiling']}"
    )
    if skill_sims:
        top = sorted(skill_sims.items(), key=lambda kv: -kv[1])[:5]
        timing.log(f"    skills      {stats(skill_sims.values())}")
        timing.log("      top skills: " + ", ".join(f"{k} {v:.2f}" for k, v in top))
    by_source = {}
    for emp_id, per_source in chunk_sims.items():
        for source, value in per_source.items():
            by_source.setdefault(source, []).append((value, emp_id))
    for source, pairs in by_source.items():
        timing.log(f"    {source:<11} {stats([v for v, _ in pairs])}")
        best = sorted(pairs, reverse=True)[:3]
        timing.log("      top: " + ", ".join(f"{labels.get(e, e)} {v:.2f}" for v, e in best))

def run_search(query: str, employees: list) -> Outcome:
    """Rank employees for a free-text query. Hits are sorted by score, best first."""
    outcome = Outcome()
    if not (query or "").strip() or not employees:
        return outcome

    with timing.span("search: load data"):
        bundle = _load_bundle(employees)
    vocab = {key for info in bundle.values() for key in info["skills"]}
    outcome.concern_mode, outcome.concepts = parse_query(query, vocab)
    if not outcome.concepts:
        return outcome

    model = models.get_embed_model()
    labels = {e.id: e.name for e in employees}
    per_concept = []
    for concept in outcome.concepts:
        with timing.span(f"search: vectors '{concept}'"):
            skill_sims, chunk_sims, anchors = _semantic(concept, model, outcome.concern_mode, outcome.notes)
        _explain(concept, skill_sims, chunk_sims, anchors, labels)
        per_concept.append((concept, _keyword_regex(concept), skill_sims, chunk_sims, anchors))

    direct = compact_skill(query)
    hits = []
    for employee in employees:
        info = bundle[employee.id]
        name_hit = len(direct) >= 2 and (
            direct in compact_skill(employee.name) or direct in compact_skill(employee.employee_code)
        )
        if name_hit:
            score = 1.0
        else:
            scores = [
                _concept_score(
                    employee.id, info, outcome.concern_mode, regex,
                    compact_skill(concept), skill_sims, chunk_sims, anchors,
                )
                for concept, regex, skill_sims, chunk_sims, anchors in per_concept
            ]
            score = _combine(scores)
        points = int(round(score * 100))
        if points > 0:
            hits.append(Hit(employee.id, points, stars_for(points)))

    names = {e.id: e.name.lower() for e in employees}
    hits.sort(key=lambda h: (-h.score, names[h.employee_id]))
    outcome.hits = hits
    if timing.enabled():
        timing.log(
            "  SEARCH result: "
            + ", ".join(f"{labels[h.employee_id]} {h.score}pt {h.stars}stars" for h in hits[:8])
        )
    return outcome