# Tredence Bench Hub

Internal tool for bench employees. HR imports profiles (resume + one-slider), managers add project-based reviews, a local AI writes a summary per employee, and project managers search the bench and open profiles.

**Everything runs locally.** Ollama models only, no cloud, no API keys, no Docker. SQLite now, Postgres later.

## Stack

Streamlit (>=1.37) | SQLModel + SQLite (`bench.db`) | sqlite-vec (vector search) | pdfplumber, python-docx, python-pptx | Ollama | python-dotenv | pandas

## Setup (Windows, VS Code, venv)

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env

ollama pull qwen2.5:7b              # summary model (default)
ollama pull qwen3-embedding:0.6b    # search model (default)
# optional alternatives for the sidebar dropdowns:
ollama pull qwen2.5:3b
ollama pull llama3.1:8b
ollama pull nomic-embed-text

streamlit run app.py
```

Always run commands from the **project root**. Relative paths (`bench.db`, `.streamlit/`, `assets/`) depend on it.

## Configuration (`.env`)

| Key | Default | Purpose |
|---|---|---|
| `BENCH_DATABASE_URL` | `sqlite:///bench.db` | Database. Not `DATABASE_URL` (clashes with a system variable). |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Default summary model |
| `EMBED_MODEL` | `qwen3-embedding:0.6b` | Default search (embedding) model |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama address |
| `OLLAMA_NUM_CTX` / `OLLAMA_MAX_OUTPUT` / `OLLAMA_TIMEOUT` / `OLLAMA_KEEP_ALIVE` | 4096 / 600 / 180 / 30m | Summary model limits |
| `EMBED_TIMEOUT` / `EMBED_MAX_CHARS` / `EMBED_BATCH` | 60 / 6000 / 16 | Embedding limits |
| `IMPORT_FOLDER` | empty | Default folder for bulk import |
| `SUMMARY_ON_FEEDBACK` | `false` | `true` regenerates the AI summary whenever a manager submits a review |
| `CONTACT_ALLOWED_DOMAIN` | `tredence.com` | Send IM / Send Mail work only for emails in this domain |

## Project layout

```
bench-platform/
├── app.py                      Streamlit entry point: sidebar, Admin/HR tab, Dashboard tab
├── src/
│   ├── db/database.py          Models, engine, all save/read functions, vector queries
│   ├── ai/
│   │   ├── ai_engine.py        Text extraction (PDF/DOCX/PPTX) + local LLM summary
│   │   ├── embeddings.py       Ollama embeddings (model-aware, cached queries)
│   │   └── models.py           Model registry, current selection, score anchors
│   ├── search/
│   │   ├── skills.py           Skill normalization and de-duplication
│   │   ├── indexer.py          Chunking + skill vocabulary, embeds only what changed
│   │   └── search.py           Query parsing, scoring, stars (no Streamlit)
│   ├── services/
│   │   ├── importer.py         Folder scan, grouping, draft building
│   │   └── jobs.py             Background queue (summary and embed tasks)
│   └── ui/
│       ├── ui_theme.py         Styling only: CSS, logo, header, chips, footer
│       └── import_ui.py       Folder import screens
├── tests/test_database.py      DB tests on a throwaway DB
├── assets/                     logo.png, banner.png
├── .streamlit/config.toml      Theme
└── requirements.txt, .env, .env.example, .gitignore
```

Dependency rule: `ui` -> `services` -> `search` / `ai` / `db`. Only `app.py` and `src/ui/*` import Streamlit.

## Features

**Admin / HR tab**
- Bulk import from a folder. Files are grouped by Employee ID (the longest number in the file name), or by name when there is no ID. Choose rows, the AI reads them into drafts, you review and edit them, then upload.
- Add or edit a single employee (ID, name, email, skills, status, resume, one-slider).
- Manager feedback form: manager, project, rating 1-5, pros, cons, comments. The same manager and project replaces the earlier review.

**Manager Dashboard tab**
- Search box with hybrid search (see below), skills filter (must have all), status filter, minimum-match slider in stars.
- Results table with status chips and a **Match** column (1-5 stars, best first). **View** scrolls to the profile.
- Profile: organization summary, strengths, skills (HR-typed plus AI, merged), manager reviews with average rating, resume and one-slider downloads, **Send IM** (Teams) and **Send Mail** (mailto). Contact buttons act only for emails in `CONTACT_ALLOWED_DOMAIN`; otherwise they do nothing.

**Sidebar**
- Two dropdowns, **AI summary model** and **Search model**, each with an installed/ready check. Defaults come from `.env`. The choice applies to the whole app while it runs and is not saved across restarts.
- "Profiles updating: N" while background jobs run.

## How search works

Each query is split into concepts: "Azure ML knowledge, with sql expertise" becomes `azure ml` and `sql`. Words like *bad, weak, lacks* switch to concern mode, which searches only manager cons.

Each employee gets a score per concept, taking the best of:

| Evidence | Weight |
|---|---|
| Exact skill match | 100 |
| Related skill (skill vectors, e.g. "natural language processing" ~ "NLP") | up to 95 |
| Word found in reviews or AI summary / one-slider / resume text | 90 / 85 / 75 |
| A passage that means the same thing (chunk vectors) | up to 85 (reviews, summary) / 80 / 70 |

Concepts are combined so covering all of them beats covering one. Stars: 80-100 = 5, 60-79 = 4, 40-59 = 3, 20-39 = 2, below 20 = 1. Name and Employee ID matches score 100.

Vectors are stored per embedding model (`SkillVector`, `ChunkVector`). Switching the search model re-indexes in the background. Score anchors per model live in `models.py` and are tuned on real data.

## Background jobs

One worker thread runs tasks in order, since Ollama handles one job at a time.
- `summary`: regenerates the AI summary, then queues `embed`.
- `embed`: updates chunk and skill vectors for one employee. Unchanged text is skipped by hash.
- A sidebar fragment checks every few seconds for missing or outdated vectors (after imports, HR edits, model switches) and queues them. Failures show on the employee profile; Regenerate retries.
- The folder-import review stays synchronous because you need the draft summary while reviewing.

## Data model (SQLite)

`Employee`, `Document` (latest file per type, with extracted text), `Feedback` (one per employee + manager + project), `Summary` (one per employee), `SkillVector`, `ChunkVector`. `EmployeeEmbedding` is a leftover from an earlier design and is not used.

Rules: every DB function opens its own session; emails are stored lowercase; Employee ID is a unique string. New tables are created automatically on start, and existing data is never dropped. Delete `bench.db*` manually only when you want a fresh start.

## Tests

```powershell
python -m tests.test_database
```
Uses `test_bench.db` and deletes it afterwards. It never touches `bench.db`.

## Troubleshooting

- **Logo or banner missing:** check `assets/` and that you run from the project root.
- **`no such function: vec_distance_cosine`:** `pip install sqlite-vec` and restart. The extension loads on every SQLite connection.
- **Sidebar says model not installed:** run the `ollama pull` command it shows.
- **Search shows keyword matches only:** the search model is unavailable or still indexing. Check the sidebar.
- **Tabs jump or the page dims:** restart Streamlit and hard refresh (Ctrl+F5).
- **qwen2.5:3b summaries add details not in the sources.** Use the 7B model when accuracy matters.

## Roadmap

Skill filter "any" option, delete employee or documents from the UI, input-hash skip for summaries, login and roles, Postgres with pgvector and Alembic, file storage outside the DB, OCR for scanned files, logging and backups, per-user model selection, search tuning set (`tests/search_eval`), removal of the unused `EmployeeEmbedding` code.