# Tredence Bench Hub

An internal tool for bench staffing. HR uploads or imports profiles (resume + one-slider). A **local** AI model
writes each profile (title, summary, skills, projects). Managers add project reviews. Project managers search
the bench by skills or meaning and open profiles. Everything runs on one laptop: no cloud, no API keys, no Docker.

Stack: Streamlit, SQLModel + SQLite (`bench.db`), sqlite-vec, Ollama (summary model + embedding model),
pdfplumber / python-docx / python-pptx.

## 1. Quick start (Windows, VS Code, venv)

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    ollama pull phi4-mini
    ollama pull qwen3-embedding:0.6b
    copy .env.example .env
    streamlit run app.py

Run everything from the project root. After editing `.env` do a full restart of Streamlit (it is read once).

Tests (no Ollama needed):

    python -m tests.test_database
    python -m tests.test_ai_engine

## 2. Configuration (`.env`)

| Key | Default | Meaning |
|---|---|---|
| `BENCH_DATABASE_URL` | `sqlite:///bench.db` | Database (never `DATABASE_URL`) |
| `OLLAMA_MODEL` | `phi4-mini` | Summary model |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama address |
| `OLLAMA_NUM_CTX` | `6144` | Context window |
| `OLLAMA_KEEP_ALIVE` | `30m` | How long models stay loaded (`24h` avoids cold starts) |
| `OLLAMA_MAX_OUTPUT` | `600` | Upper output limit per call |
| `OLLAMA_TIMEOUT` | `180` | Per call; do not raise, shorten the work instead |
| `OLLAMA_NUM_THREAD` | `0` | CPU threads for the summary model (8 leaves room for the UI on 10 cores) |
| `MAX_RESUME_CHARS` / `MAX_SLIDER_CHARS` | `9000` / `6000` | Safety caps only |
| `EMBED_MODEL` | `qwen3-embedding:0.6b` | Search model |
| `EMBED_TIMEOUT`, `EMBED_MAX_CHARS`, `EMBED_BATCH` | `60`, `6000`, `16` | Embedding limits |
| `IMPORT_FOLDER` | empty | Default folder for the folder import |
| `CONTACT_ALLOWED_DOMAIN` | `tredence.com` | Contact buttons work only for this domain |
| `TIMING_LOG` | `0` | `1` prints timing lines (see section 6) |
| `TIMING_LOG_FILE` | empty | Optional file, e.g. `logs/timing.log` |

## 3. Every input: where it lives, when it is read, when it changes

| Input | Comes from | Stored in | Read when | Updated when | AI? |
|---|---|---|---|---|---|
| Employee ID, name, email, status, primary skills | HR form / folder import | `Employee` | lists, filters, header, search | HR edits | No |
| Title typed by HR | HR form | `Employee.designation` | header, search index (always wins) | HR edits | No |
| Resume and one-slider | upload / import | `Document` (bytes + extracted text) | downloads, AI input, word search | a new upload replaces the old file | Text is read by code |
| Certifications found in files | parser | `Certification` (source `ai`) | profile, search index | every upload and summary job (replaced) | **No (code)** |
| Certifications typed by HR | HR form / import review | `Certification` (source `manual`) | same | only when HR changes the box; never overwritten | No |
| AI title | AI call 1 | `Summary.ai_designation` | header if HR left the title empty | every AI run | **Local LLM** |
| Executive summary, skills | AI call 1 | `Summary` | profile, skill filter, skill search | every AI run | **Local LLM** |
| Projects (max 4) | AI call 2 | `Summary.projects` | profile, Primary vs Secondary skills | every AI run | **Local LLM** |
| Fingerprint of the files | job | `Summary.input_hash` | start of each summary job | after a fully successful run | No |
| Manager reviews | manager form | `Feedback` | Manager Review section, average, search | manager submits (same manager + project replaces) | No |
| Vectors (meaning) | index job | `ChunkVector`, `SkillVector` | search only | index job, when a source's text hash changed | **Embedding model** |
| Folder-import drafts | background read | memory only (`jobs._reads`) | review window | until uploaded or Streamlit restarts | **Local LLM** |
| Scores and stars | search | not stored | computed on every search | every search | Query vector only |

## 4. What happens when something changes

| Event | What runs | What the user sees |
|---|---|---|
| HR saves a profile **with** a file | save, certification parser (instant), search-index job, summary job | certifications at once; summary within a few minutes |
| HR saves **without** a file | save, search-index job | searchable in seconds |
| HR edits certifications | HR list replaces all rows, index job | shown at once |
| Folder import: **Read** | one background job per profile (both AI calls + parser); page returns at once | progress line, then a **Review** button |
| Folder import: **Upload** (new person) | saves employee, files, reviewed certifications, summary, projects | profile complete after upload |
| Folder import, existing person | files replaced, summary job queued | old summary shown until the job ends |
| Manager adds a review | review saved, search-index job only (**no AI summary**) | visible at once; meaning search within seconds |
| Regenerate button | forced AI run | "Profile is being updated..." notice |
| Regenerate all | every profile with files; unchanged ones skipped by the fingerprint | sidebar: "Profiles updating: N" |
| Summary model switched | fingerprint includes the model | next Regenerate all redoes everyone |
| Search model switched | stale-vector check re-indexes in the background | search uses new vectors when done |
| Prompts changed in code | bump `PROMPT_VERSION` | old fingerprints stop matching |
| Streamlit restarted | queued jobs and unreviewed import drafts are lost | stale check re-queues missing vectors |

## 5. AI versus not AI

- **Local LLM (slow, writes text, temperature 0, only uses what is in the files):** title (headline + role),
  executive summary, AI skills, projects. Measured with Phi-4-mini on 10 CPU cores: about 5 tokens/s writing,
  about 80 s per profile warm (+25 s if the model was unloaded).
- **Embedding model (fast, makes vectors):** text chunks, skill names, the search query (about 0.07 s warm).
- **No AI at all:** reading files, certifications, reviews and average rating, filters, Primary/Secondary skills,
  keyword matching, scoring, stars, contact buttons, downloads, timing log.

## 6. Timing log (`TIMING_LOG=1`)

One block per action, printed in the Streamlit terminal (and in `TIMING_LOG_FILE` if set):

    START emp=1226 (Girish Reddy) src=regenerate model=phi4-mini queue_wait=0.3s
      read documents (db)              0.01s
      input: resume 5200 chars, one-slider 3100 chars
      AI summary   load 0.0s | read 18.2s (1150 tok) | write 31.0s (190 tok, 6.1 tok/s)
      AI projects  load 0.0s | read 2.9s (642 tok) | write 24.5s (150 tok, 6.1 tok/s)
    DONE  emp=1226 (Girish Reddy) src=regenerate total=77.0s (AI read 21.1s, AI write 55.5s, 2 calls) ok
    EMBED emp=1226 (Girish Reddy) src=regenerate model=qwen3-embedding:0.6b wait=0.4s run=2.1s
    BATCH src=regenerate_all  12 profiles: 9 ran, 3 skipped, 0 failed, wall 11m 42s

- `src` is the trigger: `upload`, `save`, `folder_import`, `folder_upload`, `regenerate`, `regenerate_all`, `review`, `stale_check`.
- `load` = model loading, `read` = the model reading the files, `write` = the model writing.
- Flags: `HIT OUTPUT CAP` (answer cut, JSON probably broken), `attempt 2` (first answer unusable), `CONTEXT NEARLY FULL`.
- Search prints `SEARCH '<concept>'` statistics (raw similarities), a `SEARCH result` line and `search: ...` time lines.

## 7. Searching, walk-through: "leadership"

1. One concept, no negative words: normal (positive) mode. The query becomes a vector.
2. The engine checks every profile:
   - Review says "showed strong leadership": word found in a review, score 90, about 5 stars.
   - Resume says "mentored teams of 3-5": no exact word, the meaning match finds it; resume weight is lower, 3-4 stars.
   - A skill named "Leadership" (typed or AI): exact skill, score 100.
3. "leader" or "led" alone does not match the word search (whole words only); only the meaning match catches them.
4. A low-rated review (1-2) saying "lacks leadership" is a concern, found only with a negative query such as "weak leadership".
5. Timing: a saved review shows at once; word matches work at once; meaning matches start seconds later.

Known limit: a vocabulary gap ("graphs" vs "visualization") depends on the embedding model, and the star
anchors in `models.py` are starting values that still need tuning on real data (see section 11).

## 8. Database utilities (stop Streamlit first)

    python -m tests.db_backup     bench.db -> bench_db_backup.db (replaces the old backup)
    python -m tests.db_restore    bench_db_backup.db -> bench.db
    python -m tests.db_clear      delete all rows, keep the tables

Each shows row counts and asks you to type `YES` (`--yes` skips the question). `bench.db` is never deleted automatically.

## 9. Project layout

    app.py                 sidebar, Admin/HR tab, Dashboard tab
    src/db/database.py     tables, bulk queries, vector tables
    src/ai/                ai_engine.py (files, prompts, model calls), embeddings.py, models.py
    src/search/            skills.py, indexer.py, search.py
    src/services/          jobs.py (workers), importer.py (folder import), timing.py (timing log)
    src/ui/                ui_theme.py, import_ui.py
    tests/                 test_database.py, test_ai_engine.py, db_backup.py, db_restore.py, db_clear.py
    assets/, .streamlit/config.toml, requirements.txt, .env, .env.example, MASTER_CONTEXT.md

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Dropdown shows the wrong default model | `OLLAMA_MODEL` set in `.env` or in the shell; `.env` is read once | fix `.env`, clear the shell variable, restart Streamlit fully |
| First search after a break takes seconds | the embedding model was unloaded | `OLLAMA_KEEP_ALIVE=24h` |
| First profile after a break is 25 s slower | the summary model was unloaded | same |
| Page dims / freezes on clicks | Streamlit re-runs the whole script; the CPU is fully used by the model | `OLLAMA_NUM_THREAD=8`; keep heavy work in background jobs |
| `AttributeError: ... has no attribute 'read_counts'` | `jobs.py` edits missing | apply the background-read edits in `jobs.py` |
| `use_container_width` / `st.components.v1.html` warnings | deprecated in Streamlit 1.64 | use `width="stretch"` and `st.iframe(...)` |
| New database column not found | `init_app` is cached | full Streamlit restart |
| Every profile shows about the same stars | anchors below the noise level of the search model | tune anchors (section 11) |

## 11. Known limits and next steps

- Search scoring: fuzzy matches start too low (noise around cosine 0.38-0.43 scores 25-35 points). Planned: the
  title as its own search source, anchors above the noise, generic suffix words as filler, an optional synonym file,
  and relative scoring once there are 40+ profiles.
- Each click re-runs the whole page, including the search. Planned: fragments for the dashboard and import
  sections, a short cache for the installed-model check, no file bytes on every re-run.
- Jobs and import drafts live in memory (lost on restart). One user at a time until login and roles exist.
- Later: delete employee from the UI, HR-editable projects, OCR for scanned files, Postgres (pgvector) + Alembic,
  file storage outside the database, login and roles.

## 12. The 60-second explanation

"HR uploads a resume and a one-slider, or imports a folder. The system reads the text and copies the
certifications exactly as written, with no AI. A local AI model on our own laptop, with no cloud, then writes the
title, a three-sentence summary, the skills and the top four projects, using only what is in the files.
Managers add project reviews, which are stored exactly as written; the AI never rewrites them.
Everything sits in one database file. A second small AI turns every piece of text into numbers that capture
meaning, so searching 'leadership' also finds 'led a team of five'.
A search mixes exact words, which score highest, with meaning matches, then turns the score into one to five
stars. Nothing is recomputed unless the files, the model or the prompt change, and all of it runs on our own machine."