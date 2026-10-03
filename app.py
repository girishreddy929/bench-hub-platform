import os
from urllib.parse import quote

import pandas as pd
import streamlit as st

import src.ai.ai_engine as ai
from src.ai import models
import src.db.database as db
from src.search import search as hsearch
import src.ui.ui_theme as ui_theme
import streamlit.components.v1 as components
from src.services import jobs
from src.ui.import_ui import render_import_section
from src.search.skills import compact_skill, merge_skills

APP_TITLE = "Tredence Bench Hub"  # rename the app here

st.set_page_config(page_title=APP_TITLE, page_icon=ui_theme.page_icon(), layout="wide")

MAX_UPLOAD_MB = 10
SUMMARY_ON_FEEDBACK = os.getenv("SUMMARY_ON_FEEDBACK", "false").strip().lower() == "true"

RESUME_TYPES = ["pdf", "docx"]
SLIDER_TYPES = ["pptx", "docx", "pdf"]
MIME = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


@st.cache_resource
def init_app() -> bool:
    db.init_db()
    return True


init_app()
st.session_state.setdefault("nonce", 0)  # bumped to reset forms after saving


# ------------------------------- helpers ----------------------------------
def flash(kind: str, message: str) -> None:
    st.session_state.setdefault("flash", []).append((kind, message))


def show_flash() -> None:
    box = st.container()  # always present, so the tabs keep a stable position
    for kind, message in st.session_state.pop("flash", []):
        getattr(box, kind)(message)


def label(emp) -> str:
    return f"{emp.name} [{emp.employee_code}] ({emp.email})"


def fmt_dt(value) -> str:
    return value.strftime("%Y-%m-%d %H:%M") if value else ""

CONTACT_DOMAIN = os.getenv("CONTACT_ALLOWED_DOMAIN", "tredence.com").strip().lower()


def can_contact(emp) -> bool:
    """Contact buttons are live only for company email addresses."""
    return (emp.email or "").strip().lower().endswith("@" + CONTACT_DOMAIN)


def mail_link(emp) -> str:
    first = (emp.name or "").split()[0] if (emp.name or "").strip() else "there"
    subject = f"Bench resource inquiry: {emp.name} ({emp.employee_code})"
    body = (
        f"Hi {first},\r\n\r\n"
        "I came across your profile on Tredence Bench Hub and would like to "
        "discuss a possible project opportunity. Could we connect this week?\r\n\r\n"
        "Thanks"
    )
    return f"mailto:{emp.email}?subject={quote(subject)}&body={quote(body)}"


def teams_link(emp) -> str:
    return f"https://teams.microsoft.com/l/chat/0/0?users={quote(emp.email, safe='@')}"


def _noop() -> None:
    pass

def refresh_summary(employee_id: int) -> bool:
    """Regenerate and save the AI summary. Never raises; queues a message."""
    docs = db.get_documents(employee_id)
    feedback = db.get_feedback(employee_id)
    resume, slider = docs.get("resume"), docs.get("one_slider")
    try:
        with st.spinner("Generating summary with the local model (can take a minute)..."):
            result = ai.generate_summary(
                resume.extracted_text if resume else "",
                slider.extracted_text if slider else "",
                feedback,
            )
        db.save_summary(
            employee_id,
            result["organization_summary"],
            result["top_strengths"],
            result["skills"],
        )
    except ai.SummaryError as exc:
        flash("warning", f"Saved, but the summary was not updated: {exc}")
        return False
    flash("success", "AI summary updated.")
    return True

@st.fragment(run_every=3)
def job_status() -> None:
    """Sidebar: queues stale search vectors, shows running updates, reruns once when one finishes."""
    jobs.queue_stale_embeddings()  # throttled; keeps search vectors current
    ids = jobs.pending_ids()
    finished = st.session_state.get("jobs_seen", set()) - ids
    st.session_state["jobs_seen"] = ids
    if ids:
        st.info(f"Profiles updating: {len(ids)}")
    if finished:
        st.rerun()        

def handle_employee_save(selected_id, code, name, email, skills, status, resume_file, slider_file):
    uploads = [("resume", resume_file), ("one_slider", slider_file)]

    for _, f in uploads:
        if f is not None and f.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"{f.name} is larger than {MAX_UPLOAD_MB} MB.")
            return

    try:
        employee_id = db.save_employee(code, name, email, skills, status, employee_id=selected_id)
    except ValueError as exc:
        st.error(str(exc))
        return

    saved_any = False
    for doc_type, f in uploads:
        if f is None:
            continue
        data = f.getvalue()
        try:
            text = ai.extract_text(f.name, data)
        except Exception as exc:
            text = ""
            flash("warning", f"Could not read {f.name}: {exc}. Stored, but not used for the summary.")
        else:
            if not text:
                flash("warning", f"No text found in {f.name} (scanned or image-only?). Stored, but not used for the summary.")
        try:
            db.save_document(employee_id, doc_type, f.name, data, text)
            saved_any = True
        except ValueError as exc:
            flash("error", str(exc))

    flash("success", f"Saved {name.strip()} (ID {code.strip()}).")
    jobs.queue("embed", employee_id)  # skill or status edits become searchable right away
    if saved_any:
        jobs.queue_summary(employee_id)
        flash("info", "Summary is updating in the background.")
    st.session_state["pending_admin_select"] = employee_id
    st.session_state["nonce"] += 1
    st.rerun()


def handle_feedback_submit(employee_id, manager, project, rating, pros, cons, comments):
    try:
        replaced = db.add_feedback(employee_id, manager, project, rating, pros, cons, comments)
    except ValueError as exc:
        st.error(str(exc))
        return
    flash(
        "success",
        "Replaced this manager's earlier review for the same project." if replaced else "Feedback saved.",
    )
    if SUMMARY_ON_FEEDBACK:
        jobs.queue_summary(employee_id)  # the summary job queues the embedding afterwards
        flash("info", "Summary is updating in the background.")
    else:
        jobs.queue("embed", employee_id)  # keep search current, no AI summary
    st.session_state["nonce"] += 1
    st.rerun()


# --------------------------------- page -----------------------------------
ui_theme.apply_theme()
ui_theme.render_header(APP_TITLE)

installed = models.installed_models()


def _model_status(name: str, label: str) -> None:
    if installed is None:
        st.error("Ollama is not reachable. Start it (ollama serve).")
    elif models.is_installed(name, installed):
        st.success(f"{label} ready ({name})")
    else:
        st.error(f"Not installed. Run: ollama pull {name}")


def _pick_summary_model() -> None:
    models.set_summary_model(st.session_state["sel_summary_model"])
    jobs.clear_errors()


def _pick_embed_model() -> None:
    models.set_embed_model(st.session_state["sel_embed_model"])
    jobs.clear_errors()


with st.sidebar:
    st.header("Local AI")
    st.session_state["sel_summary_model"] = models.get_summary_model()
    st.selectbox(
        "AI summary model", models.summary_choices(),
        format_func=models.summary_label, key="sel_summary_model", on_change=_pick_summary_model,
    )
    _model_status(models.get_summary_model(), "Summary model")
    st.session_state["sel_embed_model"] = models.get_embed_model()
    st.selectbox(
        "Search model", models.embed_choices(),
        format_func=models.embed_label, key="sel_embed_model", on_change=_pick_embed_model,
    )
    _model_status(models.get_embed_model(), "Search model")
    st.caption("Models run on your machine. No data leaves it.")
    job_status()
show_flash()
# jobs.queue_stale_embeddings()  # keeps search vectors current (imports, HR skill edits, new reviews)


tab_admin, tab_dash = st.tabs(["Admin / HR Portal", "Manager Dashboard"])

# ------------------------------ Tab 1: Admin ------------------------------
with tab_admin:
    render_import_section(refresh_summary)
    st.divider()

    employees = db.get_all_employees()
    by_id = {e.id: e for e in employees}
    nonce = st.session_state["nonce"]

    if "pending_admin_select" in st.session_state:  # must run before the selectbox
        st.session_state["admin_emp_select"] = st.session_state.pop("pending_admin_select")

    st.subheader("Add or edit one employee")
    selected_id = st.selectbox(
        "Choose employee",
        options=[None] + list(by_id),
        format_func=lambda i: "New Employee" if i is None else label(by_id[i]),
        key="admin_emp_select",
    )
    current = by_id.get(selected_id)
    k = f"{selected_id}_{nonce}"

    with st.form(f"employee_form_{k}"):
        code = st.text_input("Employee ID", value=current.employee_code if current else "", key=f"code_{k}")
        name = st.text_input("Name", value=current.name if current else "", key=f"name_{k}")
        email = st.text_input("Email", value=current.email if current else "", key=f"email_{k}")
        skills = st.text_input(
            "Primary skills (comma-separated)",
            value=current.primary_skills if current else "",
            key=f"skills_{k}",
        )
        status_index = (
            db.STATUSES.index(current.availability_status)
            if current and current.availability_status in db.STATUSES
            else 0
        )
        status = st.selectbox("Availability status", db.STATUSES, index=status_index, key=f"status_{k}")
        resume_file = st.file_uploader("Resume (PDF or DOCX)", type=RESUME_TYPES, key=f"resume_{k}")
        slider_file = st.file_uploader("One-Slider (PPTX, DOCX or PDF)", type=SLIDER_TYPES, key=f"slider_{k}")
        submitted = st.form_submit_button("Save")

    if current:
        docs = db.get_documents(current.id)
        st.caption(
            "Current files: "
            + "; ".join(
                f"{t.replace('_', '-')}: {docs[t].file_name} ({fmt_dt(docs[t].uploaded_at)})"
                if t in docs
                else f"{t.replace('_', '-')}: none"
                for t in db.DOC_TYPES
            )
            + ". Uploading a new file replaces the old one."
        )

    if submitted:
        handle_employee_save(selected_id, code, name, email, skills, status, resume_file, slider_file)

    st.divider()
    st.subheader("Manager feedback")
    if not employees:
        st.info("Add an employee first.")
    else:
        fb_id = st.selectbox(
            "Employee",
            options=list(by_id),
            format_func=lambda i: label(by_id[i]),
            key="fb_emp_select",
        )
        with st.form(f"feedback_form_{nonce}"):
            manager = st.text_input("Manager name", key=f"fb_mgr_{nonce}")
            project = st.text_input("Project name", key=f"fb_proj_{nonce}")
            rating = st.slider("Rating", 1, 5, 3, key=f"fb_rating_{nonce}")
            pros = st.text_area("Pros", key=f"fb_pros_{nonce}")
            cons = st.text_area("Cons", key=f"fb_cons_{nonce}")
            comments = st.text_area("Comments", key=f"fb_comments_{nonce}")
            fb_submitted = st.form_submit_button("Submit feedback")
        if fb_submitted:
            handle_feedback_submit(fb_id, manager, project, rating, pros, cons, comments)


# --------------------------- Tab 2: Dashboard -----------------------------

SCROLL_JS = """
<script>
setTimeout(function () {
  const el = window.parent.document.getElementById('profile-top');
  if (el) el.scrollIntoView({behavior: 'smooth', block: 'start'});
}, 250);
</script>
"""

def render_details(emp) -> None:
    
    st.markdown('<div id="profile-top"></div>', unsafe_allow_html=True)
    st.divider()
    st.subheader(emp.name)
    st.caption(f"Employee ID {emp.employee_code}  |  {emp.email}  |  {emp.availability_status}")

    im_col, mail_col, _ = st.columns([1, 1, 5])
    if can_contact(emp):
        im_col.link_button("Send IM", teams_link(emp), use_container_width=True)
        mail_col.link_button("Send Mail", mail_link(emp), use_container_width=True)
    else:
        im_col.button("Send IM", key=f"im_{emp.id}", on_click=_noop, use_container_width=True)
        mail_col.button("Send Mail", key=f"mail_{emp.id}", on_click=_noop, use_container_width=True)
    
    busy = emp.id in jobs.pending_ids()
    failure = jobs.error_for(emp.id)
    if busy:
        st.info("Summary is being updated in the background. The previous summary is shown until it finishes.")
    elif failure:
        st.warning(f"Last background update failed: {failure}")
        
    summary = db.get_summary(emp.id)
    typed_skills = [s for s in emp.primary_skills.split(",") if s.strip()]

    if summary:
        st.markdown("#### Organization Summary")
        st.write(summary["organization_summary"])
        st.markdown("#### Strengths")
        for item in summary["top_strengths"]:
            st.markdown(f"- {item}")
        st.markdown("#### Employee Skills")
        all_skills = merge_skills(typed_skills, summary["skills"])
        st.markdown(" ".join(f"`{s}`" for s in all_skills) or "_None listed_")
        st.caption(f"Summary generated {fmt_dt(summary['generated_at'])} UTC")
    else:
        st.info("No summary yet. Upload a one-slider/resume or add feedback, or click Regenerate.")
        if typed_skills:
            st.markdown("#### Employee Skills")
            st.markdown(" ".join(f"`{s}`" for s in merge_skills(typed_skills)))

    if st.button("Regenerate summary", key=f"regen_{emp.id}", disabled=busy):
        jobs.queue_summary(emp.id)
        st.rerun()

    st.markdown("#### Manager Review")
    feedback = db.get_feedback(emp.id)
    if not feedback:
        st.caption("No manager feedback yet.")
    else:
        average = sum(f.rating for f in feedback) / len(feedback)
        st.metric("Average rating", f"{average:.1f} / 5", help=f"{len(feedback)} entries")
        for f in feedback:
            with st.expander(f"Manager Review: {f.manager_name}  -  {f.project_name or 'No project'}  ({f.rating}/5, {fmt_dt(f.created_at)})"):
                st.markdown(f"**Pros:** {f.pros or '-'}")
                st.markdown(f"**Cons:** {f.cons or '-'}")
                st.markdown(f"**Comments:** {f.comments or '-'}")

    st.markdown("#### Files")
    docs = db.get_documents(emp.id)
    columns = st.columns(2)
    for column, doc_type, button_label in (
        (columns[0], "resume", "Download Resume"),
        (columns[1], "one_slider", "Download One-Slider"),
    ):
        doc = docs.get(doc_type)
        with column:
            if doc:
                ext = os.path.splitext(doc.file_name)[1].lower()
                st.download_button(
                    button_label,
                    data=doc.file_data,
                    file_name=doc.file_name,
                    mime=MIME.get(ext, "application/octet-stream"),
                    key=f"dl_{doc_type}_{emp.id}",
                )
            else:
                st.button(button_label, disabled=True, key=f"dl_off_{doc_type}_{emp.id}")


def _view(employee_id: int) -> None:
    st.session_state["view_emp"] = employee_id
    st.session_state["scroll_to_profile"] = True


with tab_dash:
    all_employees = db.get_all_employees()
    if not all_employees:
        st.info("No employees yet. Add or import some in the Admin / HR Portal tab.")
    else:
        ai_skills = db.get_all_skills_map()
        skills_of = {
            e.id: merge_skills([s for s in e.primary_skills.split(",") if s.strip()], ai_skills.get(e.id, []))
            for e in all_employees
        }
        keys_of = {eid: {compact_skill(s) for s in sk} for eid, sk in skills_of.items()}

        skill_names = {}
        for sk in skills_of.values():
            for s in sk:
                skill_names.setdefault(compact_skill(s), s)
        skill_options = sorted(skill_names.values(), key=str.lower)

        f1, f2, f3, f4 = st.columns([3, 3, 2, 2])
        query = f1.text_input("Search (skills, projects, name or ID)", key="dash_search").strip()
        skill_filter = f2.multiselect("Skills (must have all)", skill_options, key="dash_skills")
        status_filter = f3.multiselect("Status", db.STATUSES, key="dash_status")
        min_stars = f4.slider("Minimum match (stars)", 1, 5, 2, key="dash_min_stars", disabled=not query)
        wanted = {compact_skill(s) for s in skill_filter}

        def allowed(e) -> bool:
            if wanted and not wanted <= keys_of[e.id]:
                return False
            if status_filter and e.availability_status not in status_filter:
                return False
            return True

        candidates = [e for e in all_employees if allowed(e)]
        hit_by_id = {}
        if query:
            outcome = hsearch.run_search(query, candidates)
            for note in outcome.notes:
                st.warning(note)
            if outcome.concern_mode:
                st.caption("Showing manager concerns that match your search.")
            hit_by_id = {h.employee_id: h for h in outcome.hits if h.stars >= min_stars}
            candidate_by_id = {e.id: e for e in candidates}
            filtered = [candidate_by_id[h.employee_id] for h in outcome.hits if h.employee_id in hit_by_id]
        else:
            filtered = candidates
        st.caption(f"{len(filtered)} of {len(all_employees)} employees")

        if not filtered:
            st.warning("No employees match the filters.")
        else:
            widths = [1.2, 2, 3, 4, 1.5, 1.3, 1] if query else [1.2, 2, 3, 4, 1.5, 1]
            titles = ["Employee ID", "Name", "Email", "Skills", "Status"] + (["Match"] if query else []) + [""]
            for column, title in zip(st.columns(widths), titles):
                column.markdown(f"**{title}**")
            viewing = st.session_state.get("view_emp")
            with st.container(height=380):
                for e in filtered:
                    cells = st.columns(widths)
                    skills_text = ", ".join(skills_of[e.id])
                    cells[0].write(e.employee_code)
                    cells[1].write(e.name)
                    cells[2].write(e.email)
                    cells[3].write(skills_text if len(skills_text) <= 70 else skills_text[:67] + "...")
                    cells[4].markdown(ui_theme.status_chip(e.availability_status), unsafe_allow_html=True)
                    if query:
                        cells[5].write(hsearch.star_text(hit_by_id[e.id].stars))
                    cells[-1].button(
                        "View", key=f"view_{e.id}", on_click=_view, args=(e.id,),
                        type="primary" if viewing == e.id else "secondary",
                    )
        
        chosen = next((e for e in all_employees if e.id == st.session_state.get("view_emp")), None)
        if chosen:
            render_details(chosen)
            if st.session_state.pop("scroll_to_profile", False):
                components.html(SCROLL_JS, height=0)
        
ui_theme.render_footer()