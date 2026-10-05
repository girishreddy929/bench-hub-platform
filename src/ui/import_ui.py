"""Streamlit UI for the folder import: scan, select, review dialog, upload."""

import os

import pandas as pd
import streamlit as st

import src.db.database as db
import src.services.importer as imp
from src.services import jobs, timing


# ---------------------------- state helpers -------------------------------
def _drafts() -> list:
    return st.session_state.setdefault("drafts", [])


def _find(uid):
    return next((d for d in _drafts() if d.uid == uid), None)


def _clear_edit_keys(uid) -> None:
    for name in ("code", "name", "email", "skills", "status", "designation", "certs"):
        st.session_state.pop(f"ed_{name}_{uid}", None)


def _start_edit(uid) -> None:
    _clear_edit_keys(uid)
    st.session_state["editing"] = uid


def _cancel_edit() -> None:
    uid = st.session_state.get("editing")
    if uid:
        _clear_edit_keys(uid)
    st.session_state["editing"] = None


def _delete(uid) -> None:
    st.session_state["drafts"] = [d for d in _drafts() if d.uid != uid]
    _clear_edit_keys(uid)
    if st.session_state.get("editing") == uid:
        st.session_state["editing"] = None


def _save_edit(uid) -> None:
    ss = st.session_state
    d = _find(uid)
    new_code = ss[f"ed_code_{uid}"].strip()
    if new_code and any(o.code == new_code and o.uid != uid for o in _drafts()):
        ss["edit_error"] = f"Employee ID {new_code} is already used by another profile in this list."
        return
    d.code = new_code
    d.name = ss[f"ed_name_{uid}"].strip()
    d.email = ss[f"ed_email_{uid}"].strip()
    d.skills = ss[f"ed_skills_{uid}"].strip()
    d.status = ss[f"ed_status_{uid}"]
    d.designation = ss[f"ed_designation_{uid}"].strip()
    d.certs = ss[f"ed_certs_{uid}"].strip()
    d.exists = bool(new_code) and db.get_employee_by_code(new_code) is not None
    d.error = ""
    _clear_edit_keys(uid)
    ss["editing"] = None


def _status(d) -> str:
    if d.error:
        label = "Error"
    elif imp.problems(d):
        label = "Fix needed"
    elif d.notes:
        label = "Check"
    else:
        label = "OK"
    return label + (" (updates existing)" if d.exists else "")


# ------------------------------ dialog ------------------------------------
@st.dialog("Review imported profiles", width="large")
def review_dialog(refresh_summary):
    drafts = _drafts()
    if not drafts:
        st.success("Nothing left to review.")
        return

    st.caption("Check the mapping. Pencil = correct it (profiles without an Employee ID need one typed in). Bin = do not upload.")
    error = st.session_state.pop("edit_error", None)
    if error:
        st.error(error)

    for d in list(drafts):
        with st.container(border=True):
            c = st.columns([1.6, 2.6, 3.6, 0.6, 0.6])
            c[0].markdown(f"**{_status(d)}**")
            c[1].markdown(
                f"**{d.name or '(no name)'}**  \nID {d.code or '(missing)'}"
                + (f"  \n{d.designation}" if d.designation else "")
            )
            c[2].markdown(f"{d.email or '(no email)'}  \n{d.skills or '(no skills)'}")
            c[3].button("✏️", key=f"edit_{d.uid}", on_click=_start_edit, args=(d.uid,), help="Edit")
            c[4].button("🗑️", key=f"del_{d.uid}", on_click=_delete, args=(d.uid,), help="Do not upload")
            st.caption(
                f"Resume: {os.path.basename(d.resume_path) if d.resume_path else 'none'}  |  "
                f"One-slider: {os.path.basename(d.slider_path) if d.slider_path else 'none'}"
            )
            cert_count = len([ln for ln in d.certs.splitlines() if ln.strip()])
            st.caption(f"Certifications: {cert_count}  |  Projects found: {len(d.projects)}")
            for note in imp.problems(d) + d.notes + ([d.error] if d.error else []):
                st.caption(f"⚠ {note}")

            if st.session_state.get("editing") == d.uid:
                with st.form(f"edit_form_{d.uid}"):
                    st.text_input("Employee ID", value=d.code, key=f"ed_code_{d.uid}")
                    st.text_input("Name", value=d.name, key=f"ed_name_{d.uid}")
                    st.text_input("Email", value=d.email, key=f"ed_email_{d.uid}")
                    st.text_input("Skills (comma-separated)", value=d.skills, key=f"ed_skills_{d.uid}")
                    st.text_input("Designation", value=d.designation, key=f"ed_designation_{d.uid}")
                    st.text_area(
                        "Certifications (one per line: Name | Issuer | Year)",
                        value=d.certs, key=f"ed_certs_{d.uid}", height=110,
                    )
                    st.selectbox(
                        "Status", db.STATUSES,
                        index=db.STATUSES.index(d.status) if d.status in db.STATUSES else 0,
                        key=f"ed_status_{d.uid}",
                    )
                    b1, b2 = st.columns(2)
                    b1.form_submit_button("Save changes", on_click=_save_edit, args=(d.uid,))
                    b2.form_submit_button("Cancel", on_click=_cancel_edit)

    ready = [d for d in drafts if not imp.problems(d)]
    st.divider()
    if st.button(f"Upload {len(ready)} profile(s) to database", type="primary",
                 disabled=not ready, key="do_upload"):
        bar = st.progress(0.0)
        scan = st.session_state.get("scan")
        done = failed = 0
        with timing.span(f"IMPORT UPLOAD {len(ready)} profiles"):
            for i, d in enumerate(ready, 1):
                try:
                    emp_id, regen = imp.upload_draft(d)
                    if regen:
                        refresh_summary(emp_id)
                    drafts.remove(d)
                    if scan:
                        scan["groups"].pop(d.uid, None)  # uploaded rows leave the scan table
                    done += 1
                except (ValueError, OSError) as exc:
                    d.error = str(exc)
                    failed += 1
                bar.progress(i / len(ready))

        if scan is not None:
            if scan["groups"]:
                st.session_state["scan_nonce"] = st.session_state.get("scan_nonce", 0) + 1
            else:
                st.session_state.pop("scan", None)
        st.session_state.setdefault("flash", []).append(
            ("success", f"Uploaded {done} profile(s)." + (f" {failed} failed, see the review window." if failed else ""))
        )
        if failed:
            st.rerun(scope="fragment")  # stay open and show the errors
        st.rerun()  # close the dialog


# ------------------------------ page section ------------------------------

@st.fragment(run_every=3)
def _read_progress() -> None:
    """Shows background reading progress; re-runs the page once when a profile finishes."""
    counts = jobs.read_counts()
    waiting = counts["queued"] + counts["running"]
    if waiting:
        st.info(f"Reading profiles with AI: {waiting} left, {counts['done']} ready. You can keep working.")
    finished = counts["done"] + counts["failed"]
    if finished != st.session_state.get("reads_finished_seen", 0):
        st.session_state["reads_finished_seen"] = finished
        st.rerun()
        
def render_import_section(refresh_summary) -> None:
    st.subheader("Import profiles from a folder")
    st.caption(
        "Files are matched by Employee ID in the file name, e.g. "
        "`1226_girish reddy resume.pdf` and `girish one slider 1226.pptx`. "
        "Files without an ID are listed too, but need an ID typed in before upload."
    )
    folder = st.text_input("Folder path", value=os.getenv("IMPORT_FOLDER", ""), key="import_folder")

    if st.button("Scan folder", key="scan_btn"):
        try:
            groups, skipped = imp.scan_folder(folder.strip().strip('"'))
        except ValueError as exc:
            st.session_state.pop("scan", None)
            st.error(str(exc))
        else:
            st.session_state["scan"] = {"groups": groups, "skipped": skipped}
            st.session_state["scan_nonce"] = st.session_state.get("scan_nonce", 0) + 1

    scan = st.session_state.get("scan")
    if scan:
        groups = scan["groups"]
        if scan["skipped"]:
            with st.expander(f"{len(scan['skipped'])} file(s) skipped"):
                for message in scan["skipped"]:
                    st.caption(message)
        if not groups:
            st.warning("No PDF/DOCX/PPTX files found in that folder.")
        else:
            keys = list(groups)
            select_all = st.checkbox("Select all", value=True, key="imp_select_all")
            in_db = {e.employee_code for e in db.get_all_employees()}
            table = pd.DataFrame(
                [
                    {
                        "Select": select_all,
                        "Employee ID": groups[k]["code"] or "(missing)",
                        "Name (from file)": imp.display_name(groups[k]),
                        "Resume": os.path.basename(groups[k]["resume"]) if groups[k]["resume"] else "-",
                        "One-slider": os.path.basename(groups[k]["one_slider"]) if groups[k]["one_slider"] else "-",
                        "In database": "Yes" if groups[k]["code"] in in_db else "No",
                    }
                    for k in keys
                ]
            )
            edited = st.data_editor(
                table, hide_index=True, width="stretch",
                disabled=[c for c in table.columns if c != "Select"],
                key=f"imp_table_{select_all}_{st.session_state.get('scan_nonce', 0)}",
            )
            chosen = [keys[i] for i in edited.index[edited["Select"].to_numpy()]]
            if st.button(f"Read {len(chosen)} selected profile(s) with AI", type="primary",
                         disabled=not chosen, key="read_btn"):
                started = sum(1 for key in chosen if jobs.queue_read(key, groups[key]))
                skipped = len(chosen) - started
                st.session_state.setdefault("flash", []).append(
                    (
                        "info",
                        f"Reading {started} profile(s) in the background."
                        + (f" {skipped} already queued or waiting for review." if skipped else ""),
                    )
                )
                st.rerun()

    _read_progress()
    counts = jobs.read_counts()
    if counts["done"] or counts["failed"]:
        if st.button(f"Review {counts['done']} read profile(s)", type="primary", key="review_btn"):
            new, failed = jobs.take_reads()
            have = {d.uid for d in _drafts()}
            st.session_state["drafts"] = _drafts() + [d for d in new if d.uid not in have]
            st.session_state["editing"] = None
            for uid, message in failed.items():
                st.error(f"Could not read {uid}: {message}")
            review_dialog(refresh_summary)
    if st.session_state.get("drafts") and st.button("Reopen review window", key="reopen_btn"):
        review_dialog(refresh_summary)