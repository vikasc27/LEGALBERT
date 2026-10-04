import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr

from audit import SecurityAudit
from authentication import Authenticator, User
from authorization import Permission, has_permission
from storage import DocumentStore
from summarizer import LegalSummarizer

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
BASE_DIR = Path(__file__).parent

audit = SecurityAudit()
authenticator = Authenticator(audit)
document_store = DocumentStore(BASE_DIR / "data")
summarizer = LegalSummarizer()


def _session_user(request: Optional[gr.Request]) -> Optional[User]:
    session_id = getattr(request, "session_hash", None) if request else None
    return authenticator.current_user(session_id) if session_id else None


def _authorized(request: Optional[gr.Request], permission: Permission, action: str) -> Optional[User]:
    # Resolve identity from the server-side session, then enforce permission before
    # touching uploaded files or invoking the LegalBERT summarizer.
    user = _session_user(request)
    if user is None:
        audit.record("anonymous", action, "ACCESS DENIED")
        return None
    if not has_permission(user.role, permission):
        audit.record(user.username, action, "ACCESS DENIED")
        return None
    return user


def _denied(action: str) -> str:
    return f"Access denied: permission required for {action}."


def login(username: str, password: str, request: gr.Request):
    session_id = getattr(request, "session_hash", "")
    user = authenticator.login(username, password, session_id)
    if not user:
        return (
            "Login failed. Check your username and password.",
            gr.update(visible=True), gr.update(visible=False), "", "", gr.update(visible=True),
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
        )
    return (
        f"Logged in as {user.username} ({user.role}).",
        gr.update(visible=False), gr.update(visible=True), user.role, f"## {user.role} dashboard", gr.update(visible=False),
        gr.update(visible=user.role == "Admin"), gr.update(visible=user.role in {"Admin", "Lawyer"}),
        gr.update(visible=user.role == "Admin"),
    )


def logout(request: gr.Request):
    authenticator.logout(getattr(request, "session_hash", ""))
    return (
        "Logged out.", gr.update(visible=True), gr.update(visible=False), "", "",
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
    )


def validate_pdf(pdf_path: str) -> Tuple[bool, str]:
    if not pdf_path:
        return False, "Please select a PDF file."
    path = Path(pdf_path)
    if path.suffix.lower() != ".pdf":
        return False, "Only PDF files are accepted."
    if not path.is_file():
        return False, "The uploaded file could not be found."
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        return False, "The PDF exceeds the 20 MB upload limit."
    with path.open("rb") as stream:
        if stream.read(5) != b"%PDF-":
            return False, "The file signature is not a valid PDF signature."
    return True, ""


def summarize_document(pdf_path: str, request: gr.Request) -> str:
    user = _authorized(request, Permission.UPLOAD_DOCUMENT, "document_upload")
    if user is None:
        return _denied("document upload")
    if not has_permission(user.role, Permission.GENERATE_SUMMARY):
        audit.record(user.username, "summary_generation", "ACCESS DENIED")
        return _denied("summary generation")
    valid, error = validate_pdf(pdf_path)
    if not valid:
        audit.record(user.username, "document_upload", "FAILED")
        return error
    try:
        # DocumentStore copies the validated upload into private application storage.
        document = document_store.add(user.username, Path(pdf_path).name, Path(pdf_path))
        audit.record(user.username, "document_upload", "SUCCESS")
        document.summary = summarizer.summarize_pdf(str(document.stored_path))
        audit.record(user.username, "summary_generation", "SUCCESS")
        return f"Document ID: {document.document_id}\n\n{document.summary}"
    except Exception as error:
        audit.record(user.username, "summary_generation", "FAILED")
        return f"Unable to process the PDF: {error}"


def list_documents(request: gr.Request) -> str:
    user = _authorized(request, Permission.VIEW_DOCUMENTS, "document_view")
    if user is None:
        return _denied("document viewing")
    documents = document_store.visible_to(user.username, user.role)
    if not documents:
        return "No authorized documents are available."
    return "\n\n".join(
        f"ID: {document.document_id}\nOwner: {document.owner}\nFile: {document.filename}\n\n{document.summary}"
        for document in documents
    )


def delete_document(document_id: str, request: gr.Request) -> str:
    user = _authorized(request, Permission.DELETE_DOCUMENT, "document_deletion")
    if user is None:
        return _denied("document deletion")
    if document_store.delete((document_id or "").strip()):
        audit.record(user.username, "document_deletion", "SUCCESS")
        return "Document deleted."
    audit.record(user.username, "document_deletion", "FAILED")
    return "Document not found."


def add_user(username: str, password: str, role: str, request: gr.Request) -> str:
    user = _authorized(request, Permission.MANAGE_USERS, "user_management")
    if user is None:
        return _denied("user management")
    return authenticator.add_user(user, username, password, role)


def remove_user(username: str, request: gr.Request) -> str:
    user = _authorized(request, Permission.MANAGE_USERS, "user_management")
    if user is None:
        return _denied("user management")
    return authenticator.remove_user(user, username)


def get_users(request: gr.Request) -> str:
    user = _authorized(request, Permission.MANAGE_USERS, "user_listing")
    if user is None:
        return _denied("user management")
    return authenticator.user_listing()


def get_audit_log(request: gr.Request) -> str:
    user = _authorized(request, Permission.VIEW_AUDIT_LOGS, "audit_log_view")
    if user is None:
        return _denied("audit log viewing")
    return audit.read()


def create_interface():
    with gr.Blocks(title="Secure Legal Document Summarizer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Secure Legal Document Summarizer\nLegalBERT summaries protected by authentication and RBAC.")
        message = gr.Markdown()
        role_state = gr.State("")
        with gr.Group(visible=True) as login_panel:
            username = gr.Textbox(label="Username")
            password = gr.Textbox(label="Password", type="password")
            login_button = gr.Button("Log in", variant="primary")
        with gr.Group(visible=False) as dashboard:
            with gr.Row():
                role_label = gr.Markdown()
                logout_button = gr.Button("Log out")
            with gr.Group(visible=False) as admin_panel:
                gr.Markdown("## Admin dashboard")
                with gr.Row():
                    new_username = gr.Textbox(label="New username")
                    new_password = gr.Textbox(label="New password", type="password")
                    new_role = gr.Dropdown(["Admin", "Lawyer", "Viewer"], label="Role", value="Viewer")
                with gr.Row():
                    add_button = gr.Button("Create user")
                    remove_username = gr.Textbox(label="Username to delete")
                    remove_button = gr.Button("Delete user")
                user_result = gr.Textbox(label="User management result")
                users_button = gr.Button("Refresh user list")
                users_output = gr.Textbox(label="Users", lines=5)
                logs_button = gr.Button("View security audit log")
                logs_output = gr.Textbox(label="Security audit log", lines=10)
            with gr.Group(visible=False) as upload_panel:
                gr.Markdown("## LegalBERT summarization")
                pdf_input = gr.File(label="Upload legal PDF", file_types=[".pdf"], type="filepath")
                analyze_button = gr.Button("Analyze document", variant="primary")
                summary_output = gr.Textbox(label="Summary", lines=24)
            with gr.Group(visible=True):
                gr.Markdown("## Authorized summaries")
                refresh_button = gr.Button("Refresh documents")
                documents_output = gr.Textbox(label="Documents and summaries", lines=18)
            with gr.Group(visible=False) as delete_panel:
                gr.Markdown("## Document management")
                delete_id = gr.Textbox(label="Document ID")
                delete_button = gr.Button("Delete document")
                delete_result = gr.Textbox(label="Deletion result")

        login_outputs = [message, login_panel, dashboard, role_state, role_label, admin_panel, upload_panel, delete_panel]
        login_button.click(login, [username, password], login_outputs)
        logout_button.click(logout, outputs=login_outputs)
        analyze_button.click(summarize_document, pdf_input, summary_output)
        refresh_button.click(list_documents, outputs=documents_output)
        delete_button.click(delete_document, delete_id, delete_result)
        add_button.click(add_user, [new_username, new_password, new_role], user_result)
        remove_button.click(remove_user, remove_username, user_result)
        users_button.click(get_users, outputs=users_output)
        logs_button.click(get_audit_log, outputs=logs_output)
    return demo


if __name__ == "__main__":
    create_interface().launch(server_name="127.0.0.1", server_port=int(os.getenv("PORT", "7860")))
