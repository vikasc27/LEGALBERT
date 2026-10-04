from pathlib import Path

from audit import SecurityAudit
from authentication import Authenticator
from authorization import Permission, has_permission


def test_roles_have_least_privilege(tmp_path: Path):
    audit = SecurityAudit(tmp_path / "security.log")
    auth = Authenticator(audit)
    assert auth.login("admin", "Admin@123", "admin-session").role == "Admin"
    assert auth.login("lawyer", "Lawyer@123", "lawyer-session").role == "Lawyer"
    assert auth.login("viewer", "Viewer@123", "viewer-session").role == "Viewer"
    assert has_permission("Admin", Permission.DELETE_DOCUMENT)
    assert has_permission("Lawyer", Permission.GENERATE_SUMMARY)
    assert not has_permission("Lawyer", Permission.MANAGE_USERS)
    assert not has_permission("Viewer", Permission.UPLOAD_DOCUMENT)
    assert not has_permission("Viewer", Permission.DELETE_DOCUMENT)


def test_invalid_password_does_not_create_session(tmp_path: Path):
    audit = SecurityAudit(tmp_path / "security.log")
    auth = Authenticator(audit)
    assert auth.login("admin", "wrong", "session") is None
    assert auth.current_user("session") is None
    assert "status=FAILED" in (tmp_path / "security.log").read_text(encoding="utf-8")


def test_admin_only_user_management(tmp_path: Path):
    audit = SecurityAudit(tmp_path / "security.log")
    auth = Authenticator(audit)
    lawyer = auth.login("lawyer", "Lawyer@123", "lawyer-session")
    admin = auth.login("admin", "Admin@123", "admin-session")
    assert lawyer is not None and admin is not None
    assert "Access denied" in auth.add_user(lawyer, "newviewer", "Password@123", "Viewer")
    assert "created" in auth.add_user(admin, "newviewer", "Password@123", "Viewer")
    assert auth.login("newviewer", "Password@123", "new-session") is not None
    assert "ACCESS DENIED" in (tmp_path / "security.log").read_text(encoding="utf-8")


def test_logout_removes_server_side_session(tmp_path: Path):
    audit = SecurityAudit(tmp_path / "security.log")
    auth = Authenticator(audit)
    assert auth.login("viewer", "Viewer@123", "session") is not None
    auth.logout("session")
    assert auth.current_user("session") is None
