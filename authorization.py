from enum import Enum
from typing import Iterable


class Permission(str, Enum):
    UPLOAD_DOCUMENT = "upload_document"
    GENERATE_SUMMARY = "generate_summary"
    VIEW_DOCUMENTS = "view_documents"
    DELETE_DOCUMENT = "delete_document"
    MANAGE_USERS = "manage_users"
    VIEW_AUDIT_LOGS = "view_audit_logs"


ROLE_PERMISSIONS = {
    "Admin": frozenset(Permission),
    "Lawyer": frozenset({
        Permission.UPLOAD_DOCUMENT,
        Permission.GENERATE_SUMMARY,
        Permission.VIEW_DOCUMENTS,
    }),
    "Viewer": frozenset({Permission.VIEW_DOCUMENTS}),
}


def has_permission(role: str, permission: Permission) -> bool:
    """Return whether a server-side role grants the requested permission.

    UI visibility is only a convenience; every protected callback calls this function.
    """
    return permission in ROLE_PERMISSIONS.get(role, frozenset())


def require_permission(role: str, permissions: Iterable[Permission]) -> bool:
    return all(has_permission(role, permission) for permission in permissions)
