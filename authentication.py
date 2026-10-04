import secrets
from dataclasses import dataclass
from typing import Dict, Optional

import bcrypt

from audit import SecurityAudit


@dataclass(frozen=True)
class User:
    username: str
    password_hash: bytes
    role: str


class Authenticator:
    def __init__(self, audit: SecurityAudit):
        self.audit = audit
        self.users: Dict[str, User] = {
            "admin": User("admin", bcrypt.hashpw(b"Admin@123", bcrypt.gensalt()), "Admin"),
            "lawyer": User("lawyer", bcrypt.hashpw(b"Lawyer@123", bcrypt.gensalt()), "Lawyer"),
            "viewer": User("viewer", bcrypt.hashpw(b"Viewer@123", bcrypt.gensalt()), "Viewer"),
        }
        self.sessions: Dict[str, str] = {}

    def login(self, username: str, password: str, session_id: str) -> Optional[User]:
        user = self.users.get((username or "").strip().lower())
        valid = (
            user is not None
            and isinstance(password, str)
            and bool(session_id)
            and bcrypt.checkpw(password.encode("utf-8"), user.password_hash)
        )
        if not valid:
            self.audit.record(username, "login", "FAILED")
            return None
        # Only the server stores the session-to-user mapping; the client never supplies a role.
        self.sessions[session_id] = user.username
        self.audit.record(user.username, "login", "SUCCESS")
        return user

    def current_user(self, session_id: str) -> Optional[User]:
        username = self.sessions.get(session_id)
        return self.users.get(username) if username else None

    def logout(self, session_id: str) -> None:
        user = self.current_user(session_id)
        if user:
            self.audit.record(user.username, "logout", "SUCCESS")
        self.sessions.pop(session_id, None)

    def add_user(self, actor: User, username: str, password: str, role: str) -> str:
        if actor.role != "Admin":
            self.audit.record(actor.username, "user_management", "ACCESS DENIED")
            return "Access denied. Admin role required."
        username = (username or "").strip().lower()
        if not username or not password or role not in {"Admin", "Lawyer", "Viewer"}:
            return "Username, password, and a valid role are required."
        if username in self.users:
            return "That username already exists."
        self.users[username] = User(username, bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()), role)
        self.audit.record(actor.username, "user_management", "SUCCESS")
        return f"User {username} created."

    def remove_user(self, actor: User, username: str) -> str:
        if actor.role != "Admin":
            self.audit.record(actor.username, "user_management", "ACCESS DENIED")
            return "Access denied. Admin role required."
        username = (username or "").strip().lower()
        if username == actor.username:
            return "You cannot delete your own account."
        if username not in self.users:
            return "User not found."
        del self.users[username]
        for session_id, active_user in list(self.sessions.items()):
            if active_user == username:
                del self.sessions[session_id]
        self.audit.record(actor.username, "user_management", "SUCCESS")
        return f"User {username} deleted."

    def user_listing(self) -> str:
        return "\n".join(f"{user.username} ({user.role})" for user in self.users.values())
