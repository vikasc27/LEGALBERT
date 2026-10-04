import logging
from pathlib import Path


LOG_PATH = Path(__file__).with_name("security.log")


class SecurityAudit:
    def __init__(self, log_path: Path = LOG_PATH):
        self.log_path = Path(log_path)
        self.logger = logging.getLogger(f"legalbert.security.{self.log_path.resolve()}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
            self.logger.addHandler(handler)

    def record(self, username: str, action: str, status: str) -> None:
        safe_user = username or "anonymous"
        self.logger.info("username=%s | action=%s | status=%s", safe_user, action, status)

    def read(self) -> str:
        try:
            return self.log_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return "No security events recorded yet."
