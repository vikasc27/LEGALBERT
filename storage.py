import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


@dataclass
class Document:
    document_id: str
    owner: str
    filename: str
    stored_path: Path
    summary: str = ""
    created_at: str = ""


class DocumentStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.documents: Dict[str, Document] = {}

    def add(self, owner: str, filename: str, source_path: Path) -> Document:
        document_id = secrets.token_urlsafe(10)
        safe_name = Path(filename).name
        destination = self.root / f"{document_id}_{safe_name}"
        # Copy into application-owned storage before PDF parsing so the temporary upload
        # supplied by the UI can disappear without interrupting processing.
        destination.write_bytes(source_path.read_bytes())
        document = Document(document_id, owner, safe_name, destination,
                            created_at=datetime.now(timezone.utc).isoformat())
        self.documents[document_id] = document
        return document

    def get(self, document_id: str) -> Optional[Document]:
        return self.documents.get(document_id)

    def delete(self, document_id: str) -> bool:
        document = self.documents.pop(document_id, None)
        if not document:
            return False
        document.stored_path.unlink(missing_ok=True)
        return True

    def visible_to(self, username: str, role: str):
        if role == "Admin":
            return list(self.documents.values())
        return [document for document in self.documents.values() if document.owner == username]
