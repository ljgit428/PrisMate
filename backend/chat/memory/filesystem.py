"""Filesystem backends for the local ``list_memory_files`` / ``read_memory_file`` tools.

The chat pipeline and the character-draft pipeline share the same tool specs but
read from different sources:

- ``CharacterMemoryFilesystem`` reads the persisted Memory Explorer VFS for a
  saved ``Character`` (schema/wiki/raw layers, knowledge assets, transcripts).
- ``StagedUploadMemoryFilesystem`` reads the files the user just uploaded while
  creating a character, *before* the character row exists.

Both expose the same two methods so ``tasks._execute_local_memory_tool`` stays
backend-agnostic.
"""
from __future__ import annotations

import os
from typing import Any

from ..models import AttachmentKind
from ..soul import list_memory_explorer_path, read_memory_explorer_file


class CharacterMemoryFilesystem:
    """Wraps the saved-character Memory Explorer VFS."""

    def __init__(self, character) -> None:
        self.character = character

    def list_memory_files(self, path_prefix: str = "", recursive: bool = False, max_entries: int = 40) -> dict:
        return list_memory_explorer_path(
            self.character,
            path_prefix=path_prefix,
            recursive=recursive,
            max_entries=max_entries,
        )

    def read_memory_file(self, path: str, max_chars: int = 6000) -> dict:
        return read_memory_explorer_file(self.character, path=path, max_chars=max_chars)


class StagedUploadMemoryFilesystem:
    """Lists/reads freshly uploaded character files (pre-save) as a flat tree.

    Each upload is a dict::

        {
            "name": "dialogue.txt",
            "kind": "text" | "image",
            "mime_type": "text/plain",
            "content": "…",          # text content for text files, '' otherwise
            "file_url": "…",
        }
    """

    UPLOAD_ROOT = "raw/character_setup/uploads"

    def __init__(self, uploads: list[dict[str, Any]]) -> None:
        self.uploads = list(uploads or [])
        self._by_path: dict[str, dict[str, Any]] = {}
        for index, upload in enumerate(self.uploads):
            name = self._safe_name(upload.get("name") or f"upload-{index + 1}")
            upload["_path"] = f"{self.UPLOAD_ROOT}/{name}"
            self._by_path[upload["_path"]] = upload

    @staticmethod
    def _safe_name(name: str) -> str:
        normalized = os.path.basename((name or "").strip()) or "uploaded-file"
        return normalized.replace("\\", "_").replace("/", "_")

    def _entry(self, upload: dict[str, Any]) -> dict[str, Any]:
        kind = upload.get("kind") or AttachmentKind.TEXT
        content = upload.get("content") or ""
        return {
            "path": upload["_path"],
            "entry_type": "file",
            "layer": "raw",
            "title": upload.get("name") or os.path.basename(upload["_path"]),
            "kind": kind,
            "read_hint": "Original file uploaded while creating this character.",
            "is_locked": True,
            "can_user_edit": False,
            "can_auto_update": False,
            "updated_at": "",
            "manageable": False,
            "asset_id": None,
            "preview_kind": "image" if kind == AttachmentKind.IMAGE else "text",
            "size_hint": len(content),
        }

    def list_memory_files(self, path_prefix: str = "", recursive: bool = False, max_entries: int = 40) -> dict:
        try:
            safe_max_entries = max(1, min(int(max_entries or 40), 200))
        except (TypeError, ValueError):
            safe_max_entries = 40

        normalized_prefix = (path_prefix or "").strip().strip("/")
        entries = [self._entry(upload) for upload in self.uploads]
        if normalized_prefix:
            entries = [
                entry for entry in entries
                if entry["path"] == normalized_prefix or entry["path"].startswith(f"{normalized_prefix}/")
            ]

        return {
            "path_prefix": normalized_prefix or "/",
            "entries": entries[:safe_max_entries],
            "error": "",
            "truncated": len(entries) > safe_max_entries,
        }

    def read_memory_file(self, path: str, max_chars: int = 6000) -> dict:
        try:
            safe_max_chars = max(200, min(int(max_chars or 6000), 12000))
        except (TypeError, ValueError):
            safe_max_chars = 6000

        normalized_path = (path or "").strip().strip("/")
        upload = self._by_path.get(normalized_path)
        if upload is None:
            return {"path": normalized_path, "error": "File not found in staged uploads."}

        kind = upload.get("kind") or AttachmentKind.TEXT
        content = upload.get("content") or ""
        truncated = len(content) > safe_max_chars
        return {
            "path": normalized_path,
            "layer": "raw",
            "title": upload.get("name") or os.path.basename(normalized_path),
            "kind": kind,
            "read_hint": "Original file uploaded while creating this character.",
            "content": content[:safe_max_chars],
            "truncated": truncated,
            "manageable": False,
            "asset_id": None,
            "preview_kind": "image" if kind == AttachmentKind.IMAGE else "text",
            "file_url": upload.get("file_url", ""),
            "mime_type": upload.get("mime_type", ""),
        }
