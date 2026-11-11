"""Durable case review context, with a bounded excerpt for model prompts."""

import json
from collections.abc import Collection
from datetime import datetime, timezone
from pathlib import Path
from .case_paths import safe_case_path


class ReviewHistory:
    def __init__(self, case_dir: str, report_dir: str | None = None):
        self.case_directory = Path(case_dir)
        self.directory = Path(report_dir) if report_dir else self.case_directory / ".foamagent"
        self.path = self.directory / "review_history.jsonl"

    def append(self, event: str, **details) -> None:
        # Journaling must not create a missing case before the runner validates it.
        if not self.case_directory.is_dir():
            return
        path = safe_case_path(self.directory, "", self.path.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **details}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def context(self, max_bytes: int = 64_000, *, events: Collection[str] | None = None,
                stage: str | None = None) -> str:
        """Keep the full journal on disk; supply only its recent tail to the LLM."""
        path = safe_case_path(self.directory, "", self.path.name)
        if not path.exists():
            return ""
        with path.open("rb") as handle:
            handle.seek(0, 2)
            start = max(0, handle.tell() - max_bytes)
            handle.seek(start)
            excerpt = handle.read().decode("utf-8", errors="replace")
        if events is not None or stage is not None:
            selected = []
            for line in excerpt.splitlines():
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue  # A bounded excerpt may begin within an event.
                if ((events is None or entry.get("event") in events)
                        and (stage is None or entry.get("stage") == stage)):
                    selected.append(line)
            excerpt = "\n".join(selected)
        if start:
            excerpt = "[Earlier context omitted; excerpt may begin within an event.]\n" + excerpt
        return excerpt
