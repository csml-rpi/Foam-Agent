"""Optional, per-run JSONL capture for successful model calls."""

from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import RLock


class DatasetRecorder:
    def __init__(self, path: str = "", case_id: str = ""):
        self.path = path
        self.case_id = case_id
        self._pending = []
        self._deferred = False
        self._lock = RLock()

    def record(self, **record):
        if not self.path:
            return
        record = {
            "case_id": self.case_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **record,
        }
        with self._lock:
            if self._deferred:
                self._pending.append(record)
            else:
                self._write([record])

    def _write(self, records):
        if not records:
            return
        path = Path(self.path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    @contextmanager
    def defer(self):
        """Flush planning calls only after output ownership checks succeed."""
        with self._lock:
            self._deferred = True
        try:
            yield
        except BaseException:
            with self._lock:
                self._pending.clear()
            raise
        else:
            with self._lock:
                self._write(self._pending)
                self._pending.clear()
        finally:
            self._deferred = False


def defer_dataset_logging(client):
    recorder = getattr(client, "dataset_recorder", None)
    return recorder.defer() if recorder is not None and recorder.path else nullcontext()


def invoke_logged(client, *args, log_context=None, **kwargs):
    # Keep existing injected clients usable when capture is disabled.
    recorder = getattr(client, "dataset_recorder", None)
    if recorder is not None and recorder.path and log_context:
        kwargs["log_context"] = log_context
    return client.invoke(*args, **kwargs)
