"""Run blocking tool work without occupying the MCP event loop."""

import asyncio
from contextlib import nullcontext
from pathlib import Path
from threading import Lock
from weakref import WeakValueDictionary

from logger import logging_context, setup_logging
from services.output_safety import output_is_owned


_case_locks = WeakValueDictionary()
_registry_lock = Lock()
_active_tools = set()


def _case_lock(case_dir):
    path = Path(case_dir).expanduser().resolve()
    with _registry_lock:
        # An owned ancestor can clear all nested outputs. Keep its live lock
        # identity even while overwrite temporarily removes the marker.
        owner = next((parent for parent in reversed((path, *path.parents))
                      if parent in _case_locks or output_is_owned(parent)), path)
        lock = _case_locks.get(owner)
        if lock is None:
            lock = asyncio.Lock()
            _case_locks[owner] = lock
        return lock


async def run_tool(worker, request, ctx, *, case_dir=None, log_output=False):
    loop = asyncio.get_running_loop()
    path = case_dir() if callable(case_dir) else case_dir

    def notify(method, *args):
        # Context methods and protocol I/O run on their owning event loop.
        return asyncio.run_coroutine_threadsafe(getattr(ctx, method)(*args), loop)

    def execute():
        with logging_context():
            if log_output and Path(path).is_dir():
                setup_logging(str(path), append=True)
            return worker(request, notify)

    async def serialized():
        # Wait on the protocol loop without consuming a worker thread. Keep
        # ownership until native execution or stop confirmation actually ends.
        async with _case_lock(path) if path is not None else nullcontext():
            return await asyncio.to_thread(execute)

    task = asyncio.create_task(serialized())
    _active_tools.add(task)

    def finished(done):
        _active_tools.discard(done)
        if not done.cancelled():
            done.exception()  # Retrieve failures even if the protocol awaiter was cancelled.

    task.add_done_callback(finished)
    return await asyncio.shield(task)
