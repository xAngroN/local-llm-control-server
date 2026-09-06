"""Container event watcher: streams podman events to a callback.

This module classifies nothing and sets no state -- it only forwards
events for one container (die/stop/start) to a callback.  The stream
catches sudden deaths (OOM kills, external ``podman kill``); hung but
still alive processes are the job of the separate health check.

The watch loop runs in a daemon thread.  If the event stream breaks
(podman restart, process death) the thread restarts it after a short
delay, until :meth:`ContainerEventWatcher.stop` is called.  Exceptions
raised by the callback are logged and never kill the watcher thread.

``stop()`` must be able to end the watcher even while the thread is
blocked waiting for the next event from a *live* stream -- the normal
case for a running container.  A plain generator cannot be ``close()``d
from another thread while it is suspended, so the watcher wraps the
stream in :class:`StreamHandle`: ``stop()`` resolves the underlying
``podman events`` subprocess and terminates it, which forces an EOF on
the blocked ``readline`` so the thread wakes and exits promptly.
"""

from __future__ import annotations

import logging
import subprocess
import threading
from collections.abc import Callable, Iterator

from llamactl.podman import Podman

logger = logging.getLogger(__name__)

#: Delay before restarting a broken event stream.
RESTART_DELAY_SECONDS = 2.0
#: How long ``stop()`` waits for the thread to exit after the subprocess
#: is terminated.  Terminating the process forces an EOF on the blocked
#: read, so the thread should finish well within this bound.
STOP_JOIN_TIMEOUT_SECONDS = 10.0


class StreamHandle:
    """Wraps a podman events stream so ``stop()`` can end the subprocess.

    The underlying :class:`Podman.events` generator does not expose its
    ``Popen`` object, and a generator cannot be closed from another
    thread while it is suspended on a read.  This handle therefore
    resolves the process lazily -- on first close -- and terminates it
    directly, which is what actually unblocks a thread parked in
    ``readline``.  ``close()`` is idempotent and safe to call from the
    watcher thread's ``finally`` block as well.
    """

    def __init__(self, stream: Iterator[dict]) -> None:
        self._stream = stream
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def __iter__(self) -> "StreamHandle":
        return self

    def __next__(self) -> dict:
        return next(self._stream)

    def close(self) -> None:
        with self._lock:
            if self._proc is not None:
                self._terminate(self._proc)
                return
            self._proc = self._resolve_proc(self._stream)
        if self._proc is None:
            # No subprocess to kill (e.g. an in-memory fake stream);
            # closing the iterator still lets its finally block run.
            self._safe_close(self._stream)
            return
        self._terminate(self._proc)
        # Closing the generator also runs its finally (terminate + wait),
        # which is a harmless second cleanup pass.
        self._safe_close(self._stream)

    @staticmethod
    def _safe_close(obj: object) -> None:
        close = getattr(obj, "close", None)
        if close is None:
            return
        try:
            close()
        except Exception:
            pass

    @staticmethod
    def _terminate(proc: subprocess.Popen) -> None:
        try:
            if proc.poll() is None:
                proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except OSError:
                pass

    @staticmethod
    def _resolve_proc(stream: object) -> subprocess.Popen | None:
        """Best-effort lookup of the Popen owned by the events generator.

        When a generator is suspended at its ``yield``, ``gi_frame`` is the
        generator's own frame -- the one that holds the ``Popen`` -- so we
        search it first and fall back to the frame that started it.
        """
        gi = getattr(stream, "gi_frame", None)
        frames = []
        if gi is not None:
            frames.append(gi)
            back = gi.f_back
            if back is not None:
                frames.append(back)
        for frame in frames:
            for value in frame.f_locals.values():
                if isinstance(value, subprocess.Popen):
                    return value
        return None


class ContainerEventWatcher:
    """Watch podman events for one container and forward them to a callback."""

    def __init__(
        self,
        podman: Podman,
        container_name: str,
        on_event: Callable[[dict], None],
    ) -> None:
        self.podman = podman
        self.container_name = container_name
        self.on_event = on_event
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # The stream handle currently owned by the thread, plus a lock so
        # the thread and stop() always agree on which subprocess is live.
        self._current_stream: StreamHandle | None = None
        self._stream_lock = threading.Lock()

    def _filters(self) -> list[str]:
        return [
            "--filter", f"container={self.container_name}",
            "--filter", "event=die",
            "--filter", "event=stop",
            "--filter", "event=start",
        ]

    def _take_stream(self, handle: StreamHandle | None) -> None:
        """Record the stream owned by the thread so stop() can end it."""
        with self._stream_lock:
            self._current_stream = handle

    def _drop_stream(self, handle: StreamHandle | None) -> None:
        """Clear the current-stream slot if it still points at this handle."""
        if handle is None:
            return
        with self._stream_lock:
            if self._current_stream is handle:
                self._current_stream = None

    def start(self) -> None:
        """Start the watcher daemon thread; returns immediately."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        thread = threading.Thread(
            target=self._run,
            name=f"container-events-{self.container_name}",
            daemon=True,
        )
        self._thread = thread
        thread.start()

    def stop(self) -> None:
        """Stop the watcher: ends the thread and the podman subprocess.

        Setting the stop event wakes any slice of the restart wait, and
        closing the current stream terminates the underlying ``podman
        events`` process -- forcing an EOF on a thread blocked reading
        the next event so it exits promptly.  No callbacks arrive
        afterwards.
        """
        self._stop_event.set()
        with self._stream_lock:
            handle = self._current_stream
            self._current_stream = None
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=STOP_JOIN_TIMEOUT_SECONDS)
        self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            handle: StreamHandle | None = None
            try:
                stream = self.podman.events(self._filters())
                handle = StreamHandle(stream)
                self._take_stream(handle)
                for event in handle:
                    if self._stop_event.is_set():
                        break
                    try:
                        self.on_event(event)
                    except Exception:
                        logger.exception(
                            "event callback failed for container %s",
                            self.container_name,
                        )
            except Exception:
                logger.exception(
                    "event stream for container %s broke",
                    self.container_name,
                )
            finally:
                # Closing the handle terminates the podman subprocess.
                if handle is not None:
                    try:
                        handle.close()
                    except Exception:
                        pass
                    self._drop_stream(handle)
            if self._stop_event.is_set():
                break
            # Wait out the restart delay in one slice; stop() wakes it.
            self._stop_event.wait(RESTART_DELAY_SECONDS)
