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
case for a running container.  A plain iterator cannot be woken from
another thread while it is suspended on a read, so the watcher holds a
reference to the stream the thread is reading from (see
:meth:`Podman.events`, whose :class:`EventStream` exposes its
subprocess) and terminates that subprocess on ``stop()``: killing the
process forces an EOF on the blocked ``readline`` so the thread wakes
and exits promptly.
"""

from __future__ import annotations

import logging
import subprocess
import threading
from collections.abc import Callable

from llamactl.podman import Podman

logger = logging.getLogger(__name__)

#: Delay before restarting a broken event stream.
RESTART_DELAY_SECONDS = 2.0
#: How long ``stop()`` waits for the thread to exit after the subprocess
#: is terminated.  Terminating the process forces an EOF on the blocked
#: read, so the thread should finish well within this bound.
STOP_JOIN_TIMEOUT_SECONDS = 10.0


class StreamHandle:
    """Wraps a podman event stream so ``stop()`` can end the subprocess.

    The underlying :class:`Podman.events` stream lazily starts its
    subprocess on first iteration and then exposes it as ``stream.process``.
    This wrapper keeps that stream, polls for the process reference once
    the thread has begun iterating, and terminates the process on
    ``close()`` -- forcing an EOF on a reader blocked in ``next()``.
    A stream whose process is never observed is simply closed, which
    lets an in-memory iterator run its cleanup.  ``close()`` is idempotent
    and safe to call from the watcher thread's ``finally`` block as well
    as from ``stop()``.
    """

    def __init__(self, stream: object) -> None:
        self._stream = stream
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def _attach_process(self) -> None:
        """Record the stream's subprocess once it has been started."""
        proc = getattr(self._stream, "process", None)
        if isinstance(proc, subprocess.Popen):
            with self._lock:
                self._proc = proc

    def close(self) -> None:
        # Everything happens under the lock so the thread's ``finally``
        # block and stop() can both call close() on the same handle and
        # still terminate the subprocess exactly once: whichever call
        # first detaches ``_proc`` is the one that terminates it, the
        # other sees ``_proc is None`` and does nothing.  If the process
        # was started but not yet attached (the thread is still on the
        # first next()), read it straight off the stream under the lock
        # and attach it here.
        stream = self._stream
        with self._lock:
            if self._proc is None:
                maybe = getattr(stream, "process", None)
                if isinstance(maybe, subprocess.Popen):
                    self._proc = maybe
            proc, self._proc = self._proc, None
        if proc is not None:
            self._terminate(proc)
        self._safe_close(stream)

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
        # the thread and stop() always agree on which stream is live.
        self._current_stream: StreamHandle | None = None
        self._stream_lock = threading.Lock()

    def _filters(self) -> list[str]:
        return [
            "--filter", f"container={self.container_name}",
            "--filter", "event=die",
            "--filter", "event=stop",
            "--filter", "event=start",
        ]

    def _open_stream(self) -> StreamHandle:
        """Start a podman event stream and wrap it for termination."""
        return StreamHandle(self.podman.events(self._filters()))

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
        terminating the current subprocess forces an EOF on a thread
        blocked reading the next event so it exits promptly.  No
        callbacks arrive afterwards.
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
                handle = self._open_stream()
                self._take_stream(handle)
                stream = handle._stream
                while not self._stop_event.is_set():
                    # Attach the lazily-started subprocess as soon as it
                    # exists so stop() can terminate it directly.
                    if handle._proc is None:
                        handle._attach_process()
                    try:
                        event = next(stream)
                    except StopIteration:
                        break
                    except Exception:
                        logger.exception(
                            "event stream for container %s broke",
                            self.container_name,
                        )
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
