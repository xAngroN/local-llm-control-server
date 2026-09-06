"""Container event watcher: streams podman events to a callback.

This module classifies nothing and sets no state -- it only forwards
events for one container (die/stop/start) to a callback.  The stream
catches sudden deaths (OOM kills, external ``podman kill``); hung but
still alive processes are the job of the separate health check.

The watch loop runs in a daemon thread.  If the event stream breaks
(podman restart, process death) the thread restarts it after a short
delay, until :meth:`ContainerEventWatcher.stop` is called.  Exceptions
raised by the callback are logged and never kill the watcher thread.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from llamactl.podman import Podman

logger = logging.getLogger(__name__)

#: Delay before restarting a broken event stream.
RESTART_DELAY_SECONDS = 2.0


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

    def _filters(self) -> list[str]:
        return [
            "--filter", f"container={self.container_name}",
            "--filter", "event=die",
            "--filter", "event=stop",
            "--filter", "event=start",
        ]

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
        """Stop the watcher: ends the thread and the podman subprocess."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=10)
        self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            stream = None
            try:
                stream = self.podman.events(self._filters())
                for event in stream:
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
                # Closing the iterator terminates the podman subprocess.
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
            if self._stop_event.is_set():
                break
            # Wait out the restart delay in small slices so stop() stays
            # responsive even while we are waiting.
            self._stop_event.wait(RESTART_DELAY_SECONDS)
