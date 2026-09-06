"""Runtime metrics collection for the llamactl control server.

The :class:`MetricsCollector` merges two independent sources that must
both survive being missing (test environments have neither sysfs nor a
model server):

* GPU VRAM, read from sysfs. The device node lives at
  ``/sys/class/drm/card<N>/device/mem_info_vram_used`` where ``<N>``
  changes between boots, hence the configurable glob.
* Token throughput and the loaded model, read from the llama.cpp model
  server (``/props`` and ``/metrics``). Throughput is *not* computed
  here: llama.cpp already publishes per-second counters in its
  Prometheus output, so the parser only extracts those values.

Neither source is ever asked to fail the other: every read degrades to
``None`` values instead of raising, so ``GET /metrics`` on the control
server (port 8081, not to be confused with the model server's own
``/metrics`` on port 8080) always answers.
"""

from __future__ import annotations

import glob

import httpx

from .state import InstanceState

#: Bytes per (decimal) gibibyte used for the *_gib conversions.
_GIB = 1024.0**3

#: HTTP timeout for model-server requests, in seconds.
_TIMEOUT = 3.0


def _parse_prometheus(text: str) -> dict[str, float]:
    """Parse Prometheus exposition text line by line.

    Lines are of the form ``<name> <value>``; comment lines starting
    with ``#`` and blank lines are skipped. Lines that do not split
    into exactly two fields are ignored.
    """
    result: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        name, raw = parts
        try:
            result[name] = float(raw)
        except ValueError:
            continue
    return result


def _read_int_file(path: str) -> int | None:
    """Read an integer from a file, returning ``None`` when unreadable."""
    try:
        with open(path, "r", encoding="ascii") as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


class MetricsCollector:
    """Collect VRAM usage and model-server throughput for one report."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        vram_glob: str = "/sys/class/drm/card*/device/mem_info_vram_used",
    ) -> None:
        self._base_url = base_url
        self._vram_glob = vram_glob

    # ------------------------------------------------------------------
    # VRAM
    # ------------------------------------------------------------------

    def read_vram(self) -> dict:
        """Read used and total VRAM in bytes from sysfs.

        The first path matching :attr:`_vram_glob` is used; the
        neighbouring ``mem_info_vram_total`` file (same directory, name
        swapped) gives the total. Returns
        ``{"used_bytes", "total_bytes", "used_gib", "total_gib"}``;
        every value is ``None`` when the matching file is absent or
        unreadable -- never an exception.
        """
        used_bytes: int | None = None
        total_bytes: int | None = None
        matches = glob.glob(self._vram_glob)
        if matches:
            first = sorted(matches)[0]
            used_bytes = _read_int_file(first)
            total_path = first.replace("mem_info_vram_used", "mem_info_vram_total")
            total_bytes = _read_int_file(total_path)
        return {
            "used_bytes": used_bytes,
            "total_bytes": total_bytes,
            "used_gib": (used_bytes / _GIB) if used_bytes is not None else None,
            "total_gib": (total_bytes / _GIB) if total_bytes is not None else None,
        }

    # ------------------------------------------------------------------
    # Model server
    # ------------------------------------------------------------------

    def read_model_metrics(self) -> dict:
        """Query the model server's ``/props`` and ``/metrics``.

        Returns ``{"model", "prompt_tps", "generation_tps"}`` extracted
        from the answers. Any transport failure (connection refused,
        timeout, HTTP error) yields ``None`` for every field instead of
        an exception.
        """
        model: str | None = None
        prompt_tps: float | None = None
        generation_tps: float | None = None

        try:
            response = httpx.get(self._base_url.rstrip("/") + "/props", timeout=_TIMEOUT)
            if response.status_code == 200:
                props = response.json()
                if isinstance(props, dict):
                    candidate = props.get("model")
                    if isinstance(candidate, str):
                        model = candidate
        except (httpx.HTTPError, ValueError):
            pass

        try:
            response = httpx.get(self._base_url.rstrip("/") + "/metrics", timeout=_TIMEOUT)
            if response.status_code == 200:
                samples = _parse_prometheus(response.text)
                if samples["llama_eval_time"]:
                    prompt_tps = samples["llama_eval_count"] / samples["llama_eval_time"]
                if samples["llama_sample_time"]:
                    generation_tps = samples["llama_sample_count"] / samples["llama_sample_time"]
        except (httpx.HTTPError, ValueError):
            pass

        return {
            "model": model,
            "prompt_tps": prompt_tps,
            "generation_tps": generation_tps,
        }

    # ------------------------------------------------------------------
    # Merged report
    # ------------------------------------------------------------------

    def collect(self, state: InstanceState, profile: str | None) -> dict:
        """Build one merged metrics report for the given tracker state.

        In state ``ready`` the model server is queried (throughput plus
        loaded model) and the current VRAM consumption is included; in
        every other state the model server is *not* contacted at all,
        ``model`` is ``None`` and only the VRAM baseline is reported.
        """
        vram = self.read_vram()
        result: dict = {
            "state": state.value,
            "profile": profile,
            "model": None,
            "prompt_tps": None,
            "generation_tps": None,
            "vram": vram,
        }
        if state is InstanceState.READY:
            model_metrics = self.read_model_metrics()
            result["model"] = model_metrics["model"]
            result["prompt_tps"] = model_metrics["prompt_tps"]
            result["generation_tps"] = model_metrics["generation_tps"]
        return result
