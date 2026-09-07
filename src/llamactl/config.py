"""Configuration profile schema and podman argument rendering for llamactl.

Arguments are always produced as flat lists of strings (never as a composed
shell string) so they can be passed to ``subprocess`` without ``shell=True``.
"""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


#: Optional performance / speculative-decoding tuning knobs shared by
#: :class:`CommonConfig` (global default) and :class:`Profile` (override).
#:
#: Each entry is ``(attribute, flag, kind)``:
#:
#: * ``value`` -- render ``[flag, str(value)]`` when the resolved value is
#:   not ``None``.
#: * ``bool``  -- render ``[flag_on]`` for ``True`` / ``[flag_off]`` for
#:   ``False`` / nothing for ``None`` (``flag`` is the ``(on, off)`` pair).
#:
#: llama.cpp flag reference (verified against
#: ``ghcr.io/ggml-org/llama.cpp:server-vulkan``):
#:
#: * ``-ngl / --n-gpu-layers``     -> ``n_gpu_layers``
#: * ``-fa / --flash-attn``        -> ``flash_attn`` ("on"|"off"|"auto")
#: * ``-ub / --ubatch-size``       -> ``ubatch_size`` (physical micro-batch)
#: * ``-cb / -nocb``               -> ``cont_batching`` (continuous batching)
#: * ``--spec-type``               -> ``spec_type`` (e.g. "draft-mtp" = MTP)
#: * ``--spec-draft-n-max``        -> ``spec_draft_n_max`` (spec-max depth)
#: * ``--spec-draft-n-min``        -> ``spec_draft_n_min``
#:
#: Note: the older ``-b / --batch-size`` (logical batch) is the mandatory
#: :attr:`Profile.batch_size` field, kept separate from the optional
#: ``-ub`` physical micro-batch above.
_TUNING_SPECS: tuple[tuple[str, object, str], ...] = (
    ("n_gpu_layers", "--n-gpu-layers", "value"),
    ("flash_attn", "--flash-attn", "value"),
    ("ubatch_size", "--ubatch-size", "value"),
    ("cont_batching", ("--cont-batching", "--no-cont-batching"), "bool"),
    ("cache_reuse", "--cache-reuse", "value"),
    ("spec_type", "--spec-type", "value"),
    ("spec_draft_n_max", "--spec-draft-n-max", "value"),
    ("spec_draft_n_min", "--spec-draft-n-min", "value"),
)

#: Names of the tuning attributes, in render order.
_TUNING_FIELDS: tuple[str, ...] = tuple(spec[0] for spec in _TUNING_SPECS)


@dataclass(frozen=True)
class Profile:
    """A single backend profile.

    ``ctx_size`` is the TOTAL context size. It is divided across the
    ``parallel`` slots (see :attr:`slot_ctx_size`), not multiplied per slot.

    The tuning fields (``ubatch_size``, ``n_gpu_layers``, ``flash_attn``,
    ``cont_batching``, ``spec_type``, ``spec_draft_n_max``,
    ``spec_draft_n_min``) are all optional: ``None`` means "inherit the
    value set on ``[common]``", and if that is also ``None`` the flag is
    not emitted at all and llama.cpp uses its own built-in default. See
    :data:`_TUNING_SPECS` for the flag mapping.
    """

    name: str
    model: str  # file name of the GGUF model
    image: str  # backend container image
    ctx_size: int  # total context size
    kv_cache_type_k: str
    kv_cache_type_v: str
    parallel: int  # number of parallel slots
    batch_size: int  # -b / --batch-size: logical maximum batch
    # Optional tuning knobs (None -> inherit from CommonConfig, then default).
    ubatch_size: int | None = None  # -ub / --ubatch-size: physical micro-batch
    n_gpu_layers: int | None = None  # -ngl / --n-gpu-layers
    flash_attn: str | None = None  # -fa / --flash-attn: on|off|auto
    cont_batching: bool | None = None  # -cb / -nocb
    cache_reuse: int | None = None  # --cache-reuse: min KV-shift reuse chunk
    spec_type: str | None = None  # --spec-type: e.g. draft-mtp for MTP
    spec_draft_n_max: int | None = None  # --spec-draft-n-max: spec-max depth
    spec_draft_n_min: int | None = None  # --spec-draft-n-min
    extra_args: tuple[str, ...] = ()

    @property
    def slot_ctx_size(self) -> int:
        """Context size per slot: total context divided across slots."""
        if self.parallel < 1:
            raise ValueError(
                f"parallel must be >= 1, got {self.parallel!r}"
            )
        if self.ctx_size % self.parallel != 0:
            raise ValueError(
                f"ctx_size {self.ctx_size} is not evenly divisible "
                f"by parallel {self.parallel}"
            )
        return self.ctx_size // self.parallel


@dataclass(frozen=True)
class CommonConfig:
    """Settings shared by all profiles (kept in the config file).

    The tuning fields mirror those on :class:`Profile` and act as the
    global default: a profile that leaves a tuning field ``None`` inherits
    the value set here. See :data:`_TUNING_SPECS`.
    """

    image: str
    models_dir: str
    container_name: str
    host_port: int
    # Global tuning defaults (None -> not set; llama.cpp default applies).
    ubatch_size: int | None = None
    n_gpu_layers: int | None = None
    flash_attn: str | None = None
    cont_batching: bool | None = None
    cache_reuse: int | None = None
    spec_type: str | None = None
    spec_draft_n_max: int | None = None
    spec_draft_n_min: int | None = None
    extra_args: tuple[str, ...] = field(default_factory=tuple)


def resolve_tuning(profile: Profile, common: CommonConfig) -> dict[str, object]:
    """Return the effective tuning values for ``profile``.

    For every tuning attribute the profile value wins when it is not
    ``None``; otherwise the :class:`CommonConfig` value is used. Values
    that are ``None`` at both levels stay ``None`` in the result. This is
    the single source of truth for both argument rendering and the API's
    read-back of a profile's effective configuration.
    """
    resolved: dict[str, object] = {}
    for name in _TUNING_FIELDS:
        profile_value = getattr(profile, name)
        resolved[name] = (
            profile_value if profile_value is not None
            else getattr(common, name)
        )
    return resolved


def render_tuning_args(profile: Profile, common: CommonConfig) -> list[str]:
    """Render only the optional tuning flags for a profile.

    Emitted in the fixed order of :data:`_TUNING_SPECS`; a tuning value
    that resolves to ``None`` contributes nothing, so a profile (and
    common) that set no tuning fields render exactly as before.
    """
    resolved = resolve_tuning(profile, common)
    args: list[str] = []
    for name, flag, kind in _TUNING_SPECS:
        value = resolved[name]
        if value is None:
            continue
        if kind == "bool":
            flag_on, flag_off = flag  # type: ignore[misc]
            args.append(flag_on if value else flag_off)
        else:
            args.extend([flag, str(value)])  # type: ignore[list-item]
    return args


def render_server_args(profile: Profile, common: CommonConfig) -> list[str]:
    """Render the llama.cpp server arguments for a profile.

    Order: model, ctx-size, parallel, batch-size, cache-type-k,
    cache-type-v, host, port, then the resolved tuning flags
    (:func:`render_tuning_args`), then ``common.extra_args`` and finally
    ``profile.extra_args``. Raw ``extra_args`` still win last, so an
    operator can always override a rendered flag by repeating it there.
    """
    args: list[str] = [
        "--model",
        f"/models/{profile.model}",
        "--ctx-size",
        str(profile.ctx_size),
        "--parallel",
        str(profile.parallel),
        "--batch-size",
        str(profile.batch_size),
        "--cache-type-k",
        profile.kv_cache_type_k,
        "--cache-type-v",
        profile.kv_cache_type_v,
        "--host",
        "0.0.0.0",
        "--port",
        str(common.host_port),
    ]
    args.extend(render_tuning_args(profile, common))
    args.extend(common.extra_args)
    args.extend(profile.extra_args)
    return args


def render_podman_args(profile: Profile, common: CommonConfig) -> list[str]:
    """Render the full command line after ``podman``.

    Order: podman run flags, image, server arguments.
    """
    args: list[str] = [
        "run",
        "--rm",
        "--detach",
        "--name",
        common.container_name,
        "--device",
        "/dev/dri",
        "--device",
        "/dev/kfd",
        "--group-add",
        "keep-groups",
        "--publish",
        f"{common.host_port}:{common.host_port}",
        "--volume",
        f"{common.models_dir}:/models:z",
        profile.image,
    ]
    args.extend(render_server_args(profile, common))
    return args


#: Tuning keys that are integers (``None`` when absent).
_INT_TUNING_KEYS = frozenset(
    {
        "ubatch_size",
        "n_gpu_layers",
        "cache_reuse",
        "spec_draft_n_max",
        "spec_draft_n_min",
    }
)
#: Tuning keys that are booleans (``None`` when absent).
_BOOL_TUNING_KEYS = frozenset({"cont_batching"})
#: Tuning keys that are plain strings (``None`` when absent).
_STR_TUNING_KEYS = frozenset({"flash_attn", "spec_type"})

#: Keys of a ``[profiles.<name>]`` table that are consumed by the schema.
_PROFILE_KEYS = {
    "model",
    "image",
    "ctx_size",
    "kv_cache_type_k",
    "kv_cache_type_v",
    "parallel",
    "batch_size",
    "extra_args",
    *_TUNING_FIELDS,
}


def _read_tuning(table: dict) -> dict[str, object]:
    """Extract and type-coerce the optional tuning keys from a TOML table.

    Every tuning key is optional; absent keys map to ``None`` so that
    :func:`resolve_tuning` treats them as "not set at this level".
    ``flash_attn`` is validated against the accepted llama.cpp values.
    """
    result: dict[str, object] = {}
    for key in _TUNING_FIELDS:
        if key not in table:
            result[key] = None
            continue
        raw = table[key]
        if key in _INT_TUNING_KEYS:
            result[key] = int(raw)
        elif key in _BOOL_TUNING_KEYS:
            if not isinstance(raw, bool):
                raise ValueError(
                    f"{key!r} must be a boolean, got {raw!r}"
                )
            result[key] = raw
        else:  # string keys
            result[key] = str(raw)
    flash = result["flash_attn"]
    if flash is not None and flash not in ("on", "off", "auto"):
        raise ValueError(
            f"flash_attn must be 'on', 'off' or 'auto', got {flash!r}"
        )
    return result


def _default_profiles_path() -> Path:
    """Resolve ``config/profiles.toml`` relative to the repo root."""
    package_root = Path(__file__).resolve().parent.parent.parent
    return package_root / "config" / "profiles.toml"


def resolve_profiles_path(path: Path | None = None) -> Path:
    """Resolve the active profiles file path.

    Resolution order (shared by :func:`load_config` and the profile
    write helpers so reads and writes always hit the same file):

    1. ``path`` if given.
    2. the ``LLAMACTL_PROFILES`` environment variable.
    3. ``config/profiles.toml`` relative to the repository root.
    """
    if path is not None:
        return path
    env = os.environ.get("LLAMACTL_PROFILES")
    return Path(env) if env else _default_profiles_path()


def load_config(path: Path | None = None) -> tuple[CommonConfig, dict[str, Profile]]:
    """Load the profile file and return ``(common, profiles)``.

    The file path is resolved by :func:`resolve_profiles_path`.

    Unknown keys inside a ``[profiles.<name>]`` table raise
    :class:`ValueError` naming the offending key.
    """
    path = resolve_profiles_path(path)
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    if "common" not in data:
        raise ValueError(f"missing [common] table in {path}")
    common = _build_common(data["common"])
    profiles: dict[str, Profile] = {}
    for name, table in data.get("profiles", {}).items():
        profiles[name] = _build_profile(name, table, common.image)
    return common, profiles


def _build_common(table: dict) -> CommonConfig:
    return CommonConfig(
        image=table["image"],
        models_dir=table["models_dir"],
        container_name=table["container_name"],
        host_port=int(table["host_port"]),
        extra_args=tuple(table.get("extra_args", ())),
        **_read_tuning(table),
    )


#: Required (non-optional) keys of a ``[profiles.<name>]`` table.
_REQUIRED_PROFILE_KEYS = (
    "model",
    "ctx_size",
    "kv_cache_type_k",
    "kv_cache_type_v",
    "parallel",
    "batch_size",
)


def _build_profile(name: str, table: dict, default_image: str) -> Profile:
    for key in table:
        if key not in _PROFILE_KEYS:
            raise ValueError(f"unknown key {key!r} in profile {name!r}")
    missing = [k for k in _REQUIRED_PROFILE_KEYS if k not in table]
    if missing:
        raise ValueError(
            f"profile {name!r} is missing required key(s): "
            + ", ".join(repr(k) for k in missing)
        )
    return Profile(
        name=name,
        model=table["model"],
        image=table.get("image", default_image),
        ctx_size=int(table["ctx_size"]),
        kv_cache_type_k=table["kv_cache_type_k"],
        kv_cache_type_v=table["kv_cache_type_v"],
        parallel=int(table["parallel"]),
        batch_size=int(table["batch_size"]),
        extra_args=tuple(table.get("extra_args", ())),
        **_read_tuning(table),
    )


def build_profile(name: str, table: dict, default_image: str) -> Profile:
    """Validate a raw profile table and return the :class:`Profile`.

    Public wrapper over :func:`_build_profile` for API-driven creation:
    rejects unknown/missing keys and bad types (:class:`ValueError`) and
    *eagerly* checks ``ctx_size`` divisibility by ``parallel`` (which is
    otherwise only validated lazily via :attr:`Profile.slot_ctx_size`),
    so an invalid profile is refused before it is ever persisted.
    """
    profile = _build_profile(name, table, default_image)
    profile.slot_ctx_size  # noqa: B018 -- raises ValueError if not divisible
    return profile


# ---------------------------------------------------------------------------
# Comment-preserving persistence of ``[profiles.<name>]`` tables
# ---------------------------------------------------------------------------

#: Canonical order in which profile keys are serialized back to TOML.
_PROFILE_WRITE_ORDER = (
    "model",
    "image",
    "ctx_size",
    "kv_cache_type_k",
    "kv_cache_type_v",
    "parallel",
    "batch_size",
    *_TUNING_FIELDS,
    "extra_args",
)


def _toml_str(value: str) -> str:
    """Serialize a string as a TOML basic string (double-quoted)."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_value(value: object) -> str:
    """Serialize a scalar / string-array profile value to TOML text."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return _toml_str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_str(str(item)) for item in value) + "]"
    raise ValueError(f"cannot serialize value {value!r} to TOML")


def format_profile_table(name: str, table: dict) -> str:
    """Render a ``[profiles.<name>]`` TOML table from a raw key/value dict.

    Only keys actually present in ``table`` are emitted (so defaults are
    never written), in the canonical :data:`_PROFILE_WRITE_ORDER`; any
    remaining keys follow in their original order.
    """
    ordered = [k for k in _PROFILE_WRITE_ORDER if k in table]
    ordered += [k for k in table if k not in _PROFILE_WRITE_ORDER]
    lines = [f"[profiles.{name}]"]
    for key in ordered:
        lines.append(f"{key} = {_toml_value(table[key])}")
    return "\n".join(lines) + "\n"


def _profile_block_span(lines: list[str], name: str) -> tuple[int, int] | None:
    """Return ``(start, end)`` line indices of a ``[profiles.<name>]`` block.

    The block runs from its header line up to (but excluding) the next
    top-level ``[...]`` table header or end of file. Returns ``None`` when
    the table is not present. Comments *above* the header are left with
    whatever precedes the block; comments *inside* the block belong to it.
    """
    header = f"[profiles.{name}]"
    start = None
    for i, line in enumerate(lines):
        if line.strip() == header:
            start = i
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start + 1, len(lines)):
        stripped = lines[j].strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            end = j
            break
    return start, end


def append_profile(path: Path, name: str, table: dict) -> None:
    """Append a new ``[profiles.<name>]`` block, preserving existing text.

    Raises :class:`ValueError` if the table already exists (callers check
    first, but this is a last-line guard against clobbering a block).
    """
    text = path.read_text(encoding="utf-8")
    if _profile_block_span(text.splitlines(), name) is not None:
        raise ValueError(f"profile {name!r} already present in {path}")
    block = format_profile_table(name, table)
    if text.strip():
        text = text.rstrip("\n") + "\n\n" + block
    else:
        text = block
    path.write_text(text, encoding="utf-8")


def overwrite_profile(path: Path, name: str, table: dict) -> None:
    """Replace an existing ``[profiles.<name>]`` block in place.

    Preserves everything else in the file, including comments *above* the
    block; comments *inside* the old block are replaced along with it.
    Raises :class:`ValueError` when the block is absent.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    span = _profile_block_span(lines, name)
    if span is None:
        raise ValueError(f"profile {name!r} not found in {path}")
    start, end = span
    block_lines = format_profile_table(name, table).splitlines()
    lines[start:end] = block_lines
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def remove_profile(path: Path, name: str) -> None:
    """Delete a ``[profiles.<name>]`` block, preserving everything else.

    Trailing blank lines left behind are collapsed so the file does not
    accumulate empty gaps. Raises :class:`ValueError` when absent.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    span = _profile_block_span(lines, name)
    if span is None:
        raise ValueError(f"profile {name!r} not found in {path}")
    start, end = span
    # Also swallow blank lines immediately following the block so removing
    # a middle table does not leave a double gap.
    while end < len(lines) and lines[end].strip() == "":
        end += 1
    del lines[start:end]
    text = "\n".join(lines).rstrip("\n")
    path.write_text(text + "\n" if text else "", encoding="utf-8")
