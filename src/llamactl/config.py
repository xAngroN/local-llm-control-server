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
    {"ubatch_size", "n_gpu_layers", "spec_draft_n_max", "spec_draft_n_min"}
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


def load_config(path: Path | None = None) -> tuple[CommonConfig, dict[str, Profile]]:
    """Load the profile file and return ``(common, profiles)``.

    The file path is resolved in this order:

    1. ``path`` if given.
    2. the ``LLAMACTL_PROFILES`` environment variable.
    3. ``config/profiles.toml`` relative to the repository root.

    Unknown keys inside a ``[profiles.<name>]`` table raise
    :class:`ValueError` naming the offending key.
    """
    if path is None:
        env = os.environ.get("LLAMACTL_PROFILES")
        path = Path(env) if env else _default_profiles_path()
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


def _build_profile(name: str, table: dict, default_image: str) -> Profile:
    for key in table:
        if key not in _PROFILE_KEYS:
            raise ValueError(f"unknown key {key!r} in profile {name!r}")
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
