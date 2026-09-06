"""Configuration profile schema and podman argument rendering for llamactl.

Arguments are always produced as flat lists of strings (never as a composed
shell string) so they can be passed to ``subprocess`` without ``shell=True``.
"""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Profile:
    """A single backend profile.

    ``ctx_size`` is the TOTAL context size. It is divided across the
    ``parallel`` slots (see :attr:`slot_ctx_size`), not multiplied per slot.
    """

    name: str
    model: str  # file name of the GGUF model
    image: str  # backend container image
    ctx_size: int  # total context size
    kv_cache_type_k: str
    kv_cache_type_v: str
    parallel: int  # number of parallel slots
    batch_size: int
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
    """Settings shared by all profiles (kept in the config file)."""

    image: str
    models_dir: str
    container_name: str
    host_port: int
    extra_args: tuple[str, ...] = field(default_factory=tuple)


def render_server_args(profile: Profile, common: CommonConfig) -> list[str]:
    """Render the llama.cpp server arguments for a profile.

    Order: model, ctx-size, parallel, batch-size, cache-type-k,
    cache-type-v, host, port, then ``common.extra_args`` and finally
    ``profile.extra_args``.
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


#: Keys of a ``[profiles.<name>]`` table that are consumed by the schema.
_PROFILE_KEYS = {
    "model",
    "ctx_size",
    "kv_cache_type_k",
    "kv_cache_type_v",
    "parallel",
    "batch_size",
    "extra_args",
}


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
    )
