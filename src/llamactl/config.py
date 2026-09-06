"""Configuration profile schema and podman argument rendering for llamactl.

Arguments are always produced as flat lists of strings (never as a composed
shell string) so they can be passed to ``subprocess`` without ``shell=True``.
"""

from dataclasses import dataclass, field


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
