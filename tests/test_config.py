"""Tests for llamactl.config profile schema and podman argument rendering."""

from pathlib import Path

import pytest

from llamactl.config import (
    CommonConfig,
    Profile,
    load_config,
    render_podman_args,
    render_server_args,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLED_PROFILES = REPO_ROOT / "config" / "profiles.toml"


def _profile(**kwargs) -> Profile:
    defaults = dict(
        name="p1",
        model="model.gguf",
        image="quay.io/llamactl/backend:latest",
        ctx_size=32768,
        kv_cache_type_k="q8_0",
        kv_cache_type_v="q8_0",
        parallel=4,
        batch_size=512,
        extra_args=(),
    )
    defaults.update(kwargs)
    return Profile(**defaults)


def _common(**kwargs) -> CommonConfig:
    defaults = dict(
        image="quay.io/llamactl/base:latest",
        models_dir="/srv/models",
        container_name="llama-p1",
        host_port=8080,
        extra_args=(),
    )
    defaults.update(kwargs)
    return CommonConfig(**defaults)


class TestSlotCtxSize:
    def test_total_ctx_split_across_slots(self) -> None:
        profile = _profile(ctx_size=32768, parallel=4)
        assert profile.slot_ctx_size == 8192
        # total context stays 32768, it is not multiplied per slot
        assert profile.ctx_size == 32768

    def test_single_slot(self) -> None:
        profile = _profile(ctx_size=1024, parallel=1)
        assert profile.slot_ctx_size == 1024

    def test_parallel_less_than_one_raises(self) -> None:
        profile = _profile(parallel=0)
        with pytest.raises(ValueError):
            profile.slot_ctx_size
        profile = _profile(parallel=-3)
        with pytest.raises(ValueError):
            profile.slot_ctx_size

    def test_non_divisible_ctx_raises(self) -> None:
        profile = _profile(ctx_size=100, parallel=3)
        with pytest.raises(ValueError):
            profile.slot_ctx_size


class TestRenderServerArgs:
    def test_order_and_values(self) -> None:
        profile = _profile(
            model="qwen.gguf",
            kv_cache_type_k="f16",
            kv_cache_type_v="q8_0",
            batch_size=256,
        )
        common = _common(host_port=8080)
        assert render_server_args(profile, common) == [
            "--model",
            "/models/qwen.gguf",
            "--ctx-size",
            "32768",
            "--parallel",
            "4",
            "--batch-size",
            "256",
            "--cache-type-k",
            "f16",
            "--cache-type-v",
            "q8_0",
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
        ]

    def test_common_then_profile_extra_args(self) -> None:
        profile = _profile(extra_args=("--log-disable",))
        common = _common(extra_args=("-ngl", "999", "-fa", "on", "--jinja"))
        args = render_server_args(profile, common)
        assert args[-6:] == ["-ngl", "999", "-fa", "on", "--jinja", "--log-disable"]
        # common extra args come before profile extra args
        assert args.index("--jinja") < args.index("--log-disable")


class TestRenderPodmanArgs:
    def test_full_command_order(self) -> None:
        profile = _profile(image="quay.io/llamactl/backend:v1")
        common = _common(host_port=8080)
        args = render_podman_args(profile, common)
        assert args == [
            "run",
            "--rm",
            "--detach",
            "--name",
            "llama-p1",
            "--device",
            "/dev/dri",
            "--device",
            "/dev/kfd",
            "--group-add",
            "keep-groups",
            "--publish",
            "8080:8080",
            "--volume",
            "/srv/models:/models:z",
            "quay.io/llamactl/backend:v1",
            "--model",
            "/models/model.gguf",
            "--ctx-size",
            "32768",
            "--parallel",
            "4",
            "--batch-size",
            "512",
            "--cache-type-k",
            "q8_0",
            "--cache-type-v",
            "q8_0",
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
        ]

    def test_flat_list_of_strings_image_before_server_args(self) -> None:
        profile = _profile()
        common = _common()
        args = render_podman_args(profile, common)
        assert all(isinstance(a, str) for a in args)
        assert not any(c in a for a in args for c in (";", "&", "|", ">", "<", "`"))
        image_idx = args.index(profile.image)
        server_idx = args.index("--model")
        assert image_idx < server_idx
        # podman flags come first
        assert args[0] == "run"
        assert "--name" in args[: image_idx]
        assert "--publish" in args[: image_idx]


class TestLoadConfig:
    def test_bundled_file_loads(self) -> None:
        common, profiles = load_config(BUNDLED_PROFILES)
        assert isinstance(common, CommonConfig)
        assert set(profiles) == {"fast", "large", "safe", "shared"}

    def test_bundled_common_values(self) -> None:
        common, _ = load_config(BUNDLED_PROFILES)
        assert common.models_dir == "/var/home/bazzite/models"
        assert common.container_name == "llamactl-model"
        assert common.host_port == 8080
        extra = tuple(common.extra_args)
        for value in ("-ngl", "999", "-fa", "on", "--jinja"):
            assert value in extra
        assert extra[:5] == ("-ngl", "999", "-fa", "on", "--jinja")

    def test_shared_splits_ctx_across_slots(self) -> None:
        _, profiles = load_config(BUNDLED_PROFILES)
        shared = profiles["shared"]
        assert shared.ctx_size == 32768
        assert shared.parallel == 4
        assert shared.slot_ctx_size == 8192

    def test_ported_profiles_pin_parallel_one(self) -> None:
        _, profiles = load_config(BUNDLED_PROFILES)
        for name in ("fast", "large", "safe"):
            profile = profiles[name]
            assert profile.parallel == 1, name
            # single slot: total context is not split
            assert profile.slot_ctx_size == profile.ctx_size

    def test_render_podman_args_for_each_profile(self) -> None:
        common, profiles = load_config(BUNDLED_PROFILES)
        for name, profile in profiles.items():
            args = render_podman_args(profile, common)
            assert args[0] == "run"
            assert all(isinstance(a, str) for a in args)
            assert profile.image in args
            # common extra flags are applied for every profile
            assert "--jinja" in args
            assert "-ngl" in args and args[args.index("-ngl") + 1] == "999"

    def test_env_var_is_respected(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("LLAMACTL_PROFILES", str(BUNDLED_PROFILES))
        common, profiles = load_config()
        assert set(profiles) == {"fast", "large", "safe", "shared"}
        assert common == load_config(BUNDLED_PROFILES)[0]

    def test_default_path_without_env(self, monkeypatch) -> None:
        monkeypatch.delenv("LLAMACTL_PROFILES", raising=False)
        common, profiles = load_config()
        assert set(profiles) == {"fast", "large", "safe", "shared"}
        assert common == load_config(BUNDLED_PROFILES)[0]

    def test_unknown_profile_key_raises_value_error(self, tmp_path) -> None:
        file = tmp_path / "profiles.toml"
        file.write_text(
            "[common]\n"
            "image = \"img\"\n"
            "models_dir = \"/m\"\n"
            "container_name = \"c\"\n"
            "host_port = 8080\n"
            "[profiles.fast]\n"
            "model = \"m.gguf\"\n"
            "ctx_size = 1024\n"
            "kv_cache_type_k = \"f16\"\n"
            "kv_cache_type_v = \"f16\"\n"
            "parallel = 1\n"
            "batch_size = 128\n"
            "bogus_key = 1\n"
        )
        with pytest.raises(ValueError, match="bogus_key"):
            load_config(file)
