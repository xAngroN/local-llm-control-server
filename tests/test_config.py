"""Tests for llamactl.config profile schema and podman argument rendering."""

from pathlib import Path

import pytest

from llamactl.config import (
    CommonConfig,
    Profile,
    append_profile,
    build_profile,
    format_profile_table,
    load_config,
    overwrite_profile,
    remove_profile,
    render_podman_args,
    render_server_args,
    resolve_tuning,
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
        # -ngl / -fa are now typed tuning fields, not raw extra_args.
        assert common.n_gpu_layers == 999
        assert common.flash_attn == "on"
        assert common.cont_batching is True
        extra = tuple(common.extra_args)
        assert extra == ("--jinja", "--metrics")
        # The removed --draft-max / --draft flags (hard start error in the
        # current llama.cpp) must not linger anywhere in the config.
        assert "--draft-max" not in extra
        assert "--draft" not in extra

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
            # -ngl is rendered from the typed common tuning field now.
            assert "--n-gpu-layers" in args
            assert args[args.index("--n-gpu-layers") + 1] == "999"

    def test_all_profiles_render_with_metrics_flag(self) -> None:
        common, profiles = load_config(BUNDLED_PROFILES)
        for name in ("fast", "large", "safe", "shared"):
            args = render_server_args(profiles[name], common)
            assert "--metrics" in args, name

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


class TestTuning:
    """Optional tuning knobs: -ub, -ngl, -fa, -cb, MTP/spec-decoding."""

    def test_unset_tuning_emits_nothing(self) -> None:
        # A profile and common with no tuning fields must render exactly as
        # before -- no stray flags leak in.
        args = render_server_args(_profile(), _common())
        for flag in (
            "--ubatch-size",
            "--n-gpu-layers",
            "--flash-attn",
            "--cont-batching",
            "--no-cont-batching",
            "--cache-reuse",
            "--spec-type",
            "--spec-draft-n-max",
            "--spec-draft-n-min",
        ):
            assert flag not in args, flag

    def test_all_tuning_fields_render_expected_flags(self) -> None:
        profile = _profile(
            ubatch_size=256,
            n_gpu_layers=999,
            flash_attn="on",
            cont_batching=True,
            cache_reuse=256,
            spec_type="draft-mtp",
            spec_draft_n_max=4,
            spec_draft_n_min=0,
        )
        args = render_server_args(profile, _common())

        def pair(flag: str) -> str:
            return args[args.index(flag) + 1]

        assert pair("--ubatch-size") == "256"
        assert pair("--n-gpu-layers") == "999"
        assert pair("--flash-attn") == "on"
        assert "--cont-batching" in args
        assert pair("--cache-reuse") == "256"
        assert pair("--spec-type") == "draft-mtp"
        assert pair("--spec-draft-n-max") == "4"
        assert pair("--spec-draft-n-min") == "0"

    def test_cont_batching_false_renders_disable_flag(self) -> None:
        args = render_server_args(_profile(cont_batching=False), _common())
        assert "--no-cont-batching" in args
        assert "--cont-batching" not in args

    def test_profile_overrides_common_tuning(self) -> None:
        common = _common(n_gpu_layers=999, flash_attn="on", ubatch_size=512)
        profile = _profile(n_gpu_layers=0, ubatch_size=None)
        resolved = resolve_tuning(profile, common)
        # Profile value wins when set...
        assert resolved["n_gpu_layers"] == 0
        # ...common value is inherited when the profile leaves it None...
        assert resolved["ubatch_size"] == 512
        assert resolved["flash_attn"] == "on"
        # ...and unset-at-both-levels stays None.
        assert resolved["spec_type"] is None

    def test_zero_is_not_treated_as_unset(self) -> None:
        # 0 is a meaningful value (e.g. n-gpu-layers=0) and must still be
        # emitted, not swallowed by the None-inheritance logic.
        args = render_server_args(_profile(n_gpu_layers=0), _common(n_gpu_layers=999))
        assert args[args.index("--n-gpu-layers") + 1] == "0"

    def test_tuning_from_toml(self, tmp_path) -> None:
        file = tmp_path / "profiles.toml"
        file.write_text(
            "[common]\n"
            "image = \"img\"\n"
            "models_dir = \"/m\"\n"
            "container_name = \"c\"\n"
            "host_port = 8080\n"
            "n_gpu_layers = 999\n"
            "flash_attn = \"on\"\n"
            "cont_batching = true\n"
            "[profiles.mtp]\n"
            "model = \"m.gguf\"\n"
            "ctx_size = 1024\n"
            "kv_cache_type_k = \"f16\"\n"
            "kv_cache_type_v = \"f16\"\n"
            "parallel = 1\n"
            "batch_size = 128\n"
            "ubatch_size = 256\n"
            "spec_type = \"draft-mtp\"\n"
            "spec_draft_n_max = 4\n"
        )
        common, profiles = load_config(file)
        assert common.n_gpu_layers == 999
        assert common.cont_batching is True
        p = profiles["mtp"]
        assert p.ubatch_size == 256
        assert p.spec_type == "draft-mtp"
        assert p.spec_draft_n_max == 4
        # common tuning is inherited where the profile is silent
        resolved = resolve_tuning(p, common)
        assert resolved["n_gpu_layers"] == 999
        assert resolved["flash_attn"] == "on"

    def test_invalid_flash_attn_raises(self, tmp_path) -> None:
        file = tmp_path / "profiles.toml"
        file.write_text(
            "[common]\n"
            "image = \"img\"\n"
            "models_dir = \"/m\"\n"
            "container_name = \"c\"\n"
            "host_port = 8080\n"
            "flash_attn = \"yes\"\n"
        )
        with pytest.raises(ValueError, match="flash_attn"):
            load_config(file)


_SRC_WITH_COMMENTS = """\
# top comment
[common]
image = "img"
models_dir = "/m"
container_name = "c"
host_port = 8000

# comment above safe
[profiles.safe]
model = "safe.gguf"
ctx_size = 32768
kv_cache_type_k = "q8_0"
kv_cache_type_v = "q8_0"
parallel = 1
batch_size = 512
"""


class TestBuildProfile:
    def test_valid_table_builds(self) -> None:
        p = build_profile(
            "x",
            {
                "model": "m.gguf",
                "ctx_size": 4096,
                "kv_cache_type_k": "q8_0",
                "kv_cache_type_v": "q8_0",
                "parallel": 1,
                "batch_size": 512,
                "spec_type": "draft-mtp",
            },
            "default:img",
        )
        assert p.model == "m.gguf"
        assert p.image == "default:img"
        assert p.spec_type == "draft-mtp"

    def test_missing_required_key_raises(self) -> None:
        with pytest.raises(ValueError, match="ctx_size"):
            build_profile(
                "x",
                {
                    "model": "m.gguf",
                    "kv_cache_type_k": "q8_0",
                    "kv_cache_type_v": "q8_0",
                    "parallel": 1,
                    "batch_size": 512,
                },
                "img",
            )

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(ValueError, match="bogus"):
            build_profile(
                "x",
                {
                    "model": "m.gguf",
                    "ctx_size": 4096,
                    "kv_cache_type_k": "q8_0",
                    "kv_cache_type_v": "q8_0",
                    "parallel": 1,
                    "batch_size": 512,
                    "bogus": 1,
                },
                "img",
            )

    def test_non_divisible_ctx_raises_eagerly(self) -> None:
        # slot_ctx_size is a lazy property; build_profile must trigger it.
        with pytest.raises(ValueError):
            build_profile(
                "x",
                {
                    "model": "m.gguf",
                    "ctx_size": 100,
                    "kv_cache_type_k": "q8_0",
                    "kv_cache_type_v": "q8_0",
                    "parallel": 3,
                    "batch_size": 512,
                },
                "img",
            )


class TestProfilePersistence:
    def test_format_profile_table_orders_and_types(self) -> None:
        text = format_profile_table(
            "mtp",
            {
                "model": "m.gguf",
                "ctx_size": 4096,
                "kv_cache_type_k": "q8_0",
                "kv_cache_type_v": "q8_0",
                "parallel": 1,
                "batch_size": 512,
                "cont_batching": True,
                "spec_type": "draft-mtp",
                "extra_args": ["--foo", "bar"],
            },
        )
        assert text.startswith("[profiles.mtp]\n")
        assert 'model = "m.gguf"' in text
        assert "ctx_size = 4096" in text
        assert "cont_batching = true" in text
        assert 'extra_args = ["--foo", "bar"]' in text
        # canonical order: model before batch_size before tuning.
        assert text.index("model") < text.index("batch_size") < text.index("spec_type")

    def test_append_preserves_comments_and_roundtrips(self, tmp_path) -> None:
        f = tmp_path / "p.toml"
        f.write_text(_SRC_WITH_COMMENTS)
        append_profile(
            f, "mtp",
            {
                "model": "m.gguf", "ctx_size": 4096, "kv_cache_type_k": "q8_0",
                "kv_cache_type_v": "q8_0", "parallel": 1, "batch_size": 512,
                "spec_type": "draft-mtp",
            },
        )
        text = f.read_text()
        assert "# top comment" in text
        assert "# comment above safe" in text
        _, profiles = load_config(f)
        assert set(profiles) == {"safe", "mtp"}
        assert profiles["mtp"].spec_type == "draft-mtp"

    def test_append_existing_raises(self, tmp_path) -> None:
        f = tmp_path / "p.toml"
        f.write_text(_SRC_WITH_COMMENTS)
        with pytest.raises(ValueError, match="already"):
            append_profile(
                f, "safe",
                {
                    "model": "m.gguf", "ctx_size": 4096, "kv_cache_type_k": "q8_0",
                    "kv_cache_type_v": "q8_0", "parallel": 1, "batch_size": 512,
                },
            )

    def test_overwrite_replaces_block_preserving_rest(self, tmp_path) -> None:
        f = tmp_path / "p.toml"
        f.write_text(_SRC_WITH_COMMENTS)
        overwrite_profile(
            f, "safe",
            {
                "model": "safe.gguf", "ctx_size": 16384, "kv_cache_type_k": "f16",
                "kv_cache_type_v": "f16", "parallel": 1, "batch_size": 1024,
            },
        )
        text = f.read_text()
        assert "# top comment" in text
        assert "# comment above safe" in text
        _, profiles = load_config(f)
        assert profiles["safe"].ctx_size == 16384
        assert profiles["safe"].kv_cache_type_k == "f16"

    def test_remove_deletes_block_preserving_rest(self, tmp_path) -> None:
        f = tmp_path / "p.toml"
        f.write_text(_SRC_WITH_COMMENTS)
        remove_profile(f, "safe")
        text = f.read_text()
        assert "# top comment" in text
        assert "[profiles.safe]" not in text
        _, profiles = load_config(f)
        assert profiles == {}

    def test_remove_absent_raises(self, tmp_path) -> None:
        f = tmp_path / "p.toml"
        f.write_text(_SRC_WITH_COMMENTS)
        with pytest.raises(ValueError, match="not found"):
            remove_profile(f, "nope")
