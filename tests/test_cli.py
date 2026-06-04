"""Tests for the model-advisor CLI surfaces (advise / inspect / apply).

These tests own the *CLI* contract only; the engine/store/config are a sibling
bead.  To stay runnable before that bead lands (and to keep the CLI assertions
hermetic regardless), we:

  * load ``modeladvisor/cli.py`` as a **standalone module** via importlib, so the
    package ``__init__`` re-exports (which eagerly import the not-yet-present
    engine/store) never block these tests; and
  * inject a tiny fake engine + state.  If the real ``modeladvisor.engine`` is
    importable we still prefer it for a smoke check, but the behavioural
    assertions drive the fake so they are deterministic and self-contained.

Critically, the ``apply`` tests NEVER touch live ``/Users/jayse/Code`` config:
they operate on TEMP copies of an ``agent.toml`` / ``city.toml`` created under
``tmp_path`` and point the resolver at them via ``$ADVISOR_AGENT_TOML``.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import pytest

PACK_DIR = Path(__file__).resolve().parents[1]
CLI_PATH = PACK_DIR / "modeladvisor" / "cli.py"


def _load_cli_module():
    """Load cli.py standalone, bypassing the package __init__ re-exports."""
    spec = importlib.util.spec_from_file_location("mad_cli_under_test", CLI_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mad_cli_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


cli = _load_cli_module()


# --------------------------------------------------------------------------- #
# Fakes implementing the shared engine/state contract
# --------------------------------------------------------------------------- #

def _fake_reasons(rec_tier: str):
    """A representative `reasons` audit object (DESIGN §1.4 / §6)."""
    return {
        "candidates": [
            {"tier_id": "haiku", "q_lo": 0.41, "exp_loss": 0.004,
             "cost_diff": -0.0142},
            {"tier_id": "sonnet", "q_lo": 0.82, "exp_loss": -0.011,
             "cost_diff": -0.0114},
            {"tier_id": "opus", "q_lo": 0.71, "exp_loss": 0.0,
             "cost_diff": 0.0},
        ],
        "eval_flag": False,
        "baseline_tier": "opus",
        "recommended": rec_tier,
    }


class FakeEngine:
    """Stands in for ``modeladvisor.engine`` under the shared contract."""

    def __init__(self, *, model="claude-sonnet-4-5", tier="sonnet"):
        self.model = model
        self.tier = tier

    def recommend(self, agent, shape, cfg, store, *, provider=None, **kw):
        return {
            "tier_id": self.tier,
            "model": self.model,
            "provider": provider or "claude",
            "rationale": (f"{self.tier}: LCB q_lo=0.82 >= baseline mean 0.80 "
                          "- 0.05 tol; cheapest admitted tier"),
            "cost_delta": -0.0114,
            "reasons": _fake_reasons(self.tier),
        }

    def inspect(self, agent, shape, cfg, store, *, provider=None, **kw):
        return {
            "baseline_tier": "opus",
            "tiers": [
                {"tier_id": "haiku", "mean": 0.55, "q_lo": 0.41, "n": 3,
                 "drop_ci": [0.06, 0.31], "admit": False},
                {"tier_id": "sonnet", "mean": 0.84, "q_lo": 0.70, "n": 12,
                 "drop_ci": [-0.02, 0.14], "admit": True},
                {"tier_id": "opus", "mean": 0.80, "q_lo": 0.62, "n": 20,
                 "drop_ci": [0.0, 0.0], "admit": None},
            ],
            "widest_gating_cell": {
                "cell_key": "claude::polecat::implement::sonnet",
                "tier_id": "sonnet",
                "ci_half_width": 0.14,
            },
        }


class _FakeCfg:
    provider = "claude"

    def default_shape(self, agent):
        return "implement"


def _install_fake(monkeypatch, engine=None):
    """Wire a fake engine + state into the standalone cli module."""
    eng = engine or FakeEngine()
    monkeypatch.setattr(cli, "ENGINE", eng, raising=False)
    monkeypatch.setattr(cli, "_load_engine", lambda: eng, raising=False)
    monkeypatch.setattr(
        cli, "build_state",
        lambda **kw: cli.State(cfg=_FakeCfg(), store=object(), provider="claude"),
        raising=False,
    )
    return eng


def _run(*argv):
    """Invoke cli.main with captured stdout; return (rc, text)."""
    buf = io.StringIO()
    rc = cli.main(list(argv), out=buf)
    return rc, buf.getvalue()


# --------------------------------------------------------------------------- #
# advise
# --------------------------------------------------------------------------- #

def test_advise_text_is_wellformed(monkeypatch):
    _install_fake(monkeypatch)
    rc, text = _run("advise", "polecat", "implement")
    assert rc == 0
    # recommended tier + model + a rationale + per-tier cost differential
    assert "recommend: sonnet" in text
    assert "claude-sonnet-4-5" in text
    assert "rationale:" in text
    assert "cost differential" in text
    # cost diffs for each roster tier appear, with a sign + dollar
    assert "haiku" in text and "sonnet" in text and "opus" in text
    assert "-$0.0114" in text  # recommended tier's diff vs baseline
    assert "<- recommended" in text


def test_advise_json_emits_reasons(monkeypatch):
    _install_fake(monkeypatch)
    rc, text = _run("advise", "polecat", "implement", "--json")
    assert rc == 0
    obj = json.loads(text)
    assert obj["tier_id"] == "sonnet"
    assert obj["model"] == "claude-sonnet-4-5"
    # the audit surface: the structured reasons object, verbatim
    assert "reasons" in obj
    assert obj["reasons"]["candidates"][1]["tier_id"] == "sonnet"
    assert obj["reasons"]["candidates"][1]["q_lo"] == 0.82
    assert obj["reasons"]["eval_flag"] is False


# --------------------------------------------------------------------------- #
# inspect
# --------------------------------------------------------------------------- #

def test_inspect_text_shows_posteriors_and_widest_cell(monkeypatch):
    _install_fake(monkeypatch)
    rc, text = _run("inspect", "polecat", "implement")
    assert rc == 0
    assert "baseline tier*: opus" in text
    # per-tier posterior: mean + q_lo + quality-drop CI columns
    assert "mean" in text and "q_lo" in text and "quality-drop 95% CI" in text
    assert "[0.060, 0.310]" in text  # haiku drop CI rendered
    assert "admit" in text and "reject" in text  # gate decisions shown
    # the next-eval pointer names the widest gating cell
    assert "next eval" in text
    assert "claude::polecat::implement::sonnet" in text
    assert "0.140" in text  # CI half-width


def test_inspect_json(monkeypatch):
    _install_fake(monkeypatch)
    rc, text = _run("inspect", "polecat", "implement", "--json")
    assert rc == 0
    obj = json.loads(text)
    assert obj["baseline_tier"] == "opus"
    assert len(obj["tiers"]) == 3
    assert obj["widest_gating_cell"]["tier_id"] == "sonnet"


# --------------------------------------------------------------------------- #
# apply — dry-run computes a change but writes nothing
# --------------------------------------------------------------------------- #

FLAT_AGENT_TOML = (
    "scope = \"rig\"\n"
    "wake_mode = \"fresh\"\n"
    "idle_timeout = \"2h\"\n"
    "max_active_sessions = 5\n"
)

CITY_TOML_WITH_AGENT_BLOCK = (
    "[workspace]\n"
    "provider = \"claude\"\n"
    "\n"
    "[agent_defaults]\n"
    "default_sling_formula = \"mol-pack-default\"\n"
    "\n"
    "[[agent]]\n"
    "name = \"mayor\"\n"
    "scope = \"city\"\n"
    "\n"
    "[[agent]]\n"
    "name = \"polecat\"\n"
    "scope = \"rig\"\n"
    "max_active_sessions = 5\n"
)


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_apply_dry_run_computes_without_writing(monkeypatch, tmp_path):
    _install_fake(monkeypatch)
    agent_toml = _write(tmp_path, "agent.toml", FLAT_AGENT_TOML)
    before = agent_toml.read_text()
    monkeypatch.setenv("ADVISOR_AGENT_TOML", str(agent_toml))

    rc, text = _run("apply", "polecat", "--shape", "implement", "--dry-run")
    assert rc == 0
    assert "DRY-RUN" in text
    assert "claude-sonnet-4-5" in text
    assert "recommended tier: sonnet" in text
    # nothing written, no backup created
    assert agent_toml.read_text() == before
    assert not list(tmp_path.glob("*advisor-bak*"))


# --------------------------------------------------------------------------- #
# apply — writes the model field into a TEMP flat agent.toml and backs it up
# --------------------------------------------------------------------------- #

def test_apply_writes_flat_agent_toml_and_backs_up(monkeypatch, tmp_path):
    _install_fake(monkeypatch)
    agent_toml = _write(tmp_path, "agent.toml", FLAT_AGENT_TOML)
    monkeypatch.setenv("ADVISOR_AGENT_TOML", str(agent_toml))

    rc, text = _run("apply", "polecat", "--shape", "implement")
    assert rc == 0
    assert "applied:" in text and "backup:" in text

    # model field now set to the recommended model
    updated = agent_toml.read_text()
    assert 'model = "claude-sonnet-4-5"' in updated
    # original keys preserved (format-preserving surgical edit)
    assert 'scope = "rig"' in updated
    assert 'max_active_sessions = 5' in updated

    # a timestamped backup of the ORIGINAL content exists
    baks = list(tmp_path.glob("*advisor-bak*"))
    assert len(baks) == 1
    assert baks[0].read_text() == FLAT_AGENT_TOML


def test_apply_writes_into_matching_agent_block(monkeypatch, tmp_path):
    """`model` must land inside the [[agent]] name="polecat" block, not mayor's."""
    _install_fake(monkeypatch)
    city = _write(tmp_path, "city.toml", CITY_TOML_WITH_AGENT_BLOCK)
    monkeypatch.setenv("ADVISOR_AGENT_TOML", str(city))

    rc, text = _run("apply", "polecat", "--shape", "implement")
    assert rc == 0
    updated = city.read_text()

    # exactly one model line, and it is in the polecat block (after its name)
    assert updated.count("model = ") == 1
    polecat_idx = updated.index('name = "polecat"')
    mayor_idx = updated.index('name = "mayor"')
    model_idx = updated.index('model = "claude-sonnet-4-5"')
    assert model_idx > polecat_idx        # inside polecat block
    assert not (mayor_idx < model_idx < polecat_idx)  # not in mayor block
    # backup made
    assert len(list(tmp_path.glob("*advisor-bak*"))) == 1


# --------------------------------------------------------------------------- #
# apply — idempotence + no-op refusal
# --------------------------------------------------------------------------- #

def test_apply_is_idempotent(monkeypatch, tmp_path):
    _install_fake(monkeypatch)
    agent_toml = _write(tmp_path, "agent.toml", FLAT_AGENT_TOML)
    monkeypatch.setenv("ADVISOR_AGENT_TOML", str(agent_toml))

    rc1, _ = _run("apply", "polecat", "--shape", "implement")
    assert rc1 == 0
    after_first = agent_toml.read_text()
    assert after_first.count("model = ") == 1

    # second apply is a no-op refusal (model already == recommendation)
    rc2, text2 = _run("apply", "polecat", "--shape", "implement")
    assert rc2 == 3
    assert "refused" in text2.lower()
    # file unchanged on the refusal; still exactly one model line
    assert agent_toml.read_text() == after_first


def test_apply_refuses_noop_when_model_already_set(monkeypatch, tmp_path):
    _install_fake(monkeypatch)
    preset = FLAT_AGENT_TOML + 'model = "claude-sonnet-4-5"\n'
    agent_toml = _write(tmp_path, "agent.toml", preset)
    monkeypatch.setenv("ADVISOR_AGENT_TOML", str(agent_toml))

    rc, text = _run("apply", "polecat", "--shape", "implement")
    assert rc == 3
    assert "refused" in text.lower()
    assert "no change" in text.lower()
    # untouched, and no backup written on a refusal
    assert agent_toml.read_text() == preset
    assert not list(tmp_path.glob("*advisor-bak*"))


def test_apply_replaces_existing_different_model(monkeypatch, tmp_path):
    """A different existing model is replaced (not duplicated)."""
    _install_fake(monkeypatch)
    preset = FLAT_AGENT_TOML + 'model = "claude-opus-4-8"\n'
    agent_toml = _write(tmp_path, "agent.toml", preset)
    monkeypatch.setenv("ADVISOR_AGENT_TOML", str(agent_toml))

    rc, text = _run("apply", "polecat", "--shape", "implement")
    assert rc == 0
    updated = agent_toml.read_text()
    assert updated.count("model = ") == 1
    assert 'model = "claude-sonnet-4-5"' in updated
    assert "claude-opus-4-8" not in updated
    assert "current model:    claude-opus-4-8" in text


# --------------------------------------------------------------------------- #
# apply — unresolvable config explains cleanly
# --------------------------------------------------------------------------- #

def test_apply_unresolvable_config_explained(monkeypatch, tmp_path):
    _install_fake(monkeypatch)
    # point at a missing file → resolver raises ConfigResolveError → rc 2
    monkeypatch.setenv("ADVISOR_AGENT_TOML", str(tmp_path / "nope.toml"))
    rc, text = _run("apply", "polecat", "--shape", "implement")
    assert rc == 2
    assert "apply:" in text
    assert "missing file" in text.lower() or "could not resolve" in text.lower()


def test_apply_block_requires_named_agent(monkeypatch, tmp_path):
    """city.toml with no matching [[agent]] and no defaults → cannot resolve."""
    _install_fake(monkeypatch)
    # remove agent_defaults so there's no fallback scope, and no polecat block
    city_text = (
        "[workspace]\nprovider = \"claude\"\n\n"
        "[[agent]]\nname = \"mayor\"\nscope = \"city\"\n"
    )
    city = _write(tmp_path, "city.toml", city_text)
    monkeypatch.setenv("ADVISOR_AGENT_TOML", str(city))
    rc, text = _run("apply", "ghostagent", "--shape", "implement")
    assert rc == 2
    assert "apply:" in text


def test_apply_falls_back_to_agent_defaults(monkeypatch, tmp_path):
    """No [[agent]] match but an [agent_defaults] table exists → write there."""
    _install_fake(monkeypatch)
    city_text = (
        "[workspace]\nprovider = \"claude\"\n\n"
        "[agent_defaults]\ndefault_sling_formula = \"mol-x\"\n"
    )
    city = _write(tmp_path, "city.toml", city_text)
    monkeypatch.setenv("ADVISOR_AGENT_TOML", str(city))
    rc, text = _run("apply", "anyagent", "--shape", "implement")
    assert rc == 0
    updated = city.read_text()
    assert 'model = "claude-sonnet-4-5"' in updated
    # landed inside [agent_defaults], after its header
    assert updated.index("model = ") > updated.index("[agent_defaults]")
    assert "[agent_defaults]" in text or "agent_defaults" in text


# --------------------------------------------------------------------------- #
# apply — resolve default shape when --shape omitted
# --------------------------------------------------------------------------- #

def test_apply_uses_default_shape_when_omitted(monkeypatch, tmp_path):
    _install_fake(monkeypatch)  # _FakeCfg.default_shape -> "implement"
    agent_toml = _write(tmp_path, "agent.toml", FLAT_AGENT_TOML)
    monkeypatch.setenv("ADVISOR_AGENT_TOML", str(agent_toml))
    rc, text = _run("apply", "polecat")  # no --shape
    assert rc == 0
    assert "shape=implement" in text
    assert 'model = "claude-sonnet-4-5"' in agent_toml.read_text()


# --------------------------------------------------------------------------- #
# real engine smoke check (skipped if the sibling bead hasn't landed)
# --------------------------------------------------------------------------- #

def test_real_engine_importable_smoke():
    """If the sibling engine exists, its recommend/inspect are callable shapes.

    Skipped (not failed) when the engine bead hasn't landed yet, so this file is
    green in isolation but exercises the real contract once it is present.
    """
    try:
        from modeladvisor import engine as real_engine  # noqa: F401
        from modeladvisor import config as real_config  # noqa: F401
        from modeladvisor import store as real_store  # noqa: F401
    except Exception as e:  # ModuleNotFoundError etc.
        pytest.skip(f"sibling engine not present yet: {e}")

    assert hasattr(real_engine, "recommend")
    assert hasattr(real_engine, "inspect")
    assert hasattr(real_config, "default_config") or hasattr(
        real_config, "load_config"
    )
    assert hasattr(real_store, "CellStore")


# --------------------------------------------------------------------------- #
# real engine integration — drive cli.main against the actual engine
# (skipped when the sibling bead hasn't landed).  These lock the CLI⇄engine
# key-name alignment so a rename on either side is caught.
# --------------------------------------------------------------------------- #

def _have_real_engine():
    try:
        import modeladvisor.engine  # noqa: F401
        import modeladvisor.config  # noqa: F401
        import modeladvisor.store  # noqa: F401
        return True
    except Exception:
        return False


real_engine = pytest.mark.skipif(
    not _have_real_engine(), reason="sibling engine not present yet"
)


@real_engine
def test_real_advise_text_and_json(monkeypatch, tmp_path):
    # cold-start (no telemetry) → engine recommends the baseline tier; the CLI
    # must render a tier + rationale + per-tier cost diffs, and valid JSON.
    monkeypatch.setenv("ADVISOR_TELEMETRY_DIR", str(tmp_path))  # empty → cold-start
    rc, text = _run("advise", "polecat", "implement")
    assert rc == 0
    assert "recommend:" in text and "rationale:" in text
    assert "cost differential" in text
    assert "<- recommended" in text

    rc, js = _run("advise", "polecat", "implement", "--json")
    assert rc == 0
    obj = json.loads(js)  # must be valid JSON
    assert obj["tier_id"]  # a concrete recommended tier
    assert "candidates" in obj["reasons"]


@real_engine
def test_real_inspect_text_and_json(monkeypatch, tmp_path):
    monkeypatch.setenv("ADVISOR_TELEMETRY_DIR", str(tmp_path))
    rc, text = _run("inspect", "polecat", "implement")
    assert rc == 0
    assert "baseline tier*:" in text
    # the gate column + quality-drop CI + widest-cell pointer all render
    assert "quality-drop 95% CI" in text
    assert "baseline" in text
    assert "next eval" in text

    rc, js = _run("inspect", "polecat", "implement", "--json")
    assert rc == 0
    obj = json.loads(js)
    assert obj["tiers"] and obj["baseline_tier"]
    assert "widest_gating_cell" in obj


@real_engine
def test_real_apply_lifecycle_on_temp_config(monkeypatch, tmp_path):
    # full apply lifecycle against the real engine, on a TEMP flat agent.toml.
    monkeypatch.setenv("ADVISOR_TELEMETRY_DIR", str(tmp_path))
    agent_toml = _write(tmp_path, "agent.toml", FLAT_AGENT_TOML)
    monkeypatch.setenv("ADVISOR_AGENT_TOML", str(agent_toml))

    # dry-run writes nothing
    rc, text = _run("apply", "polecat", "--shape", "implement", "--dry-run")
    assert rc == 0 and "DRY-RUN" in text
    assert agent_toml.read_text() == FLAT_AGENT_TOML

    # real apply sets a model + backs up
    rc, text = _run("apply", "polecat", "--shape", "implement")
    assert rc == 0 and "applied:" in text and "backup:" in text
    after = agent_toml.read_text()
    assert "model = " in after and after.count("model = ") == 1
    assert len(list(tmp_path.glob("*advisor-bak*"))) == 1

    # second apply is the idempotent no-op refusal
    rc, text = _run("apply", "polecat", "--shape", "implement")
    assert rc == 3 and "refused" in text.lower()
    assert agent_toml.read_text() == after
