"""model-advisor CLI — the operator-facing surfaces.

Three subcommands, mirroring the read-side pack's ``bin/op``/``bin/outline``
ergonomics (see ``packs/ast-lens/bin/``) and the surfaces specified in
``docs/DESIGN.md`` §6 + the integration verdict in
``docs/INTEGRATION-FEASIBILITY.md`` (A):

  advise  <agent> <shape> [--json]
      Call ``engine.recommend(agent, shape, cfg, store)`` and print the
      recommended tier, a human rationale, and the cost differential vs every
      roster tier.  ``--json`` emits the structured ``reasons`` audit object.

  inspect <agent> <shape> [--json]
      Call ``engine.inspect(agent, shape, cfg, store)`` and show each tier's
      posterior (mean + credible interval on the quality DROP vs baseline),
      then name the widest *gating* cell as the next eval to run.

  apply   <agent> [--shape S] [--city PATH] [--rig NAME] [--dry-run]
      Compute the recommendation, then SET that agent's default model in gc
      config by writing/updating the ``model = "<model>"`` field for the agent
      (per the integration verdict: ``[[agent]]`` / ``[agent_defaults].model``,
      the field gc carries into ``GC_AGENT_MODEL``).  Backs the file up first,
      is idempotent, supports ``--dry-run``, prints a clear before/after, and
      refuses cleanly on a no-op (recommended tier == current) or when the
      agent/config cannot be resolved.

This module owns *only* the CLI.  The engine, store, and config are built by a
sibling bead under the shared contract::

    engine.recommend(agent, shape, cfg, store) -> {tier_id, model, rationale,
                                                   cost_delta, reasons, ...}
    engine.inspect(agent, shape, cfg, store)   -> {...}
    store.CellStore(...)        # the posterior cell store
    config.load_config(path)    # loads advisor.toml
    config.default_config()     # built-in defaults (no advisor.toml present)

The CLI imports and drives those.  It is deliberately tolerant of the engine
not being present yet (cold dev / test bootstrap): the heavy imports are lazy
and tests may inject a stub via :data:`ENGINE` / :data:`build_state`.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

PROG = "advisor"

# Cell key = provider::agent::shape::tier_id  (shared contract).
CELL_KEY = "{provider}::{agent}::{shape}::{tier_id}"


# --------------------------------------------------------------------------- #
# Engine / config / store wiring (lazy + injectable)
# --------------------------------------------------------------------------- #
#
# ``ENGINE`` and ``build_state`` are module-level hooks so tests can drive the
# CLI without the sibling engine present.  In production they resolve to the
# real ``modeladvisor.engine`` / ``modeladvisor.config`` / ``modeladvisor.store``.

ENGINE: Any = None  # set by _load_engine(); tests may monkeypatch.


def _load_engine() -> Any:
    """Import and return the real engine module, caching it in :data:`ENGINE`.

    Kept lazy so ``import modeladvisor.cli`` never hard-fails if the sibling
    engine bead has not landed yet (the CLI can still be imported, its argparse
    introspected, and ``apply``'s config editor unit-tested in isolation).
    """
    global ENGINE
    if ENGINE is not None:
        return ENGINE
    from modeladvisor import engine as _engine  # local import: see docstring

    ENGINE = _engine
    return ENGINE


@dataclass
class State:
    """Everything ``recommend``/``inspect`` need, resolved from disk/config."""

    cfg: Any
    store: Any
    provider: str


def build_state(
    *,
    advisor_toml: Optional[str] = None,
    telemetry_dir: Optional[str] = None,
    provider: Optional[str] = None,
) -> State:
    """Resolve config + cell store for a CLI invocation.

    Config (DESIGN §8 "config files the engine reads"): an explicit
    ``advisor.toml`` (``--config`` or ``$ADVISOR_TOML``) else the pack/city
    default config.

    Cell store (DESIGN §5.5 "the JSONL is the source of truth; the cache is
    rebuildable"): from the telemetry dir (``$ADVISOR_TELEMETRY_DIR`` or
    ``./.beads/telemetry``) we *rebuild* from ``invocations.jsonl`` when it is
    present (truth), else *load* the ``advisor-cells.json`` cache when present,
    else fall back to a conservative ``cold_start`` store (DESIGN §3.2 — the
    advisor must never block; worst case it recommends the baseline tier).

    Tests monkeypatch this wholesale to inject a fake ``State``; production uses
    the sibling ``config``/``store`` modules.
    """
    from modeladvisor import config as _config
    from modeladvisor import store as _store

    advisor_toml = advisor_toml or os.environ.get("ADVISOR_TOML")
    if advisor_toml and os.path.exists(advisor_toml):
        cfg = _config.load_config(advisor_toml)
    else:
        cfg = _config.default_config()

    telemetry_dir = (
        telemetry_dir
        or os.environ.get("ADVISOR_TELEMETRY_DIR")
        or os.path.join(os.getcwd(), ".beads", "telemetry")
    )
    jsonl = os.path.join(telemetry_dir, "invocations.jsonl")
    cache = os.path.join(telemetry_dir, "advisor-cells.json")
    if os.path.exists(jsonl):
        store = _store.CellStore.rebuild(cfg, jsonl)
    elif os.path.exists(cache):
        store = _store.CellStore.load(cfg, cache)
    else:
        store = _store.CellStore.cold_start(cfg)

    prov = (
        provider
        or getattr(cfg, "default_provider", None)
        or "claude"
    )
    return State(cfg=cfg, store=store, provider=prov)


# --------------------------------------------------------------------------- #
# Small output helpers
# --------------------------------------------------------------------------- #

def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dict-or-object recommendation/inspect result.

    The engine returns plain dicts per the contract, but being lenient about
    attribute-style access keeps the CLI robust to a dataclass result too.
    """
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _fmt_money(x: Optional[float]) -> str:
    """Format a dollar cost differential with a sign, e.g. ``-$0.0114``."""
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    sign = "-" if v < 0 else "+"
    return f"{sign}${abs(v):.4f}"


def _fmt_ci(ci: Any) -> str:
    """Format a credible interval as ``[lo, hi]`` (+ ``*`` if it exceeds tol).

    Accepts either a 2-tuple/list ``(lo, hi)`` or the engine's richer dict
    ``{lo, hi, mean, q_tol, exceeds_tol}`` (DESIGN §6.2 — the credible interval
    on the quality *drop* vs baseline).  A trailing ``*`` flags an interval that
    breaches the tolerance (i.e. why a cheaper tier can't be admitted).
    """
    if not ci:
        return "[n/a]"
    if isinstance(ci, Mapping):
        lo = ci.get("lo")
        hi = ci.get("hi")
        if lo is None or hi is None:
            return str(ci)
        flag = "*" if ci.get("exceeds_tol") else ""
        return f"[{float(lo):.3f}, {float(hi):.3f}]{flag}"
    try:
        lo, hi = ci
        return f"[{float(lo):.3f}, {float(hi):.3f}]"
    except (TypeError, ValueError):
        return str(ci)


def _num(x: Any, fmt: str = "{:.3f}") -> str:
    try:
        return fmt.format(float(x))
    except (TypeError, ValueError):
        return "n/a" if x is None else str(x)


# --------------------------------------------------------------------------- #
# advise
# --------------------------------------------------------------------------- #

def cmd_advise(args: argparse.Namespace, out: io.TextIOBase) -> int:
    engine = _load_engine()
    state = _resolve_state(args)

    rec = engine.recommend(
        args.agent, args.shape, state.cfg, state.store, provider=state.provider
    )

    if args.json:
        # The audit surface: emit the structured ``reasons`` verbatim, with the
        # top-line decision fields alongside for a self-contained record.
        payload = {
            "agent": args.agent,
            "shape": args.shape,
            "provider": _get(rec, "provider", state.provider),
            "tier_id": _get(rec, "tier_id"),
            "model": _get(rec, "model"),
            "cost_delta": _get(rec, "cost_delta"),
            "reasons": _get(rec, "reasons", {}),
        }
        json.dump(payload, out, indent=2, default=str)
        out.write("\n")
        return 0

    tier = _get(rec, "tier_id", "?")
    model = _get(rec, "model")
    rationale = _get(rec, "rationale", "(no rationale provided)")

    out.write(f"advise {args.agent} {args.shape}\n")
    head = f"  recommend: {tier}"
    if model:
        head += f"  ({model})"
    out.write(head + "\n")
    out.write(f"  rationale: {rationale}\n")

    # Cost differential vs each roster tier (and vs baseline).  Prefer an
    # explicit per-candidate breakdown from ``reasons``; fall back to the
    # scalar ``cost_delta`` vs baseline.
    reasons = _get(rec, "reasons", {}) or {}
    cands = _get(reasons, "candidates", None)
    if cands:
        out.write("  cost differential vs each tier:\n")
        for c in cands:
            ctier = _get(c, "tier_id", "?")
            cdiff = _get(c, "cost_diff", _get(c, "cost_delta"))
            marker = " <- recommended" if ctier == tier else ""
            extra = []
            if _get(c, "q_lo") is not None:
                extra.append(f"q_lo={_num(_get(c, 'q_lo'))}")
            if _get(c, "exp_loss") is not None:
                extra.append(f"E[L]={_fmt_money(_get(c, 'exp_loss'))}")
            suffix = ("  " + " ".join(extra)) if extra else ""
            out.write(
                f"    {ctier:<12} {_fmt_money(cdiff)}{suffix}{marker}\n"
            )
    else:
        out.write(
            f"  cost differential vs baseline: {_fmt_money(_get(rec, 'cost_delta'))}\n"
        )

    if _get(reasons, "eval_flag"):
        out.write("  note: this cell wants an eval (posterior CI is wide)\n")
    return 0


# --------------------------------------------------------------------------- #
# inspect
# --------------------------------------------------------------------------- #

def cmd_inspect(args: argparse.Namespace, out: io.TextIOBase) -> int:
    engine = _load_engine()
    state = _resolve_state(args)

    info = engine.inspect(
        args.agent, args.shape, state.cfg, state.store, provider=state.provider
    )

    if args.json:
        json.dump(_as_jsonable(info), out, indent=2, default=str)
        out.write("\n")
        return 0

    out.write(f"inspect {args.agent} {args.shape}\n")
    baseline = _get(info, "baseline_tier")
    if baseline:
        out.write(f"  baseline tier*: {baseline}\n")

    tiers = _get(info, "tiers", []) or []
    if tiers:
        out.write(
            "  {:<12} {:>6} {:>6} {:>5} {:<22} {}\n".format(
                "tier", "mean", "q_lo", "n", "quality-drop 95% CI", "gate"
            )
        )
        for t in tiers:
            tid = _get(t, "tier_id", "?")
            mean = _num(_get(t, "mean"))
            q_lo = _num(_get(t, "q_lo"))
            n = _get(t, "n", 0)
            drop_ci = _fmt_ci(_get(t, "quality_drop_ci", _get(t, "drop_ci")))
            out.write(
                "  {:<12} {:>6} {:>6} {:>5} {:<22} {}\n".format(
                    tid, mean, q_lo, n, drop_ci, _gate_label(t, tid, baseline)
                )
            )

    widest = _get(info, "widest_gating_cell") or _get(info, "widest_cell")
    if widest:
        cell = _get(widest, "cell_key", _get(widest, "tier_id", "?"))
        hw = _get(widest, "ci_halfwidth",
                  _get(widest, "ci_half_width", _get(widest, "half_width")))
        out.write(
            f"  next eval (widest gating cell): {cell}"
            + (f" (CI half-width {_num(hw)})" if hw is not None else "")
            + "\n"
        )
        rationale = _get(widest, "rationale")
        if rationale:
            out.write(f"    {rationale}\n")
    else:
        out.write("  next eval: none — no cell is gating a downgrade\n")
    return 0


def _gate_label(tier: Any, tid: str, baseline: Optional[str]) -> str:
    """Render a tier's gate decision for the inspect table.

    The real engine marks the baseline with ``role == 'baseline'`` and each
    candidate downgrade with ``admitted`` (bool).  ``admit`` is accepted as a
    fallback alias so a simpler engine/stub still renders correctly.
    """
    role = _get(tier, "role")
    if role == "baseline" or tid == baseline:
        return "baseline"
    admit = _get(tier, "admitted", _get(tier, "admit"))
    if admit is True:
        return "admit"
    if admit is False:
        return "reject"
    return "-"


def _as_jsonable(info: Any) -> Any:
    """Best-effort coerce an inspect result to something ``json.dump`` accepts."""
    if isinstance(info, Mapping):
        return info
    # dataclass / object → its __dict__ if present
    d = getattr(info, "__dict__", None)
    return d if d is not None else info


# --------------------------------------------------------------------------- #
# apply  — the v1 "actually choose the model" action (per AGENT)
# --------------------------------------------------------------------------- #

def cmd_apply(args: argparse.Namespace, out: io.TextIOBase) -> int:
    engine = _load_engine()
    state = _resolve_state(args)

    shape = args.shape
    if not shape:
        shape = _default_shape_for(state.cfg, args.agent)
    if not shape:
        out.write(
            f"apply: cannot resolve a shape for agent '{args.agent}'. "
            "Pass --shape <lookup|implement|judge|review|patrol>.\n"
        )
        return 2

    rec = engine.recommend(
        args.agent, shape, state.cfg, state.store, provider=state.provider
    )
    new_model = _get(rec, "model")
    new_tier = _get(rec, "tier_id")
    if not new_model:
        out.write(
            "apply: the engine did not return a concrete model for "
            f"{args.agent}/{shape}; nothing to apply.\n"
        )
        return 2

    # Resolve the gc config file + the scope (flat agent.toml / [[agent]] /
    # [agent_defaults]) that owns this agent's model field.
    try:
        target = resolve_agent_config(
            args.agent, city=args.city, rig=args.rig
        )
    except ConfigResolveError as e:
        out.write(f"apply: {e}\n")
        return 2

    current = read_model_field(target)

    out.write(f"apply {args.agent} (shape={shape})\n")
    out.write(f"  config: {target.path}\n")
    out.write(f"  scope:  {target.describe()}\n")
    out.write(f"  recommended tier: {new_tier}  model: {new_model}\n")
    out.write(f"  current model:    {current if current is not None else '(unset)'}\n")

    # Refuse the no-op: recommended model already in effect.
    if current is not None and current == new_model:
        out.write(
            "  refused: recommended model equals the current model "
            f"('{new_model}') — no change to apply.\n"
        )
        return 3

    if args.dry_run:
        out.write(
            f"  DRY-RUN: would set model = \"{new_model}\" "
            f"(was {current if current is not None else 'unset'}); no file written.\n"
        )
        return 0

    backup = backup_file(target.path)
    set_model_field(target, new_model)

    out.write(f"  backup: {backup}\n")
    out.write(
        f"  applied: model {current if current is not None else 'unset'} -> \"{new_model}\"\n"
    )
    out.write(
        "  note: agent picks up the new model on next session "
        "(gc exports it as GC_AGENT_MODEL at spawn).\n"
    )
    return 0


def _default_shape_for(cfg: Any, agent: str) -> Optional[str]:
    """Resolve an agent's canonical default shape from config, if any.

    DESIGN §2.2 gives each agent a canonical shape set; the engine/config own
    the real mapping.  We probe a couple of plausible accessors and otherwise
    return ``None`` (caller then requires ``--shape``).
    """
    # The real config exposes ``canonical_shapes_for(agent) -> tuple[str, ...]``
    # (DESIGN §2.2); the first canonical shape is the agent's default.  Probe a
    # few plausible accessors so the CLI tolerates config-shape drift.
    for attr in ("default_shape_for", "default_shape", "canonical_shapes_for"):
        fn = getattr(cfg, attr, None)
        if callable(fn):
            try:
                s = fn(agent)
            except Exception:
                continue
            if isinstance(s, str) and s:
                return s
            if isinstance(s, (list, tuple)) and s:
                return s[0]
    shapes = getattr(cfg, "agent_shapes", None) or getattr(cfg, "canonical_shapes", None)
    if isinstance(shapes, Mapping):
        v = shapes.get(agent)
        if isinstance(v, str):
            return v
        if isinstance(v, (list, tuple)) and v:
            return v[0]
    return None


# --------------------------------------------------------------------------- #
# gc config editing  (the heart of `apply`)
# --------------------------------------------------------------------------- #
#
# gc accepts a ``model`` field in three shapes (INTEGRATION verdict (A)):
#
#   1. a *flat* per-agent ``agent.toml`` (gastown style: the file IS the agent;
#      ``model`` is a top-level key);
#   2. an ``[[agent]]`` array-of-tables entry inside a pack/city toml, matched
#      by ``name = "<agent>"``;
#   3. an ``[agent_defaults]`` table (the default model for all agents).
#
# We perform a *surgical, format-preserving text edit* — insert or replace a
# single ``model = "..."`` line in the correct scope — rather than round-trip
# the TOML (which would drop comments/formatting and needs a writer lib the
# venv doesn't ship).  This is the same approach gc's own ``doImportAdd`` takes.

_MODEL_LINE = re.compile(r'^(?P<indent>[ \t]*)model[ \t]*=.*$', re.MULTILINE)


class ConfigResolveError(Exception):
    """Raised when the agent's config file/scope can't be resolved."""


@dataclass
class ConfigTarget:
    """Where (file + scope) an agent's ``model`` field lives / should live."""

    path: str
    kind: str  # "flat" | "agent_block" | "agent_defaults"
    # For agent_block/agent_defaults we keep the [byte] span of the table body
    # so edits stay inside the right table.
    span: Optional[tuple] = None  # (start, end) char offsets into the file text
    agent: Optional[str] = None

    def describe(self) -> str:
        if self.kind == "flat":
            return f"flat agent.toml (top-level model for '{self.agent}')"
        if self.kind == "agent_block":
            return f"[[agent]] name = \"{self.agent}\""
        if self.kind == "agent_defaults":
            return "[agent_defaults]"
        return self.kind


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def resolve_agent_config(
    agent: str,
    *,
    city: Optional[str] = None,
    rig: Optional[str] = None,
) -> ConfigTarget:
    """Locate the gc config file + scope that owns ``agent``'s model field.

    Resolution (first hit wins):

    1. ``$ADVISOR_AGENT_TOML`` — explicit file override (used by tests). If it
       contains an ``[[agent]]`` block named ``agent`` we target that; if it has
       an ``[agent_defaults]`` table we target that; otherwise it is treated as
       a flat per-agent ``agent.toml``.
    2. A flat ``<city>/.gc/system/packs/<rig?>/agents/<agent>/agent.toml`` (the
       gastown layout). ``city`` defaults to ``$GC_CITY`` / cwd; ``rig`` narrows
       the pack search.
    3. A ``[[agent]] name = "<agent>"`` block, or an ``[agent_defaults]`` table,
       inside ``<city>/city.toml``.

    Raises :class:`ConfigResolveError` if nothing resolves.
    """
    override = os.environ.get("ADVISOR_AGENT_TOML")
    if override:
        if not os.path.exists(override):
            raise ConfigResolveError(
                f"ADVISOR_AGENT_TOML points at a missing file: {override}"
            )
        return _classify_config_file(override, agent)

    city = city or os.environ.get("GC_CITY") or os.getcwd()

    # (2) flat gastown agent.toml — search common pack roots.
    candidates = []
    pack_roots = []
    sys_packs = os.path.join(city, ".gc", "system", "packs")
    if rig:
        pack_roots.append(os.path.join(sys_packs, rig))
    if os.path.isdir(sys_packs):
        for entry in sorted(os.listdir(sys_packs)):
            p = os.path.join(sys_packs, entry)
            if os.path.isdir(p) and p not in pack_roots:
                pack_roots.append(p)
    for root in pack_roots:
        candidates.append(os.path.join(root, "agents", agent, "agent.toml"))
    for cand in candidates:
        if os.path.exists(cand):
            return ConfigTarget(path=cand, kind="flat", agent=agent)

    # (3) city.toml [[agent]] / [agent_defaults]
    city_toml = os.path.join(city, "city.toml")
    if os.path.exists(city_toml):
        try:
            return _classify_config_file(city_toml, agent, require_agent=True)
        except ConfigResolveError:
            pass

    raise ConfigResolveError(
        f"could not resolve a config file for agent '{agent}'. "
        f"Looked for a flat agent.toml under {sys_packs} and an [[agent]]/"
        f"[agent_defaults] block in {city_toml}. "
        "Set ADVISOR_AGENT_TOML or pass --city/--rig."
    )


def _classify_config_file(
    path: str, agent: str, require_agent: bool = False
) -> ConfigTarget:
    """Decide how ``agent``'s model is represented inside ``path``.

    Looks for an ``[[agent]]`` table whose ``name`` matches ``agent`` first,
    then an ``[agent_defaults]`` table, then falls back to treating the whole
    file as a flat agent.toml — but *only* when the file is not structurally a
    multi-agent config.  A file that carries ``[[agent]]`` blocks is a city /
    pack config: if no block matches and there is no ``[agent_defaults]`` table,
    treating the whole file as one agent's flat config would silently write the
    model into the wrong place, so we refuse instead.
    """
    text = _read(path)

    block = _find_agent_block(text, agent)
    if block is not None:
        return ConfigTarget(path=path, kind="agent_block", span=block, agent=agent)

    defaults = _find_table(text, "agent_defaults")
    if defaults is not None:
        return ConfigTarget(
            path=path, kind="agent_defaults", span=defaults, agent=agent
        )

    # No matching scope.  Is this a multi-agent (city/pack) config?
    has_agent_blocks = any(
        hdr.startswith("[[") and hdr.strip("[]").strip() == "agent"
        for hdr, _s, _e in _table_headers(text)
    )
    if require_agent or has_agent_blocks:
        raise ConfigResolveError(
            f"{path} has no [[agent]] name = \"{agent}\" block and no "
            "[agent_defaults] table to hold the model field"
        )
    return ConfigTarget(path=path, kind="flat", agent=agent)


def _table_headers(text: str):
    """Yield (header_name, header_start, body_start) for every top-level table.

    ``header_name`` is the raw bracket content (e.g. ``agent_defaults`` or
    ``[agent]`` for an array-of-tables — note double brackets keep their inner
    ``[agent]``).  We only need coarse boundaries to scope an edit, so a simple
    line scanner is sufficient and avoids a TOML round-trip.
    """
    for m in re.finditer(r'^[ \t]*(\[\[?[^\]\n]+\]\]?)[ \t]*$', text, re.MULTILINE):
        yield m.group(1), m.start(), m.end()


def _find_table(text: str, name: str) -> Optional[tuple]:
    """Return the (body_start, body_end) char span of the ``[name]`` table body."""
    headers = list(_table_headers(text))
    for i, (hdr, _hstart, hend) in enumerate(headers):
        inner = hdr.strip("[]").strip()
        if inner == name and not hdr.startswith("[["):
            body_start = hend
            body_end = headers[i + 1][1] if i + 1 < len(headers) else len(text)
            return (body_start, body_end)
    return None


def _find_agent_block(text: str, agent: str) -> Optional[tuple]:
    """Return the body span of the ``[[agent]]`` table whose ``name == agent``.

    Also matches a ``[agent.<name>]`` / ``[crew.<name>]`` / ``[workers.<name>]``
    style table (seen in ship.toml-style configs) keyed by the dotted name.
    """
    headers = list(_table_headers(text))
    for i, (hdr, _hstart, hend) in enumerate(headers):
        body_start = hend
        body_end = headers[i + 1][1] if i + 1 < len(headers) else len(text)
        body = text[body_start:body_end]
        inner = hdr.strip("[]").strip()
        is_array = hdr.startswith("[[")
        if is_array and inner == "agent":
            # match by `name = "<agent>"` inside the body
            nm = re.search(r'^[ \t]*name[ \t]*=[ \t]*["\']([^"\']+)["\']',
                           body, re.MULTILINE)
            if nm and nm.group(1) == agent:
                return (body_start, body_end)
        elif not is_array and "." in inner:
            # dotted table like [crew.Yatima] / [workers.builder] / [agent.foo]
            _prefix, _, dotted = inner.partition(".")
            if dotted == agent:
                return (body_start, body_end)
    return None


def read_model_field(target: ConfigTarget) -> Optional[str]:
    """Return the current ``model`` string for the target, or ``None`` if unset."""
    text = _read(target.path)
    region = text
    offset = 0
    if target.span is not None:
        region = text[target.span[0]:target.span[1]]
        offset = target.span[0]
    m = re.search(
        r'^[ \t]*model[ \t]*=[ \t]*["\']([^"\']*)["\']', region, re.MULTILINE
    )
    if m:
        return m.group(1)
    # also accept an unquoted value just in case
    m = re.search(r'^[ \t]*model[ \t]*=[ \t]*([^\s#]+)', region, re.MULTILINE)
    if m:
        return m.group(1).strip().strip('"\'')
    _ = offset  # (kept for symmetry / future precise editing)
    return None


def set_model_field(target: ConfigTarget, model: str) -> None:
    """Write ``model = "<model>"`` into the target's scope, in place.

    If a ``model`` line already exists in scope it is replaced; otherwise a new
    line is inserted at the top of the scope body — but for an ``[[agent]]``
    block we insert it just *after* the block's ``name = "..."`` line so the
    block stays readable (name first).  Formatting/comments elsewhere are
    preserved byte-for-byte.
    """
    text = _read(target.path)
    new_line = f'model = "{model}"'

    if target.span is None:
        # flat agent.toml: whole-file scope.
        updated = _replace_or_insert_top(text, new_line)
    else:
        start, end = target.span
        body = text[start:end]
        after_name = target.kind == "agent_block"
        new_body = _replace_or_insert_top(
            body, new_line, indent_from=body, after_name=after_name
        )
        updated = text[:start] + new_body + text[end:]

    _atomic_write(target.path, updated)


def _replace_or_insert_top(
    body: str,
    new_line: str,
    indent_from: Optional[str] = None,
    after_name: bool = False,
) -> str:
    """Replace an existing ``model =`` line in ``body`` else insert it.

    ``indent_from`` (the scope body) is used to detect an existing indentation
    convention so an inserted line matches sibling keys.  When ``after_name`` is
    set and the body has a ``name = "..."`` line (an ``[[agent]]`` block), the
    new line is inserted right after it; otherwise it goes at the top of the
    body (after any leading blank lines).
    """
    m = _MODEL_LINE.search(body)
    if m:
        indent = m.group("indent")
        return body[: m.start()] + f"{indent}{new_line}" + body[m.end():]

    indent = ""
    src = indent_from if indent_from is not None else body
    km = re.search(r'^(?P<indent>[ \t]+)\S', src, re.MULTILINE)
    if km:
        indent = km.group("indent")

    insert = f"{indent}{new_line}\n"

    if after_name:
        nm = re.search(r'^[ \t]*name[ \t]*=.*$', body, re.MULTILINE)
        if nm:
            # insert on the line after `name = "..."`
            pos = nm.end()
            # step past the newline that terminates the name line
            nl = body.find("\n", pos)
            pos = (nl + 1) if nl != -1 else len(body)
            return body[:pos] + insert + body[pos:]

    # Insert after a leading run of blank/whitespace lines so we sit at the top
    # of the actual content (and, for a scoped body, right under the header gap).
    lead = re.match(r'^([ \t]*\n)*', body)
    pos = lead.end() if lead else 0
    return body[:pos] + insert + body[pos:]


def backup_file(path: str) -> str:
    """Copy ``path`` to a timestamped ``.bak`` sibling and return its path."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = f"{path}.advisor-bak-{ts}"
    shutil.copy2(path, backup)
    return backup


def _atomic_write(path: str, text: str) -> None:
    """Write ``text`` to ``path`` atomically (temp file + os.replace)."""
    tmp = f"{path}.advisor-tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# state resolution shared by the subcommands
# --------------------------------------------------------------------------- #

def _resolve_state(args: argparse.Namespace) -> State:
    return build_state(
        advisor_toml=getattr(args, "config", None),
        telemetry_dir=getattr(args, "telemetry_dir", None),
        provider=getattr(args, "provider", None),
    )


# --------------------------------------------------------------------------- #
# argparse
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=PROG,
        description="model-advisor — recommend / inspect / apply the "
        "cost-minimal model tier per agent·shape.",
    )
    p.add_argument(
        "--config",
        help="path to advisor.toml (else $ADVISOR_TOML or built-in defaults)",
    )
    p.add_argument(
        "--telemetry-dir",
        dest="telemetry_dir",
        help="telemetry dir holding invocations.jsonl / advisor-cells.json "
        "(else $ADVISOR_TELEMETRY_DIR or ./.beads/telemetry)",
    )
    p.add_argument(
        "--provider",
        help="provider token for the cell key (else config / 'claude')",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser(
        "advise",
        help="recommend a tier for <agent> <shape> (+ rationale, cost diff)",
    )
    a.add_argument("agent")
    a.add_argument("shape")
    a.add_argument("--json", action="store_true",
                   help="emit the structured `reasons` audit object")
    a.set_defaults(func=cmd_advise)

    i = sub.add_parser(
        "inspect",
        help="per-tier posteriors + quality-drop CIs + widest gating cell",
    )
    i.add_argument("agent")
    i.add_argument("shape")
    i.add_argument("--json", action="store_true",
                   help="emit the per-tier table + widest-cell pointer as JSON")
    i.set_defaults(func=cmd_inspect)

    ap = sub.add_parser(
        "apply",
        help="set the agent's default model in gc config to the recommendation",
    )
    ap.add_argument("agent")
    ap.add_argument("--shape", help="shape to recommend on (else agent's "
                    "canonical default)")
    ap.add_argument("--city", help="city root (else $GC_CITY or cwd)")
    ap.add_argument("--rig", help="narrow the pack search to this rig")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="print the planned change without writing")
    ap.set_defaults(func=cmd_apply)

    return p


def main(argv: Optional[Sequence[str]] = None,
         out: Optional[io.TextIOBase] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    out = out or sys.stdout
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args, out)
    except BrokenPipeError:  # pragma: no cover - piped to head etc.
        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
