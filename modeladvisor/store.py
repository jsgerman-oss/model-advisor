"""The cell store: per-cell ``Beta(a, b)`` posteriors over dispatch quality.

Pure I/O + accumulation; **no network**. The store reads the dispatch + quality
records from ``.beads/telemetry/invocations.jsonl`` (the source of truth, DESIGN
§5) and materialises per-cell ``Beta(a, b)`` posteriors via the closed-form
Bernoulli update ``a += w·q``, ``b += w·(1 − q)`` (DESIGN §1.3 Layer 1). The
materialisation is persisted to ``advisor-cells.json`` — a **rebuildable cache**:
deleting it and replaying the JSONL (priors + every ``quality`` record in order)
reproduces it exactly (DESIGN §5.5).

Cold-start priors follow DESIGN §3.2: the baseline tier ``tier*`` is seeded
optimistically (``Beta(8, 2)``, mean 0.8); cheaper tiers get a *depressed* mean
interpolated along the cost order with a small ``s_prior`` pseudocount, so their
one-sided lower bound starts below any plausible ``mu* − q_tol`` and the gate
rejects until evidence accrues. A consumer may instead supply a ``priors.json``
(confidence-seeded, DESIGN §3.1).

Hierarchical capped pseudocount pooling (DESIGN §1.3) is applied at *read* time
via :meth:`CellStore.pooled`: a cell borrows strength from its siblings (same
agent, other shapes; and same tolerance class across agents), weighted by inverse
sibling standard-error and capped by ``pool_lambda ≤ 0.5`` so pooling never
dominates own-cell evidence.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Mapping

from modeladvisor.config import AdvisorConfig

CELL_KEY_SEP = "::"


def cell_key(provider: str, agent: str, shape: str, tier_id: str) -> str:
    """Build the canonical cell key ``provider::agent::shape::tier_id`` (DESIGN §2.4)."""
    return CELL_KEY_SEP.join((provider, agent, shape, tier_id))


def parse_cell_key(key: str) -> tuple[str, str, str, str]:
    """Inverse of :func:`cell_key`. Raises ``ValueError`` on a malformed key."""
    parts = key.split(CELL_KEY_SEP)
    if len(parts) != 4:
        raise ValueError(f"malformed cell_key {key!r} (want provider::agent::shape::tier_id)")
    return parts[0], parts[1], parts[2], parts[3]


@dataclass
class Cell:
    """A single cell's Beta posterior plus bookkeeping.

    ``a``/``b`` are the *posterior* Beta parameters (prior already folded in).
    ``n`` is the raw observation count (un-weighted). ``last_update`` is the RFC3339
    timestamp of the most recent quality record applied (``None`` if prior-only).
    """

    a: float
    b: float
    n: int = 0
    last_update: str | None = None

    # ---- moments (DESIGN §1.3) ------------------------------------------- #

    @property
    def mean(self) -> float:
        return self.a / (self.a + self.b)

    @property
    def variance(self) -> float:
        s = self.a + self.b
        return (self.a * self.b) / (s * s * (s + 1.0))

    @property
    def stderr(self) -> float:
        return math.sqrt(self.variance)

    def update(self, q: float, w: float, ts: str | None) -> None:
        """Apply one weighted Bernoulli observation (DESIGN §1.3 Layer 1)."""
        self.a += w * q
        self.b += w * (1.0 - q)
        self.n += 1
        if ts is not None:
            self.last_update = ts

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "n": self.n, "last_update": self.last_update}

    @classmethod
    def from_dict(cls, d: Mapping) -> "Cell":
        return cls(
            a=float(d["a"]),
            b=float(d["b"]),
            n=int(d.get("n", 0)),
            last_update=d.get("last_update"),
        )


# --------------------------------------------------------------------------- #
# Cold-start priors (DESIGN §3)                                                 #
# --------------------------------------------------------------------------- #


def _confidence_prior(c: float) -> tuple[float, float]:
    """Confidence ``c ∈ [0, 100]`` -> ``(a_prior, b_prior)`` (DESIGN §3.1)."""
    a = max(1.0, round(c / 5.0))
    b = max(1.0, round((100.0 - c) / 5.0))
    return float(a), float(b)


def cold_start_prior(cfg: AdvisorConfig, tier_id: str, baseline_tier_id: str) -> tuple[float, float]:
    """Cost-ordered conservative cold-start prior for one tier (DESIGN §3.2).

    - Baseline tier: optimistic ``Beta(baseline_a, baseline_b)`` (default mean 0.8).
    - Cheaper tiers: depressed mean interpolated along the cost order between
      ``cold_m_lo`` (cheapest) and the baseline mean, with weak ``s_prior``.
    - More-capable-than-baseline tiers (unconstrained upgrades): seeded at the
      baseline mean (monotone assumption: at least as good).
    """
    hp = cfg.hp
    if tier_id == baseline_tier_id:
        return hp.baseline_a, hp.baseline_b

    order = list(cfg.tier_ids)  # cheapest -> most-capable
    idx = order.index(tier_id)
    base_idx = order.index(baseline_tier_id)
    m_hi = hp.baseline_mean

    if idx >= base_idx:
        # Upgrade tier (more capable than baseline): optimistic, same as baseline mean.
        mean = m_hi
    else:
        # Cheaper than baseline: interpolate the prior mean along the cost order.
        # Cheapest tier (order index 0) -> cold_m_lo; baseline -> baseline mean.
        # Robust to non-contiguous ranks (uses position in the cost order).
        span = base_idx  # number of steps from cheapest to baseline (>= 1 here)
        frac = idx / span if span > 0 else 1.0
        mean = hp.cold_m_lo + (m_hi - hp.cold_m_lo) * frac

    mean = min(max(mean, 1e-6), 1.0 - 1e-6)
    a = mean * hp.s_prior
    b = (1.0 - mean) * hp.s_prior
    return a, b


# --------------------------------------------------------------------------- #
# Cell store                                                                    #
# --------------------------------------------------------------------------- #


@dataclass
class CellStore:
    """Materialised per-cell posteriors keyed by ``provider::agent::shape::tier_id``.

    Construct one of three ways:

    - :meth:`cold_start` — priors only (no telemetry yet).
    - :meth:`rebuild`    — replay priors + ``invocations.jsonl`` (the authoritative
      build; the cache is derived from this).
    - :meth:`load`       — read a previously-persisted ``advisor-cells.json`` cache
      (falling back to cold-start priors for any cell not in the cache).

    The store is *config-aware* for prior synthesis and pooling, but otherwise a
    plain ``cell_key -> Cell`` map.
    """

    cfg: AdvisorConfig
    cells: dict[str, Cell] = field(default_factory=dict)
    #: Optional confidence-seeded priors (DESIGN §3.1): {cell_key: confidence}.
    priors: Mapping[str, float] = field(default_factory=dict)

    # ---- prior synthesis -------------------------------------------------- #

    def _prior(self, key: str) -> Cell:
        """Synthesize the prior cell for ``key`` (confidence prior wins if present)."""
        provider, agent, shape, tier_id = parse_cell_key(key)
        if key in self.priors:
            a, b = _confidence_prior(float(self.priors[key]))
        else:
            base = self.cfg.baseline_tier_id_for(agent, shape)
            a, b = cold_start_prior(self.cfg, tier_id, base)
        return Cell(a=a, b=b, n=0, last_update=None)

    def get(self, key: str) -> Cell:
        """Return the cell for ``key``, synthesising (and caching) its prior if unseen."""
        cell = self.cells.get(key)
        if cell is None:
            cell = self._prior(key)
            self.cells[key] = cell
        return cell

    def get_tier(self, provider: str, agent: str, shape: str, tier_id: str) -> Cell:
        return self.get(cell_key(provider, agent, shape, tier_id))

    # ---- construction ----------------------------------------------------- #

    @classmethod
    def cold_start(
        cls, cfg: AdvisorConfig, priors: Mapping[str, float] | None = None
    ) -> "CellStore":
        """A store seeded with priors only (no observations)."""
        return cls(cfg=cfg, cells={}, priors=dict(priors or {}))

    @classmethod
    def rebuild(
        cls,
        cfg: AdvisorConfig,
        jsonl_path: str | os.PathLike[str] | None,
        priors: Mapping[str, float] | None = None,
    ) -> "CellStore":
        """Build the store by replaying ``invocations.jsonl`` over the priors.

        This is the **authoritative** construction (DESIGN §5.5): the JSONL is the
        source of truth; ``advisor-cells.json`` is merely a cache of the result.
        Missing / empty file ⇒ a pure cold-start store. Malformed lines are
        skipped (the log is append-only and may be partially written).
        """
        store = cls.cold_start(cfg, priors)
        if jsonl_path is None:
            return store
        p = os.fspath(jsonl_path)
        if not os.path.exists(p):
            return store
        with open(p, "r", encoding="utf-8") as fh:
            store.apply_records(_iter_jsonl(fh))
        return store

    @classmethod
    def load(
        cls,
        cfg: AdvisorConfig,
        cache_path: str | os.PathLike[str],
        priors: Mapping[str, float] | None = None,
    ) -> "CellStore":
        """Load a persisted ``advisor-cells.json`` cache.

        Cells absent from the cache fall back to cold-start priors on demand. If
        the cache file is missing, this is equivalent to :meth:`cold_start`.
        """
        store = cls.cold_start(cfg, priors)
        p = os.fspath(cache_path)
        if not os.path.exists(p):
            return store
        with open(p, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        cells = doc.get("cells", doc) if isinstance(doc, Mapping) else {}
        for key, d in cells.items():
            store.cells[key] = Cell.from_dict(d)
        return store

    # ---- accumulation ----------------------------------------------------- #

    def apply_records(self, records: Iterable[Mapping]) -> int:
        """Apply an iterable of telemetry records. Returns #quality records applied.

        Only ``kind == "quality"`` records mutate a posterior. ``dispatch`` and
        ``recommendation`` records are ignored here (they carry no outcome; the
        cell credited by a quality record already carries its ``cell_key``).
        """
        applied = 0
        for rec in records:
            if not isinstance(rec, Mapping):
                continue
            if rec.get("kind") != "quality":
                continue
            if self.apply_quality(rec):
                applied += 1
        return applied

    def apply_quality(self, rec: Mapping) -> bool:
        """Apply one ``kind == "quality"`` record. Returns True if it updated a cell.

        Honours the channel-weight defaults (DESIGN §4.4): if the record carries an
        explicit ``weight`` it is used; otherwise the weight is derived from the
        ``channel`` via the configured ``w_close``/``w_review``/``w_eval``.
        Records with a non-binary / missing ``q`` are dropped (e.g. ``transient``
        failures and ``blocked`` verdicts are never emitted as quality rows, but we
        defend against a malformed one).
        """
        key = rec.get("cell_key")
        if not key:
            return False
        q = rec.get("q")
        if q is None:
            return False
        try:
            qf = float(q)
        except (TypeError, ValueError):
            return False
        if qf not in (0.0, 1.0):
            return False
        w = rec.get("weight")
        if w is None:
            w = self._channel_weight(str(rec.get("channel", "close")))
        else:
            w = float(w)
        ts = rec.get("ts")
        self.get(str(key)).update(qf, w, ts if ts is None else str(ts))
        return True

    def _channel_weight(self, channel: str) -> float:
        hp = self.cfg.hp
        return {"close": hp.w_close, "review": hp.w_review, "eval": hp.w_eval}.get(
            channel, hp.w_close
        )

    # ---- hierarchical capped pooling (DESIGN §1.3) ------------------------ #

    def pooled(self, provider: str, agent: str, shape: str, tier_id: str) -> Cell:
        """Return the *pooled* ``(a~, b~)`` cell read by the gate / decision / CI.

        Borrow strength from sibling cells of the **same tier**:

        - same agent, other shapes (cross-shape siblings), and
        - same tolerance class, other agents (cross-agent siblings).

        Weight each sibling by ``1 / sibling-stderr`` (more-certain siblings count
        for more), scale the whole pooled contribution by ``pool_lambda ≤ 0.5``,
        and normalise so the pooled pseudocount mass never exceeds ``pool_lambda ×
        own (a + b)``. This keeps own-cell evidence dominant (DESIGN §1.3 cap).
        """
        own = self.get(cell_key(provider, agent, shape, tier_id))
        lam = self.cfg.hp.pool_lambda
        if lam <= 0.0:
            return Cell(a=own.a, b=own.b, n=own.n, last_update=own.last_update)

        sib_keys = self._sibling_keys(provider, agent, shape, tier_id)
        if not sib_keys:
            return Cell(a=own.a, b=own.b, n=own.n, last_update=own.last_update)

        num_a = 0.0
        num_b = 0.0
        wsum = 0.0
        for sk in sib_keys:
            sib = self.get(sk)
            se = sib.stderr
            wsib = 1.0 / se if se > 0 else 0.0
            if wsib == 0.0:
                continue
            num_a += wsib * sib.a
            num_b += wsib * sib.b
            wsum += wsib
        if wsum == 0.0:
            return Cell(a=own.a, b=own.b, n=own.n, last_update=own.last_update)

        # Weighted-average sibling pseudocounts, then cap the injected mass at
        # lambda * own_mass so pooling never dominates own-cell evidence.
        avg_a = num_a / wsum
        avg_b = num_b / wsum
        sib_mass = avg_a + avg_b
        own_mass = own.a + own.b
        cap = lam * own_mass
        scale = (cap / sib_mass) if sib_mass > 0 else 0.0
        scale = min(scale, lam)  # also never inject more than lambda of a sibling unit

        a_pooled = own.a + scale * avg_a
        b_pooled = own.b + scale * avg_b
        return Cell(a=a_pooled, b=b_pooled, n=own.n, last_update=own.last_update)

    def _sibling_keys(self, provider: str, agent: str, shape: str, tier_id: str) -> list[str]:
        """Cell keys of pooling siblings for the given cell (same tier, sibling cells)."""
        self_key = cell_key(provider, agent, shape, tier_id)
        out: list[str] = []
        # Cross-shape siblings: same agent, this agent's other canonical shapes.
        for s in self.cfg.canonical_shapes_for(agent):
            if s == shape:
                continue
            k = cell_key(provider, agent, s, tier_id)
            if k != self_key:
                out.append(k)
        # Cross-agent siblings: other agents whose (agent, shape) shares this cell's
        # tolerance class, for the same shape + tier.
        my_class = self.cfg.tol_class_for(agent, shape).name
        for other in self.cfg.agent_shapes:
            if other == agent:
                continue
            if shape not in self.cfg.canonical_shapes_for(other):
                continue
            if self.cfg.tol_class_for(other, shape).name != my_class:
                continue
            k = cell_key(provider, other, shape, tier_id)
            if k != self_key:
                out.append(k)
        # De-dup while preserving order.
        seen: set[str] = set()
        uniq: list[str] = []
        for k in out:
            if k not in seen:
                seen.add(k)
                uniq.append(k)
        return uniq

    # ---- persistence ------------------------------------------------------ #

    def save(self, cache_path: str | os.PathLike[str]) -> str:
        """Persist the materialised posteriors to ``advisor-cells.json``.

        Only *observed* cells (those with ``n > 0`` or otherwise materialised) are
        written; priors are reproducible from config so we don't bloat the cache
        with untouched prior cells. The written document is a stable, sorted JSON
        so reruns produce byte-identical caches (eases diffing / testing).
        """
        p = os.fspath(cache_path)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        doc = {
            "schema_version": "advisor.v1",
            "baseline_tier": self.cfg.baseline_tier_id,
            "cells": {
                k: self.cells[k].to_dict()
                for k in sorted(self.cells)
                if self.cells[k].n > 0
            },
        }
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, sort_keys=True)
            fh.write("\n")
        return p

    # ---- introspection ---------------------------------------------------- #

    def observed_cells(self) -> dict[str, Cell]:
        """Cells that have at least one observation (``n > 0``)."""
        return {k: v for k, v in self.cells.items() if v.n > 0}


# --------------------------------------------------------------------------- #
# JSONL helpers                                                                 #
# --------------------------------------------------------------------------- #


def _iter_jsonl(fh: Iterable[str]) -> Iterator[dict]:
    """Yield parsed JSON objects from a JSONL stream, skipping blank/bad lines."""
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


def read_jsonl(path: str | os.PathLike[str]) -> list[dict]:
    """Read all JSON objects from a ``.jsonl`` file (skips malformed lines)."""
    p = os.fspath(path)
    if not os.path.exists(p):
        return []
    with open(p, "r", encoding="utf-8") as fh:
        return list(_iter_jsonl(fh))
