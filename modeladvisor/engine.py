"""The CC-TS decision rule — :func:`recommend` and :func:`inspect`.

A faithful, stdlib-only implementation of the four-layer Conservative Constrained
Thompson Sampling policy from ``docs/DESIGN.md`` §1. The two public entry points
are **pure functions of ``(config, store)``** — no I/O, no global mutable state,
deterministic, and seedable (the seed hook is reserved for a future genuine
Thompson-Sampling mode; v1's rule is LCB-deterministic and already needs no
randomness — DESIGN §1.4 properties 1–3).

The decision (DESIGN §1.2):

    cheapest tier whose posterior-credible quality-drop ≤ ``q_tol`` at
    ``1 − alpha = 0.95``; baseline ``tier*`` always feasible; upgrades
    unconstrained; ``Critical`` cells never downgrade.

Layers:

- **L1** Beta posteriors (read pooled, via :meth:`store.pooled`).
- **L2** conservative gate: admit a cheaper tier iff its one-sided **Wilson lower
  bound** ``q_lo = max(0, mu − z·sigma)`` clears ``mu* − q_tol`` (DESIGN §5.2,
  production default; no scipy needed).
- **L3** asymmetric-loss selection among admitted tiers:
  ``argmin E[L]`` with class multipliers ``{Critical:∞, Strict:20, Moderate:5,
  Lenient:1}`` and the cascade scaled by downstream-dependent count ``N_dep``.
- **L4** uncertainty-triggered eval — *recorded only* in v1 (the ``eval_flag`` and
  the widest-gating-cell pointer in :func:`inspect`); auto-dispatch is deferred.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Mapping

from modeladvisor.config import AdvisorConfig
from modeladvisor.store import Cell, CellStore, cell_key

# Sentinel for "no admitted downgrade beats the baseline" / cold start.
_BASELINE_RATIONALE = "no cheaper tier has cleared tolerance (insufficient evidence)"


# --------------------------------------------------------------------------- #
# Lower confidence bound (DESIGN §5.2 — Wilson-style normal LCB on the Beta)    #
# --------------------------------------------------------------------------- #


def wilson_lcb(cell: Cell, z: float) -> float:
    """One-sided lower confidence bound ``q_lo = max(0, mu − z·sigma)``.

    ``mu`` and ``sigma`` are the (pooled) Beta posterior moments. Capping at 0
    means sparse/wide cells gate-reject by default — exactly the conservative
    cold-start behaviour DESIGN §5.2 specifies. ``z = 1.645`` ≈ ``z_{0.95}`` for
    the default one-sided ``alpha = 0.05``.
    """
    return max(0.0, cell.mean - z * cell.stderr)


# --------------------------------------------------------------------------- #
# Cost / loss helpers (DESIGN §1.3 Layer 3, §2.3)                               #
# --------------------------------------------------------------------------- #


def _delta_cost(cfg: AdvisorConfig, hi_tier_id: str, lo_tier_id: str, tok_in: float, tok_out: float) -> float:
    """``Δcost(hi, lo) = cost(hi) − cost(lo)`` at the given token budget (DESIGN §2.3)."""
    hi = cfg.tier(hi_tier_id).cost(tok_in, tok_out)
    lo = cfg.tier(lo_tier_id).cost(tok_in, tok_out)
    return hi - lo


def _expected_loss(
    *,
    action: str,
    mu_tier: float,
    mu_star: float,
    delta_cost: float,
    multiplier: float,
    n_dep: int,
) -> float:
    """Expected asymmetric loss ``E[L]`` for one candidate action (DESIGN §1.3 L3).

    - ``stay`` (baseline): ``0``.
    - ``down``: ``-mu_tier·Δcost + max(0, mu* − mu_tier)·M·Δcost·N_dep``
      (savings when preserved; cascade penalty proportional to the *excess*
      failure probability vs baseline). ``Δcost = Δcost(tier*, tier_lower) > 0``.
    - ``up``: ``+Δcost(tier_higher, tier*)`` — a flat overpay (no modelled quality
      gain beyond the known-good baseline), so upgrades never beat ``stay``.
    """
    if action == "stay":
        return 0.0
    if action == "up":
        return abs(delta_cost)  # overpay vs baseline
    # action == "down"
    drop = max(0.0, mu_star - mu_tier)
    if math.isinf(multiplier):
        # Critical: any downgrade is infinitely penalised. (The gate already blocks
        # Critical downgrades; this is belt-and-braces so L3 can't select one.)
        return math.inf
    savings_term = -mu_tier * delta_cost
    cascade_term = drop * multiplier * delta_cost * float(max(1, n_dep))
    return savings_term + cascade_term


# --------------------------------------------------------------------------- #
# recommend (DESIGN §1.4, Alg. 1)                                              #
# --------------------------------------------------------------------------- #


def recommend(
    agent: str,
    shape: str,
    cfg: AdvisorConfig,
    store: CellStore,
    *,
    provider: str | None = None,
    tol_class: str | None = None,
    baseline_tier: str | None = None,
    n_dep: int = 1,
    tok_in: float | None = None,
    tok_out: float | None = None,
    seed: int | None = None,
) -> dict:
    """Recommend the cost-minimal tier for ``(agent, shape)`` under the quality gate.

    Pure / deterministic / seedable (DESIGN §1.4). ``state`` is passed in as
    ``(cfg, store)``; nothing is read from disk or the network here.

    Parameters
    ----------
    agent, shape:
        Base agent name and resolved canonical shape (the cell coordinates).
    cfg, store:
        The resolved config and the materialised cell store.
    provider:
        Provider for the cell key; defaults to ``cfg.default_provider``.
    tol_class:
        Tolerance-class name override; defaults to the cell's configured class.
    baseline_tier:
        ``tier*`` override; defaults to the cell's configured baseline.
    n_dep:
        Downstream-dependent count (blast radius) for the cascade term. Default 1.
    tok_in, tok_out:
        Realised token counts for the cost differential. If omitted, the shape's
        representative budget is used and flagged in the rationale (DESIGN §5.4).
    seed:
        Reserved for the future genuine-Thompson-Sampling mode; ignored by the
        deterministic v1 rule but echoed into ``reasons`` for reproducibility.

    Returns
    -------
    dict with keys ``tier_id``, ``model``, ``rationale`` (str), ``cost_delta``
    (vs baseline, negative = savings), ``reasons`` (the structured audit object),
    and ``posterior_summary`` (per-tier pooled posterior).
    """
    provider = provider or cfg.default_provider
    hp = cfg.hp

    # Resolve effective baseline + class (per-cell overrides win).
    base_id = baseline_tier or cfg.baseline_tier_id_for(agent, shape)
    if not cfg.has_tier(base_id):
        raise KeyError(f"baseline tier {base_id!r} not in roster")
    klass = cfg.tol_class(tol_class) if tol_class else cfg.tol_class_for(agent, shape)

    # Token budget (realised if supplied, else representative — §5.4).
    representative = tok_in is None or tok_out is None
    if representative:
        b_in, b_out = cfg.budget_for(shape)
        tok_in = b_in if tok_in is None else tok_in
        tok_out = b_out if tok_out is None else tok_out

    # Baseline posterior (pooled) -> mu*.
    base_cell = store.pooled(provider, agent, shape, base_id)
    mu_star = base_cell.mean
    gate_threshold = mu_star - klass.q_tol

    forced = cfg.is_forced_baseline(agent, shape)

    # ------------------------------------------------------------------ #
    # Build the candidate set + per-candidate audit rows.                #
    # ------------------------------------------------------------------ #
    order = list(cfg.tier_ids)  # cheapest -> most-capable
    base_idx = order.index(base_id)

    candidates: list[dict] = []
    posterior_summary: dict[str, dict] = {}

    for tid in order:
        cell = store.pooled(provider, agent, shape, tid)
        q_lo = wilson_lcb(cell, hp.z)
        idx = order.index(tid)

        posterior_summary[tid] = {
            "a": round(cell.a, 6),
            "b": round(cell.b, 6),
            "mean": round(cell.mean, 6),
            "q_lo": round(q_lo, 6),
            "n": cell.n,
        }

        if tid == base_id:
            action = "stay"
            admitted = True
            reason = "baseline (always feasible)"
            delta = 0.0
        elif idx > base_idx:
            action = "up"
            admitted = True  # upgrades unconstrained (but loss makes them lose)
            reason = "upgrade (unconstrained; only chosen if it minimises loss)"
            delta = _delta_cost(cfg, tid, base_id, tok_in, tok_out)  # overpay > 0
        else:
            action = "down"
            delta = _delta_cost(cfg, base_id, tid, tok_in, tok_out)  # savings > 0
            if forced:
                admitted = False
                reason = "force_baseline safety hatch active (downgrades disabled)"
            elif klass.is_critical:
                admitted = False
                reason = "Critical class: never downgrade (hard rule)"
            elif q_lo >= gate_threshold:
                admitted = True
                reason = f"admitted: q_lo={q_lo:.4f} >= {gate_threshold:.4f} (mu*-q_tol)"
            else:
                admitted = False
                reason = (
                    f"rejected: q_lo={q_lo:.4f} < {gate_threshold:.4f} "
                    f"(mu*-q_tol); evidence too thin (n={cell.n})"
                )

        # Expected loss for this action (only meaningful for admitted candidates,
        # but computed for all so the audit object is complete).
        exp_loss = _expected_loss(
            action=action,
            mu_tier=cell.mean,
            mu_star=mu_star,
            delta_cost=delta,
            multiplier=klass.multiplier,
            n_dep=n_dep,
        )

        # cost_diff is the signed cost vs baseline at the budget (negative = saves).
        if action == "down":
            cost_diff = -delta  # cheaper -> negative
        elif action == "up":
            cost_diff = +delta  # pricier -> positive
        else:
            cost_diff = 0.0

        candidates.append(
            {
                "tier_id": tid,
                "model": cfg.tier(tid).model,
                "action": action,
                "admitted": admitted,
                "q_lo": round(q_lo, 6),
                "mean": round(cell.mean, 6),
                "exp_loss": round(exp_loss, 8) if math.isfinite(exp_loss) else None,
                "exp_loss_inf": math.isinf(exp_loss),
                "cost_diff": round(cost_diff, 8),
                "n": cell.n,
                "reason": reason,
            }
        )

    # ------------------------------------------------------------------ #
    # Layer 3: argmin E[L] over the admitted set; ties -> cheaper tier.   #
    # ------------------------------------------------------------------ #
    admitted_rows = [c for c in candidates if c["admitted"]]
    # admitted always contains the baseline, so this is non-empty.
    def _loss_key(c: Mapping) -> tuple:
        el = math.inf if c["exp_loss"] is None else c["exp_loss"]
        return (el, order.index(c["tier_id"]))  # lower loss, then cheaper

    chosen = min(admitted_rows, key=_loss_key)
    chosen_id = chosen["tier_id"]
    chosen_cell = store.pooled(provider, agent, shape, chosen_id)

    # ------------------------------------------------------------------ #
    # Layer 4: eval flag (record only in v1, DESIGN §5.4 / §6.2).         #
    # ------------------------------------------------------------------ #
    ci_hw = _ci_halfwidth(chosen_cell, hp.alpha)
    eval_flag = ci_hw > hp.theta_eval

    cost_delta = chosen["cost_diff"]  # vs baseline (negative = savings)

    rationale = _build_rationale(
        cfg=cfg,
        provider=provider,
        agent=agent,
        shape=shape,
        chosen=chosen,
        base_id=base_id,
        mu_star=mu_star,
        klass=klass,
        gate_threshold=gate_threshold,
        candidates=candidates,
        representative=representative,
        forced=forced,
        cost_delta=cost_delta,
    )

    reasons = {
        "cell": {
            "provider": provider,
            "agent": agent,
            "shape": shape,
            "baseline_tier": base_id,
            "baseline_cell_key": cell_key(provider, agent, shape, base_id),
        },
        "class": klass.name,
        "q_tol": klass.q_tol,
        "multiplier": (None if math.isinf(klass.multiplier) else klass.multiplier),
        "critical": klass.is_critical,
        "baseline_mean": round(mu_star, 6),
        "gate_threshold": round(gate_threshold, 6),
        "alpha": hp.alpha,
        "z": hp.z,
        "n_dep": n_dep,
        "tok_in": tok_in,
        "tok_out": tok_out,
        "representative_budget": representative,
        "forced_baseline": forced,
        "seed": seed,
        "candidates": candidates,
        "advised_tier": chosen_id,
        "eval_flag": eval_flag,
        "eval_ci_halfwidth": round(ci_hw, 6),
        "theta_eval": hp.theta_eval,
    }

    return {
        "tier_id": chosen_id,
        "model": cfg.tier(chosen_id).model,
        "rationale": rationale,
        "cost_delta": round(cost_delta, 8),
        "reasons": reasons,
        "posterior_summary": posterior_summary,
    }


# --------------------------------------------------------------------------- #
# Rationale string builder                                                      #
# --------------------------------------------------------------------------- #


def _fmt_cost(x: float) -> str:
    return f"${x:+.4f}"


def _build_rationale(
    *,
    cfg: AdvisorConfig,
    provider: str,
    agent: str,
    shape: str,
    chosen: Mapping,
    base_id: str,
    mu_star: float,
    klass,
    gate_threshold: float,
    candidates: list[dict],
    representative: bool,
    forced: bool,
    cost_delta: float,
) -> str:
    """Human-readable rationale string (DESIGN §6.1 examples)."""
    chosen_id = chosen["tier_id"]
    budget_note = " (representative token budget)" if representative else ""

    if forced and chosen_id == base_id:
        return (
            f"{chosen_id} (baseline): force_baseline safety hatch active — "
            f"playing the configured tier, learning disabled."
        )

    if klass.is_critical and chosen_id == base_id:
        return (
            f"{chosen_id} (baseline): Critical class (q_tol=0) — downgrades are a "
            f"hard never-downgrade rule; baseline is the only feasible tier."
        )

    if chosen["action"] == "stay":
        # Cold start / nothing admitted: name the thin cheaper tiers.
        cheaper = [c for c in candidates if c["action"] == "down"]
        if cheaper:
            bits = ", ".join(
                f"{c['tier_id']} q_lo={c['q_lo']:.2f}" for c in cheaper
            )
            return (
                f"{chosen_id} (baseline): no cheaper tier clears tolerance — "
                f"{bits} all < {gate_threshold:.2f} threshold "
                f"(baseline mean {mu_star:.2f} − {klass.q_tol:.2f} tol)."
            )
        return f"{chosen_id} (baseline): {_BASELINE_RATIONALE}."

    if chosen["action"] == "down":
        return (
            f"{chosen_id}: LCB q_lo={chosen['q_lo']:.2f} >= baseline mean "
            f"{mu_star:.2f} − {klass.q_tol:.2f} tol = {gate_threshold:.2f}; "
            f"cheapest admitted tier; expected loss {_fmt_cost(chosen['exp_loss'] or 0.0)}"
            f" vs {base_id}; cost {_fmt_cost(cost_delta)}/dispatch{budget_note}."
        )

    # upgrade chosen (rare; only if baseline somehow had positive loss — shouldn't
    # happen since stay==0, but handled for completeness).
    return (
        f"{chosen_id}: upgrade selected; cost {_fmt_cost(cost_delta)}/dispatch"
        f"{budget_note}."
    )


# --------------------------------------------------------------------------- #
# Credible-interval helpers                                                     #
# --------------------------------------------------------------------------- #


def _ci_halfwidth(cell: Cell, alpha: float) -> float:
    """Two-sided ``1 − alpha`` normal CI half-width on the cell's posterior mean.

    Used for the Layer-4 eval trigger (DESIGN §5.4): a half-width above
    ``theta_eval`` flags the cell as wanting an eval probe.
    """
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    return z * cell.stderr


def _drop_ci(base_cell: Cell, cand_cell: Cell, alpha: float) -> tuple[float, float, float]:
    """Normal-approx ``1 − alpha`` CI on the quality DROP ``theta[tier*] − theta[tier]``.

    A difference of two Betas has no closed form; the engine is deterministic and
    stdlib-only, so we use the moment-matched normal approximation (mean = ``mu* −
    mu_tier``, variance = ``var* + var_tier``). Returns ``(lo, mean, hi)``. This is
    the quantity the constraint bounds, surfaced by :func:`inspect` (DESIGN §6.2).
    """
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    mean = base_cell.mean - cand_cell.mean
    sd = math.sqrt(base_cell.variance + cand_cell.variance)
    return (mean - z * sd, mean, mean + z * sd)


# --------------------------------------------------------------------------- #
# inspect (DESIGN §6.2)                                                         #
# --------------------------------------------------------------------------- #


def inspect(
    agent: str,
    shape: str,
    cfg: AdvisorConfig,
    store: CellStore,
    *,
    provider: str | None = None,
    tol_class: str | None = None,
    baseline_tier: str | None = None,
) -> dict:
    """Read-only deep view of the ``(agent, shape)`` cells across the whole roster.

    Returns, per tier: the pooled ``Beta(a~, b~)``, mean, the Wilson LCB ``q_lo``,
    observation count ``n``, last-update time, and admit/reject under the current
    tolerance; plus, per *candidate* (cheaper) tier, the ``1 − alpha`` credible
    interval on the quality DROP vs baseline; plus a pointer to the **widest
    gating cell** — the tier whose posterior CI half-width is widest *among the
    cells that are gating a downgrade* (i.e. rejected because the bound is too
    wide), named as the highest-value eval probe (DESIGN §6.2 / Layer 4).
    """
    provider = provider or cfg.default_provider
    hp = cfg.hp
    base_id = baseline_tier or cfg.baseline_tier_id_for(agent, shape)
    klass = cfg.tol_class(tol_class) if tol_class else cfg.tol_class_for(agent, shape)
    forced = cfg.is_forced_baseline(agent, shape)

    base_cell = store.pooled(provider, agent, shape, base_id)
    mu_star = base_cell.mean
    gate_threshold = mu_star - klass.q_tol

    order = list(cfg.tier_ids)
    base_idx = order.index(base_id)

    tiers_out: list[dict] = []
    widest_gating = None  # (halfwidth, tier_id) among gating-and-rejected cells

    for tid in order:
        cell = store.pooled(provider, agent, shape, tid)
        q_lo = wilson_lcb(cell, hp.z)
        ci_hw = _ci_halfwidth(cell, hp.alpha)
        idx = order.index(tid)

        if tid == base_id:
            role = "baseline"
            admitted = True
            drop_ci = None
        elif idx > base_idx:
            role = "upgrade"
            admitted = True
            drop_ci = None
        else:
            role = "candidate"
            if forced:
                admitted = False
            elif klass.is_critical:
                admitted = False
            else:
                admitted = q_lo >= gate_threshold
            lo, mean, hi = _drop_ci(base_cell, cell, hp.alpha)
            # The drop CI exceeding q_tol is *why* a cheaper tier can't be admitted.
            exceeds_tol = lo > klass.q_tol or hi > klass.q_tol
            drop_ci = {
                "lo": round(lo, 6),
                "mean": round(mean, 6),
                "hi": round(hi, 6),
                "q_tol": klass.q_tol,
                "exceeds_tol": exceeds_tol,
            }
            # A cell is "gating" if it's a real candidate the gate rejected on
            # uncertainty (not Critical / forced, which reject categorically).
            if (
                not admitted
                and not klass.is_critical
                and not forced
                and ci_hw > hp.theta_eval
            ):
                if widest_gating is None or ci_hw > widest_gating[0]:
                    widest_gating = (ci_hw, tid)

        tiers_out.append(
            {
                "tier_id": tid,
                "model": cfg.tier(tid).model,
                "role": role,
                "a": round(cell.a, 6),
                "b": round(cell.b, 6),
                "mean": round(cell.mean, 6),
                "q_lo": round(q_lo, 6),
                "n": cell.n,
                "last_update": cell.last_update,
                "ci_halfwidth": round(ci_hw, 6),
                "admitted": admitted,
                "quality_drop_ci": drop_ci,
            }
        )

    widest = None
    if widest_gating is not None:
        hw, tid = widest_gating
        widest = {
            "tier_id": tid,
            "cell_key": cell_key(provider, agent, shape, tid),
            "ci_halfwidth": round(hw, 6),
            "theta_eval": hp.theta_eval,
            "rationale": (
                f"widest gating cell: {tid} (CI half-width {hw:.2f} > "
                f"{hp.theta_eval:.2f}); run an eval on "
                f"{cell_key(provider, agent, shape, tid)} to resolve."
            ),
        }

    return {
        "cell": {"provider": provider, "agent": agent, "shape": shape},
        "baseline_tier": base_id,
        "baseline_mean": round(mu_star, 6),
        "class": klass.name,
        "q_tol": klass.q_tol,
        "gate_threshold": round(gate_threshold, 6),
        "alpha": hp.alpha,
        "forced_baseline": forced,
        "tiers": tiers_out,
        "widest_gating_cell": widest,
    }
