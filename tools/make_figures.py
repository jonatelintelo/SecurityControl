#!/usr/bin/env python
"""RQ1 paper figures.

Design constraints, and why each one:

* **At most three categorical series per panel.** The palette's first three slots
  are the documented all-pairs-safe subset (CVD dE 9.2 light / 9.4 dark). Past
  three, the method says fold or facet rather than cycle hues — so the extra
  direction variants go to small multiples and an appendix table, not to a fourth
  colour.
* **Identity is never colour alone.** Every series also carries a linestyle and a
  marker. An ICLR figure gets printed, photocopied and read on a projector; colour
  is the first thing to go.
* **Diverging encoding for signed data** (the delta matrix): blue<->red with a
  neutral GREY midpoint, never a hue at zero, so "no effect" reads as nothing.
* **Sequential = one hue** for magnitude-only heatmaps.
* Recessive grid and axes, thin marks, direct labels rather than a number on every
  point, and a legend whenever a panel carries more than one series.

Colour is not eyeballed: the palette is the skill's reference instance and the
three-slot subset is the configuration it documents as passing all six checks in
both modes.

    python tools/make_figures.py            # -> results/figures/*.pdf and *.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import MATCHED_CONTROL_VARIANT, RQ1_MODELS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402


def _post(m):
    """Keep only the `t_post_inst` read position.

    `causal_matrix*.csv` carries a second read position (`t_inst`), added for the
    token-resolved readout. Every gate-level quantity is defined at
    `t_post_inst`; consuming both would double-count each intervention and mix
    two incomparable residual bases.

    It deliberately does NOT restrict read LAYERS. The matrix carries every layer
    downstream of the steer layer, and which of those form the test family is an
    adjudication option (`gate_layers`) applied inside `_adjudicate_at`, so that
    it can be swept. Filtering here would silently pin that sweep to one arm.
    Older matrices lack the column.
    """
    if "read_position" in m.columns:
        m = m[m.read_position == "t_post_inst"]
    return m



R = Path("results")
OUT = R / "figures"
MODELS = list(RQ1_MODELS)
NICE = {"qwen2.5-7b": "Qwen2.5-7B", "qwen3.5-9b": "Qwen3.5-9B",
        "llama3.1-8b": "Llama-3.1-8B", "qwen3.5-35b-a3b": "Qwen3.5-35B-A3B",
        "nemotron-3-nano-30b-a3b": "Nemotron-3-Nano-30B-A3B"}

# Palette — reference instance, slots 1-3 (documented all-pairs safe).
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8880"
GRID, SURFACE = "#e6e5e1", "#fcfcfb"
# Diverging pair with a neutral grey midpoint.
DIVERGING = LinearSegmentedColormap.from_list("bl_gy_rd", ["#184f95", "#86b6ef", "#f0efec", "#f0a6a5", "#b02c2b"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blues", ["#eef4fd", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 160, "savefig.bbox": "tight",
})


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------
def fig_depth_curves() -> None:
    """Separation vs relative depth — the three variables, one panel per model."""
    # The control variable is the OVER variant in both panels. The under variant
    # exists on one model only, and the two are different directions (cos 0.72 vs a
    # 0.94 floor) — putting one in each panel under a shared legend would show two
    # variables as if they were one. The under variant gets its own panel below.
    series = [("R_harm_at_post", "harm", S1, "-", "o"),
              ("R_control_harmless", "control (over-refusal)", S2, "--", "s"),
              ("R_role_probe", "role", S3, ":", "^")]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    for ax, m in zip(axes, MODELS):
        cur = pd.read_csv(R / "rq1" / m / "emergence_curves.csv")
        for concept, label, col, ls, mk in series:
            d = cur[cur.concept == concept].sort_values("relative_depth")
            if not len(d):
                continue
            ax.fill_between(d.relative_depth, d.auc_ci_low, d.auc_ci_high,
                            color=col, alpha=0.13, linewidth=0)
            ax.plot(d.relative_depth, d.auc, color=col, ls=ls, lw=1.6,
                    marker=mk, ms=3.2, mfc=SURFACE, mew=0.9, label=label)
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (2, 3)))
        ax.set_title(NICE[m], color=INK)
        ax.set_xlabel("relative depth")
        ax.set_ylim(0.4, 1.02)
        _despine(ax)
    axes[0].set_ylabel("held-out separation (AUC)")
    axes[0].text(0.02, 0.53, "chance", color=MUTED, fontsize=7.5)
    # The under-refusal control, where it exists — same panel, distinguished by a
    # thin dotted line so it cannot be mistaken for the over variant.
    cur = pd.read_csv(R / "rq1" / MODELS[0] / "emergence_curves.csv")
    u = cur[cur.concept == "R_control"].sort_values("relative_depth")
    if len(u):
        axes[0].plot(u.relative_depth, u.auc, color=S2, ls=(0, (1, 2)), lw=1.3,
                     label="control (under-refusal)")
        axes[0].legend(loc="lower right", fontsize=7.5)
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Harm and role resolve early; control needs roughly half the network",
                 y=1.04, fontsize=10.5, color=INK)
    _save(fig, "fig1_depth_curves")


def fig_onset() -> None:
    """Emergence onset with bootstrap CIs — one row per concept, every roster model."""
    rows = []
    for m in MODELS:
        e = pd.read_csv(R / "rq1" / m / "emergence_summary.csv")
        for _, r in e.iterrows():
            if r.concept in {"R_harm_at_post", "R_control", "R_control_harmless", "R_role_probe"}:
                rows.append({"model": m, **r})
    d = pd.DataFrame(rows)
    order = ["R_role_probe", "R_harm_at_post", "R_control_harmless", "R_control"]
    lab = {"R_role_probe": "role", "R_harm_at_post": "harm",
           "R_control_harmless": "control (over)", "R_control": "control (under)"}
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    y, ticks, tlab = 0, [], []
    for c in order:
        for m in MODELS:
            s = d[(d.concept == c) & (d.model == m)]
            if not len(s):
                continue
            r = s.iloc[0]
            col = S1 if m == MODELS[0] else S2
            wide = (r.onset_90_ci_high - r.onset_90_ci_low) > 0.4
            ax.plot([r.onset_90_ci_low, r.onset_90_ci_high], [y, y],
                    color=col, lw=2.2 if not wide else 1.2,
                    ls="-" if not wide else (0, (1.5, 1.5)), solid_capstyle="round")
            ax.plot(r.onset_90, y, "o" if m == MODELS[0] else "s",
                    color=col, ms=5, mfc=SURFACE, mew=1.4)
            if wide:
                ax.text(r.onset_90_ci_high + 0.015, y, "unreliable", fontsize=7,
                        color=MUTED, va="center")
            ticks.append(y); tlab.append(f"{lab[c]}  ·  {NICE[m]}")
            y -= 1
    ax.set_yticks(ticks); ax.set_yticklabels(tlab, fontsize=8)
    ax.set_xlabel("relative depth at which separation first reaches 90% of its peak")
    ax.set_xlim(-0.02, 1.0)
    _despine(ax); ax.grid(axis="y", visible=False)
    ax.set_title("Onset depth, with bootstrap CIs", loc="left", color=INK)
    _save(fig, "fig2_onset")


def fig_geometry() -> None:
    """Cross-concept similarity against the estimator's own noise floor."""
    pairs = [("R_control", "R_control_unbalanced", "same variable\n(positive control)"),
             ("R_harm_at_post", "R_control", "harm vs control"),
             ("R_control", "R_control_harmless", "control under vs over"),
             ("R_harm_at_post", "R_control_harmless", "harm vs control (over)")]
    fig, ax = plt.subplots(figsize=(6.6, 2.4))
    ys, labs = [], []
    for i, (a, b, lab) in enumerate(pairs):
        g = pd.read_csv(R / "rq1" / MODELS[0] / "geometry_cosines.csv")
        g = g[g.same_position]
        m = g[((g.concept_a == a) & (g.concept_b == b)) | ((g.concept_a == b) & (g.concept_b == a))]
        if not len(m):
            continue
        r = m.loc[m.abs_cosine.idxmax()]
        y = -i
        ax.plot([0, r.abs_cosine], [y, y], color=GRID, lw=6, solid_capstyle="butt", zorder=1)
        ax.plot(r.abs_cosine, y, "o", color=S1, ms=8, mfc=SURFACE, mew=2, zorder=3)
        ax.plot([r.split_half_floor], [y], "|", color=S2, ms=16, mew=2.2, zorder=4)
        ax.text(r.abs_cosine - 0.02, y + 0.32, f"{r.abs_cosine:.2f}", fontsize=8,
                color=INK, ha="right")
        ys.append(y); labs.append(lab)
    ax.set_yticks(ys); ax.set_yticklabels(labs, fontsize=8)
    ax.set_xlim(0, 1.05); ax.set_xlabel("|cosine| at a matched read position")
    _despine(ax); ax.grid(axis="y", visible=False)
    ax.plot([], [], "o", color=S1, mfc=SURFACE, mew=2, label="observed")
    ax.plot([], [], "|", color=S2, ms=12, mew=2.2, label="split-half floor (same variable, disjoint halves)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, -0.42), fontsize=7.5, ncol=2)
    ax.set_title("Distinct, against an estimator that can say “same”", loc="left", color=INK)
    _save(fig, "fig3_geometry")


def fig_delta_matrix() -> None:
    """Directed steering effects: signed, so a diverging scale with a grey zero."""
    names = {"R_harm": "harm", "R_control": "control", "R_role": "role"}
    order = ["R_harm", "R_control", "R_role"]
    panels = []
    # The matched arm — `over` on every model — so the panels compare like with
    # like rather than under-refusal on one against over-refusal on another.
    for m, variant in [(m, MATCHED_CONTROL_VARIANT) for m in MODELS]:
        p = R / "rq1" / m / f"causal_matrix__{variant}.csv"
        if not p.exists():
            continue
        d = _post(pd.read_csv(p))
        d = d[(d.source != "random") & (d.kl_harmless <= 0.5) & (d.alpha == 1.0)]
        piv = d.pivot_table(index="source", columns="target", values="delta",
                            aggfunc="mean").reindex(index=order, columns=order)
        panels.append((m, piv))
    if not panels:
        return

    # ONE scale across both panels. Two panels under a single colourbar must share
    # a normalisation or the bar is a lie for one of them. Scaled on the
    # OFF-diagonal: the diagonal is partly tautological (steering a direction
    # raises its own projection by construction) and would otherwise set the range
    # and wash out the structure the figure is about.
    offs = []
    for _, piv in panels:
        o = piv.values.copy()
        np.fill_diagonal(o, np.nan)
        offs.append(o)
    v = float(np.nanmax([np.nanmax(np.abs(o)) for o in offs])) or 1.0
    norm = TwoSlopeNorm(0, -v, v)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    for ax, (m, piv), off in zip(axes, panels, offs):
        im = ax.imshow(off, cmap=DIVERGING, norm=norm)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                val = piv.values[i, j]
                if not np.isfinite(val):
                    continue
                if i == j:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor=GRID,
                                               hatch="///", edgecolor=SURFACE, lw=0))
                    ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8,
                            color=MUTED, style="italic")
                else:
                    ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8,
                            color=INK if abs(val) < 0.6 * v else SURFACE)
        ax.set_xticks(range(3), [names[c] for c in piv.columns], fontsize=8)
        ax.set_yticks(range(3), [names[c] for c in piv.index], fontsize=8)
        ax.set_xlabel("read")
        if ax is axes[0]:
            ax.set_ylabel("steered")
        ax.set_title(NICE[m], color=INK)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.03)
    cb.set_label("Δ projection, off-diagonal\n(units of the target's own class gap)", fontsize=8)
    cb.outline.set_visible(False)
    fig.suptitle("Steering one variable moves the others — asymmetrically",
                 y=1.06, fontsize=10.5, color=INK)
    fig.text(0.12, -0.06, "hatched diagonal = the variable's own effect, "
             "partly true by construction; both panels share one scale",
             fontsize=7, color=MUTED)
    _save(fig, "fig4_delta_matrix")


def fig_sensitivity() -> None:
    """O-12: does the verdict survive the adjudication's free choices?"""
    d = pd.read_csv(R / "gate_sensitivity.csv")
    # Every swept factor must appear: the title claims the verdict moves for
    # EXACTLY ONE choice, and a factor left off the plot cannot support that.
    factors = [f for f in ("null_q", "null_group", "g2_rule", "fdr_q",
                           "beh_null_q", "gate_layers") if f in d.columns]
    fig, axes = plt.subplots(1, len(factors), figsize=(10.2, 2.5), sharey=True)
    for ax, f in zip(axes, factors):
        vals = sorted(d[f].unique(), key=str)
        for i, v in enumerate(vals):
            g2 = d[d[f] == v].g2
            ax.plot(np.full(len(g2), i) + np.random.default_rng(0).normal(0, .06, len(g2)),
                    g2, ".", color=S1, ms=2.2, alpha=.35)
            ax.plot(i, g2.median(), "_", color=INK, ms=16, mew=2)
        ax.set_xticks(range(len(vals)),
                      [str(v).replace("alpha_layer", "α+layer").replace("alpha", "α") for v in vals],
                      fontsize=7.5, rotation=30, ha="right")
        ax.set_title(f, fontsize=8.5, color=INK2)
        _despine(ax); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("G2: qualifying asymmetries")
    # Read the caption off the data. A hardcoded one ("G3 = 0 in all 648
    # combinations") survived a verdict reversal here and would have printed a
    # false claim onto a paper figure.
    _n = len(d)
    _g3 = d.g3_holds.mean() if "g3_holds" in d.columns else float("nan")
    _pass = (d.verdict == "PASS").mean() if "verdict" in d.columns else float("nan")
    fig.text(0.5, -0.13,
             f"{_n:,} adjudications; G3 holds in {_g3:.1%}, verdict PASS in {_pass:.1%}; "
             f"black bar = median",
             ha="center", fontsize=7.5, color=MUTED)
    fig.suptitle("The verdict moves for exactly one choice — α-pooling, which is the known error",
                 y=1.06, fontsize=10, color=INK)
    _save(fig, "fig5_sensitivity")


def fig_style() -> None:
    """E1.7: directions refitted per style, against a within-style noise floor."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, m in zip(axes, MODELS):
        d = pd.read_csv(R / "rq1" / m / "style_vs_metadata.csv")
        groups = [("R_role_tool_vs_user", "source", "role\nby style", S1),
                  ("R_harm", "source_pair", "harm\nby source", S2),
                  ("R_harm", "category", "harm\nby category", S3)]
        for i, (c, f, lab, col) in enumerate(groups):
            g = d[(d.concept == c) & (d.factor == f)]
            if not len(g):
                continue
            x = np.full(len(g), i) + np.random.default_rng(1).normal(0, .07, len(g))
            ax.plot(x, g.cosine, "o", color=col, ms=3, mfc=SURFACE, mew=.8, alpha=.75)
            ax.plot([i - .28, i + .28], [g.cosine.median()] * 2, color=INK, lw=1.8)
            fl = g.split_half_floor.iloc[0]
            if np.isfinite(fl):
                ax.plot([i - .34, i + .34], [fl, fl], color=MUTED, lw=1.4, ls=(0, (3, 2)))
        ax.set_xticks(range(3), [g[2] for g in groups], fontsize=7.5)
        ax.set_ylim(-0.15, 1.05)
        ax.set_title(NICE[m], color=INK)
        _despine(ax); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("cosine between refits")
    axes[1].plot([], [], color=MUTED, lw=1.4, ls=(0, (3, 2)), label="split-half floor (same style)")
    axes[1].plot([], [], color=INK, lw=1.8, label="median")
    axes[1].legend(fontsize=7.5, loc="lower left")
    fig.suptitle("Every direction is style-dependent: refits agree far below their own noise floor",
                 y=1.04, fontsize=10.5, color=INK)
    _save(fig, "fig6_style")


def main() -> int:
    print("writing figures to", OUT)
    for fn in (fig_depth_curves, fig_onset, fig_geometry,
               fig_delta_matrix, fig_sensitivity, fig_style):
        try:
            fn()
        except Exception as e:
            print(f"  SKIP {fn.__name__}: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
