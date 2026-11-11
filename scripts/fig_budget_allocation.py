"""Budget-allocation figures: holding an energy budget, where the load-aware rule gains accuracy,
and which ingredient the gain comes from.

    python scripts/fig_budget_allocation.py  ->  figures/e2_sweep.png
                                                 figures/e8_gap.png
                                                 figures/e8_decomp.png
                                                 figures/e8_slo.png
                                                 figures/e11_real.png

Set FIG_DIR to write the figures somewhere else.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.environ.get("FIG_DIR", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#d8d8d4", "#ffffff"
TEAL, OCHRE = "#0b8f86", "#b0591a"          # validated CVD-safe diverging pair

plt.rcParams.update({"font.size": 8, "axes.edgecolor": GRID, "axes.linewidth": .8,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": MUTED,
                     "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "font.family": "DejaVu Sans", "axes.titlesize": 8.5})

CAPS = ["25%", "40%", "55%", "70%"]

# MEASURED: results/e8/e8_rows.json (scripts/e8_consolidated.py) -- deployable load-aware minus
# static threshold at matched energy, with the 95% paired bootstrap interval.
#
# These are LOADED and not transcribed. An earlier version of this script carried all three gap
# tables by hand; when the comparison was re-run against src/capfit.py the results files changed and
# the figures did not, so three published figures disagreed with the tables beside them. Anything a
# re-run can change is read from the re-run's own output.
import json as _json
_G = _json.load(open("results/e8/e8_rows.json"))
GAP = {rg: [(r["gap"], r["lo"], r["hi"]) for r in rows] for rg, rows in _G["gap"].items()}
CV = {"near-idle": 0.51, "moderate": 0.48, "saturated": 0.33, "bursty": 1.53}


def limits(*groups, pad=0.35):
    """Shared y-range wide enough for every error bar in the figure."""
    vals = [b for g in groups for row in g for (_, lo, hi) in row for b in (lo, hi)]
    return {"ymin": min(vals) - pad, "ymax": max(vals) + pad}


# MEASURED: results/e2/e2_transfer_tokens.json (scripts/e2_transfer_tokens.py) -- energy actually
# consumed against the authorized budget (%), budget = never + 35% of (always - never), lambda bisected
# on 5 historical traces and deployed on the 20 E8 evaluation traces. Left panel: fixed price C with true
# token counts. Right panel: the load-aware model under three token models.
import json as _json
_E2 = _json.load(open("results/e2/e2_transfer_tokens.json"))["rows"]
SWEEP_C = [3.42, 2.40, 1.70, 1.20, 0.865, 0.60, 0.454, 0.244, 0.13]
REG4 = ["near-idle", "moderate", "saturated", "bursty"]
SWEEP = {rg: [next(x["vs_budget"] for x in _E2 if x["regime"] == rg and x["frac"] == 0.35 and x["tokens"] == "true"
                   and x["model"] == "fixed" and x["C"] == c) for c in SWEEP_C] for rg in REG4}
TOKMODELS = ["true", "constant", "estimated"]
SWEEP_LOADAWARE = {rg: [next(x["vs_budget"] for x in _E2 if x["regime"] == rg and x["frac"] == 0.35
                             and x["tokens"] == t and x["model"] == "load-aware") for t in TOKMODELS] for rg in REG4}
# Categorical slots 1-4 of the dataviz reference palette, validated on white on the adjacent pairlist that
# the skill specifies for lines (worst CVD dE 9.1, normal-vision 22.9). Slot-to-regime mapping is chosen so
# the one pair that fails all-pairs (orange/yellow) goes to near-idle and saturated, two lines that never
# cross. Aqua and yellow sit below 3:1 on white, so every line is direct-labelled and has its own marker.
SERIES = {"moderate": ("#2a78d6", "o"), "near-idle": ("#eb6834", "s"),
          "bursty": ("#1baf7a", "^"), "saturated": ("#eda100", "D")}
TOK_MARK = {"true": "o", "constant": "s", "estimated": "^"}
# the measured energy per token at these batch sizes (0.134 J at 32 is plotted at 0.134).
CURVE_L = {1: 3.42, 2: 1.70, 4: 0.865, 8: 0.454, 16: 0.244, 32: 0.134}


def panel(ax, rows, labels, title, sub=None, ymin=None, ymax=None):
    xs = np.arange(len(rows))
    for x, (v, lo, hi) in zip(xs, rows):
        ax.bar(x, v, width=.62, color=TEAL if v >= 0 else OCHRE, zorder=3)
        ax.plot([x, x], [lo, hi], color=INK, lw=1.0, zorder=4)
        for b in (lo, hi):
            ax.plot([x - .1, x + .1], [b, b], color=INK, lw=1.0, zorder=4)
    ax.axhline(0, color=MUTED, lw=1.0, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlim(-.65, len(rows) - .35)
    if ymin is not None:
        ax.set_ylim(ymin, ymax)
    # With a subtitle the title has to clear a second line of text, not just the axes frame.
    ax.set_title(title, color=INK, loc="left", pad=17 if sub else 8)
    if sub:
        ax.text(0, 1.005, sub, transform=ax.transAxes, color=MUTED, fontsize=7,
                va="bottom")
    ax.grid(True, axis="y", color=GRID, lw=.6, alpha=.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ============================================================ figure: the calibration sweep
fig, (ax, axr) = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True,
                              gridspec_kw={"width_ratios": [3.4, 1.25], "wspace": 0.06})
ax.axhspan(-10, 10, color=GRID, alpha=.45, lw=0, zorder=1)
ax.axhline(0, color=MUTED, lw=1.0, zorder=2)
# Direct labels sit where each line is clear of the others (checked by eye on the render). The grey band
# is explained in the caption rather than labelled, because every line crosses it.
LABEL_AT = {"bursty": (0.30, -7, "top"), "near-idle": (0.30, 5, "bottom"),
            "moderate": (0.87, -6, "top"), "saturated": (1.20, 5, "bottom")}
for name in REG4:
    ys = SWEEP[name]
    col, mk = SERIES[name]
    ax.plot(SWEEP_C, ys, color=col, lw=2.0, marker=mk, ms=5.5, mec=SURF, mew=1.0, zorder=4, label=name)
    x0, dy, va = LABEL_AT[name]
    y0 = float(np.interp(np.log(x0), np.log(SWEEP_C[::-1]), ys[::-1]))
    ax.text(x0, y0 + dy, name, color=INK, fontsize=7.5, ha="center", va=va, zorder=5)
ax.set_xscale("log")
ax.set_xlim(0.105, 3.9)
ax.set_xticks(SWEEP_C)
ax.set_xticklabels(["3.42", "2.40", "1.70", "1.20", "0.87", "0.60", "0.45", "0.24", "0.13"])
ax.minorticks_off()
ax.set_xlabel("fixed energy cost assumed per token (J)")
ax.set_ylabel("energy spent against budget (%)")
ax.set_ylim(-62, 122)
top = ax.secondary_xaxis("top")
top.set_xticks(list(CURVE_L.values()))
top.set_xticklabels([str(k) for k in CURVE_L])
top.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
top.tick_params(colors=MUTED, labelsize=7)
top.set_xlabel("batch size at which the measured curve gives that energy", color=MUTED, fontsize=7.5)
top.spines["top"].set_color(GRID)
ax.legend(loc="upper right", frameon=False, fontsize=7.5, handlelength=2.2, borderaxespad=0.2)
ax.grid(True, axis="y", color=GRID, lw=.6, alpha=.8)
ax.set_axisbelow(True)
ax.spines["right"].set_visible(False)

axr.axhspan(-10, 10, color=GRID, alpha=.45, lw=0, zorder=1)
axr.axhline(0, color=MUTED, lw=1.0, zorder=2)
# Token model is carried by marker shape in ink, so this panel adds no colours of its own.
for i, name in enumerate(REG4):
    for j, t in enumerate(TOKMODELS):
        axr.plot([i + (j - 1) * 0.22], [SWEEP_LOADAWARE[name][j]], color=INK, marker=TOK_MARK[t], ms=4.5,
                 mfc=INK if t == "true" else SURF, mec=INK, mew=0.9, ls="none", zorder=4,
                 label=t if i == 0 else None)
worst = min(((i, j) for i in range(4) for j in range(3)), key=lambda ij: SWEEP_LOADAWARE[REG4[ij[0]]][ij[1]])
wv = SWEEP_LOADAWARE[REG4[worst[0]]][worst[1]]
axr.text(worst[0] + (worst[1] - 1) * 0.22, wv - 4, ("%+.1f" % wv).replace("-", "−"), color=INK,
         fontsize=6.5, ha="center", va="top")
axr.set_xticks(range(4))
axr.set_xticklabels(["idle", "mod.", "sat.", "burst"], rotation=0, fontsize=6.5)
axr.set_xlim(-.55, 3.55)
axr.legend(loc="upper center", frameon=False, fontsize=6.5, handlelength=1.0, borderaxespad=0.2,
           title="token count", title_fontsize=6.5)
axr.set_title("load-aware,\nno calibration", color=INK, loc="left", fontsize=8, pad=7)
axr.grid(True, axis="y", color=GRID, lw=.6, alpha=.8)
axr.set_axisbelow(True)
for s in ("top", "right"):
    axr.spines[s].set_visible(False)
axr.tick_params(axis="y", left=False)
fig.subplots_adjust(left=0.085, right=0.985, bottom=0.14, top=0.80)
fig.savefig(os.path.join(FIG_DIR, "e2_sweep.png"), dpi=300, facecolor=SURF)
print("wrote", os.path.join(FIG_DIR, "e2_sweep.png"))

# ============================================================ figure: gap by regime
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.9), sharey=True)
for ax, name in zip(axes, ["near-idle", "moderate", "saturated", "bursty"]):
    panel(ax, GAP[name], CAPS, name, sub="load CV %.2f" % CV[name], **limits(GAP.values()))
axes[0].set_ylabel("accuracy gap (pp)\nload-aware $-$ static")
for ax in axes:
    ax.set_xlabel("energy cap")
fig.tight_layout(pad=0.6, w_pad=1.0)
fig.savefig(os.path.join(FIG_DIR, "e8_gap.png"), dpi=300, facecolor=SURF)
print("wrote", os.path.join(FIG_DIR, "e8_gap.png"))

# ============================================================ figure: decomposition
# MEASURED: results/e8/e8_decomp_caps.json (scripts/e8_decomp_caps.py) -- all four caps. Ingredient is
# identity, so it takes categorical slots 1-3 (validated all-pairs on white); sign is read from the zero line.
_D = _json.load(open("results/e8/e8_decomp_caps.json"))
ING = [("static-token", "+ tokens"), ("load-aware-const", "+ load"), ("load-aware", "+ both")]
ING_COL = ["#2a78d6", "#eb6834", "#1baf7a"]
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharey=True)
for ax, name in zip(axes.flat, REG4):
    for k, (pol, lab) in enumerate(ING):
        xs, vs, los, his = [], [], [], []
        for ci, frac in enumerate((0.25, 0.40, 0.55, 0.70)):
            r = next(x for x in _D if x["regime"] == name and abs(x["frac"] - frac) < 1e-9 and x["policy"] == pol)
            xs.append(ci + (k - 1) * 0.27); vs.append(r["gap"]); los.append(r["lo"]); his.append(r["hi"])
        ax.bar(xs, vs, width=0.24, color=ING_COL[k], zorder=3, label=lab)
        for x, v, lo, hi in zip(xs, vs, los, his):
            ax.plot([x, x], [lo, hi], color=INK, lw=0.8, zorder=4)
    ax.axhline(0, color=MUTED, lw=1.0, zorder=2)
    ax.set_xticks(range(4))
    ax.set_xticklabels(CAPS)
    ax.set_xlim(-.55, 3.55)
    ax.set_ylim(-4.5, 6.0)
    ax.set_title(name, color=INK, loc="left", pad=6)
    ax.grid(True, axis="y", color=GRID, lw=.6, alpha=.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
for ax in axes[1]:
    ax.set_xlabel("energy cap")
for ax in axes[:, 0]:
    ax.set_ylabel("accuracy gap (pp)\nvs static threshold")
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(pad=0.6, w_pad=1.2, h_pad=1.4, rect=(0, 0, 1, 0.95))
fig.savefig(os.path.join(FIG_DIR, "e8_decomp.png"), dpi=300, facecolor=SURF)
print("wrote", os.path.join(FIG_DIR, "e8_decomp.png"))

# ============================================================ figure: the SLO reversal
# MEASURED: results/e8/CONSOLIDATED.md section 2 -- energy cap held at 40%, a p95 target added
# alongside it as a multiple of the never-think p95 in that regime.
SLO = {rg: [(r["gap"], r["lo"], r["hi"]) for r in rows] for rg, rows in _G["slo"].items()}
TARGETS = ["1.5×", "3×", "6×", "none"]

fig, axes = plt.subplots(1, 4, figsize=(7.2, 3.0), sharey=True)
for ax, name in zip(axes, ["near-idle", "moderate", "saturated", "bursty"]):
    panel(ax, SLO[name], TARGETS, name, **limits(SLO.values()))
    ax.set_xlabel("p95 target")
axes[0].set_ylabel("accuracy gap (pp)\nload-aware $-$ static")
fig.tight_layout(pad=0.6, w_pad=1.0)
fig.savefig(os.path.join(FIG_DIR, "e8_slo.png"), dpi=300, facecolor=SURF)
print("wrote", os.path.join(FIG_DIR, "e8_slo.png"))

# ============================================================ figure: production arrival traces
# MEASURED: results/e11/REAL_TRACE.md section 2. Azure LLM Inference Traces (Nov 2023) supply the
# arrival process only; token counts and correctness stay with the paired OTB measurements.
# (title, subtitle, gaps). The two code panels are one trace under two rescalings, so they share a
# title and the subtitle says which rescaling; that also keeps every title short.
_R = _json.load(open("results/e11/e11_rows.json"))
_E11 = [("Poisson @0.8 (control)",     "Poisson (control)", ""),
        ("Azure conv @0.8",            "Azure conv",        ""),
        ("Azure code @0.8",            "Azure code",        ""),
        ("Azure code @0.8 (thinned)",  "Azure code",        "thinned, ")]
REAL = [(title, "%sload CV %.2f" % (note, _R["shape"][key]["load_cv"]),
         [(r["gap"], r["lo"], r["hi"]) for r in _R["gaps"][key]])
        for key, title, note in _E11]

fig, axes = plt.subplots(1, 4, figsize=(7.2, 3.0), sharey=True)
for ax, (title, sub, rows) in zip(axes, REAL):
    panel(ax, rows, CAPS, title, sub=sub, **limits([r for _, _, r in REAL]))
    ax.set_xlabel("energy cap")
axes[0].set_ylabel("accuracy gap (pp)\nload-aware $-$ static")
fig.tight_layout(pad=0.6, w_pad=1.0)
fig.savefig(os.path.join(FIG_DIR, "e11_real.png"), dpi=300, facecolor=SURF)
print("wrote", os.path.join(FIG_DIR, "e11_real.png"))
