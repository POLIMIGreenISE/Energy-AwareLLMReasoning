"""The routability ceiling.

A text router predicts the OTB benchmark's label (~0.9) but not realistic traffic's (~0.5) --
INCLUDING the prior-art router's OWN labels (0.563). Horizontal bars of within-slice text->label
CV AUC (numbers from diagnose_mohseni_routability.py + analyze_label_routability.py).
Out: results/pooled/fig_ceiling.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
INK="#1F2823"; MUT="#5C6660"; ACC="#1D6B4F"; ACC2="#3E9B76"; AMBER="#9A6A15"; RED="#8C2F1B"; RULE="#D8DCD2"
plt.rcParams.update({"font.family":"Segoe UI","font.size":11,"text.color":INK,
                     "axes.edgecolor":RULE,"axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,
                     "axes.linewidth":0.9,"figure.dpi":150})
# label, AUC, colour, group
rows = [
    ("OTB — full benchmark (our router)",        0.911, ACC),
    ("OTB — hard-math half only",                0.794, ACC2),
    ("“is it hard?” (difficulty) label",          0.620, AMBER),
    ("Mohseni — its own published labels",       0.563, RED),
    ("Mohseni Big-Math (math)",                  0.562, RED),
    ("Mohseni — our energy-aware labels",        0.552, RED),
    ("GSM8K (math)",                             0.550, RED),
    ("Mohseni Nectar (chat)",                    0.537, RED),
    ("Mohseni WildChat (chat)",                  0.437, RED),
]
labels=[r[0] for r in rows]; vals=[r[1] for r in rows]; cols=[r[2] for r in rows]
fig,ax=plt.subplots(figsize=(8.4,4.6))
ypos=range(len(rows))[::-1]
bars=ax.barh(list(ypos), vals, color=cols, edgecolor="white", linewidth=0.7, zorder=3, height=0.68)
for y,v in zip(ypos,vals):
    ax.annotate(f"{v:.2f}", (v, y), xytext=(4,0), textcoords="offset points", va="center",
                fontsize=9.5, weight="bold", color=INK)
ax.axvline(0.5, color=INK, lw=1.2, ls="--", zorder=2)
ax.annotate("chance (0.50)", (0.5, len(rows)-0.4), xytext=(3,0), textcoords="offset points",
            fontsize=8.5, color=MUT, style="italic")
ax.set_yticks(list(ypos)); ax.set_yticklabels(labels, fontsize=10)
ax.set_xlim(0.40,0.98); ax.set_xlabel("Can a text router predict the routing label?  (5-fold CV AUC)", fontsize=10)
ax.set_title("The routability ceiling: OTB's label is text-predictable (0.91); realistic traffic's is not (~0.5)\n"
             "— including the prior-art router's own labels (0.56)",
             fontsize=10.6, color=INK, weight="bold", pad=10, loc="left")
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.grid(axis="x", color=RULE, lw=0.6, alpha=0.6, zorder=0)
ax.text(0.965, 1.0, "realistic traffic\n≈ chance", color=RED, fontsize=9, ha="right", va="center", weight="bold")
fig.tight_layout(); fig.savefig("results/pooled/fig_ceiling.png", bbox_inches="tight", facecolor="white")
print("wrote results/pooled/fig_ceiling.png")
