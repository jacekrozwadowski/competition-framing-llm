#!/usr/bin/env python3
"""Regenerate the three figures of the paper from data/derived (run reproduce_tables.py first)."""

import argparse, os

os.environ.setdefault("MPLBACKEND", "Agg")
import pandas as pd, matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--derived", default="data/derived")
ap.add_argument("--out", default="paper/figs")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
D = lambda f: os.path.join(a.derived, f)
e = pd.read_csv(D("efekty_v3.csv"))
l = pd.read_csv(D("przechyl_v3.csv"))
piv = pd.read_csv(D("payoff_means.csv"), index_col=0)
diag = pd.read_csv(D("diag.csv"), index_col=0)
fam = ["bielik3-11b", "mistral-7b", "qwen2.5-3b", "qwen2.5-7b", "qwen2.5-14b"]
order = [
    m
    for m in [
        "bielik3-11b/base",
        "mistral-7b/base",
        "qwen2.5-3b/base",
        "qwen2.5-3b/instruct",
        "qwen2.5-7b/base",
        "qwen2.5-7b/instruct",
        "qwen2.5-14b/base",
        "qwen2.5-14b/instruct",
    ]
    if bool(diag.loc[m, "valid"])
]
nice = {
    "bielik3-11b": "Bielik-11B",
    "mistral-7b": "Mistral-7B",
    "qwen2.5-3b": "Qwen2.5-3B",
    "qwen2.5-7b": "Qwen2.5-7B",
    "qwen2.5-14b": "Qwen2.5-14B",
}
lab = {m: f"{nice[m.split('/')[0]]} {m.split('/')[1]}" for m in order}
n = len(order)
alpha = 0.05 / (4 * n)
fig, ax = plt.subplots(figsize=(8.6, 4.0))
v = [piv.loc[m, "equal_self"] for m in order]
ax.bar(range(n), v, color="#1f77b4")
for i, x in enumerate(v):
    ax.text(i, x + 0.01, f"{x:.2f}", ha="center", fontsize=9)
ax.axhline(0.5, ls="--", c="k", lw=0.8, alpha=0.7)
ax.set_ylim(0, 0.65)
ax.set_xticks(range(n))
ax.set_xticklabels([lab[m].replace(" ", "\n", 1) for m in order], fontsize=8)
ax.set_ylabel("Mean probability assigned to competition")
ax.set_title("Descriptive competitive-choice rate in equal-self")
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(a.out, "equal_self_rates.png"), dpi=200)
plt.close()
fig, ax = plt.subplots(figsize=(8.2, 4.2))
for i, m in enumerate(order[::-1]):
    r = e[(e.mk == m) & (e.contrast.str.contains("scarce"))].iloc[0]
    sig = r.p < alpha
    ax.errorbar(
        r.est,
        i,
        xerr=[[r.est - r.lo], [r.hi - r.est]],
        fmt="o",
        color="#1f77b4",
        mfc="#1f77b4" if sig else "white",
        capsize=4,
        ms=6,
    )
ax.set_yticks(range(n))
ax.set_yticklabels([lab[m] for m in order[::-1]])
ax.axvline(0, c="k", lw=0.8)
ax.set_xlabel("Logit contrast: scarce - abundant (higher = more competitive choice)")
ax.grid(axis="x", alpha=0.25)
ax.set_title(
    f"Scarcity-framing effect in the equal-self condition\n(open marker: p above Bonferroni threshold {alpha:.4f})",
    fontsize=11,
)
plt.tight_layout()
plt.savefig(os.path.join(a.out, "scarcity_effect.png"), dpi=200)
plt.close()
fig, ax = plt.subplots(figsize=(8.6, 5.0))
cfg = {
    "charged": ("o", "#1f77b4", 0.22),
    "label": ("s", "#ff7f0e", 0.0),
    "plain": ("^", "#2ca02c", -0.22),
}
for i, m in enumerate(order[::-1]):
    for w, (mk, c, off) in cfg.items():
        r = l[(l.mk == m) & (l.wording == w) & (l.quantity == "lean")].iloc[0]
        ax.errorbar(
            r.est,
            i + off,
            xerr=[[r.est - r.lo], [r.hi - r.est]],
            fmt=mk,
            color=c,
            capsize=3,
            ms=5,
            label=w if i == 0 else None,
        )
ax.set_yticks(range(n))
ax.set_yticklabels([lab[m] for m in order[::-1]])
ax.axvline(0, c="k", lw=0.8)
ax.set_xlabel("Estimated cooperative-description lean (logit units)")
ax.grid(axis="x", alpha=0.25)
ax.set_title("Semantic lean estimated from dominance controls")
ax.legend(title="Wording", loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(a.out, "semantic_lean.png"), dpi=200)
plt.close()
print("figures written to", a.out)
