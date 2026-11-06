#!/usr/bin/env python3
"""Recompute all tables of the paper from data/results_raw_v3.csv (no GPU needed).

It executes the analysis cells (config, generator, analysis A/B/C) of the Colab notebook
unchanged, with Colab/HF login stubbed out, and then computes a few extra descriptive tables.

Usage:
    python scripts/reproduce_tables.py [--csv data/results_raw_v3.csv]
        [--notebook notebooks/Rywalizacja_v3_4_colab.ipynb] [--outdir data/derived]
        [--log logs/wyniki_v3.4.txt]
Requires: numpy, pandas, scipy, matplotlib (see requirements.txt).
"""

import argparse, contextlib, io, json, os, re, shutil, sys, types

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np, pandas as pd
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="data/results_raw_v3.csv")
ap.add_argument("--notebook", default="notebooks/Rywalizacja_v3_4_colab.ipynb")
ap.add_argument("--outdir", default="data/derived")
ap.add_argument(
    "--log",
    default=None,
    help="optional: compare the printed report with the original log",
)
a = ap.parse_args()
csv, nbp, outdir = map(os.path.abspath, (a.csv, a.notebook, a.outdir))
log = os.path.abspath(a.log) if a.log else None
os.makedirs(outdir, exist_ok=True)
shutil.copy(csv, os.path.join(outdir, "results_raw_v3.csv"))
os.chdir(outdir)  # the notebook reads/writes files in the working directory

# stubs for Colab-only imports used by the config cell
g = types.ModuleType("google")
gc = types.ModuleType("google.colab")
gc.userdata = types.SimpleNamespace(get=lambda k: "unused")
g.colab = gc
sys.modules.update({"google": g, "google.colab": gc})
hf = types.ModuleType("huggingface_hub")
hf.login = lambda *a, **k: None
sys.modules["huggingface_hub"] = hf

cells = ["".join(c["source"]) for c in json.load(open(nbp, encoding="utf8"))["cells"]]


def find(marker):
    hit = [c for c in cells if c.lstrip().startswith(marker)]
    assert len(hit) == 1, marker
    return hit[0]


parts = [
    find("# ===== KONFIGURACJA"),
    find("# ===== GENERATOR"),
    find("# ===== ANALIZA A"),
    find("# ===== ANALIZA B"),
    find("# ===== ANALIZA C"),
]
ns, buf = {}, io.StringIO()
with contextlib.redirect_stdout(buf):
    for k, src in enumerate(parts):
        exec(compile(src, f"notebook_part{k}", "exec"), ns)
report = buf.getvalue()
open("report.txt", "w", encoding="utf8").write(report)

# extra descriptive tables (absolute P(comp), per-language scarcity effect) in equal_self
pair, diag = ns["pair"], ns["diag"]
diag.to_csv("diag.csv")
pair.groupby(["mk", "payoff"]).p_comp.mean().unstack().to_csv("payoff_means.csv")
es = pair[pair.payoff == "equal_self"]
rows = {}
for mk in sorted(pair.mk.unique()):
    E = es[es.mk == mk]
    pm = E.groupby("ctx").p_comp.mean()
    T = E.groupby(["tid", "ctx"]).pair_logit.mean().unstack("ctx")
    T["eff"] = T.scarce - T.abundant
    lang = pd.Series({t: t.split("_")[0] for t in T.index})
    en, pl = T.eff[lang == "en"].values, T.eff[lang == "pl"].values
    d = pl.mean() - en.mean()
    va, vb = pl.var(ddof=1) / 6, en.var(ddof=1) / 6
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / 5 + vb**2 / 5)
    tc = stats.t.ppf(0.975, df)
    rows[mk] = dict(
        P_comp_scarce=pm["scarce"],
        P_comp_abundant=pm["abundant"],
        scarcity_effect_en=en.mean(),
        scarcity_effect_pl=pl.mean(),
        pl_minus_en=d,
        ci_lo=d - tc * se,
        ci_hi=d + tc * se,
        p_nominal=2 * stats.t.sf(abs(d / se), df),
        valid=bool(mk in ns["VALID"]),
    )
pd.DataFrame(rows).T.to_csv("scarcity_abs_lang.csv")
os.remove(
    os.path.join(outdir, "results_raw_v3.csv")
)  # temporary copy used by the notebook code
print("Wrote report.txt and tables to", outdir)

if log:
    norm = lambda s: [
        re.sub(r"\s+", " ", x).strip() for x in s.splitlines() if x.strip()
    ]
    ref = open(log, encoding="utf8").read()
    ref = ref[ref.index("1) DIAGNOSTYKA") :]
    new = report[report.index("1) DIAGNOSTYKA") :]
    diff = [(i, x, y) for i, (x, y) in enumerate(zip(norm(ref), norm(new))) if x != y]
    print(
        f"Comparison with {os.path.basename(log)}: {len(norm(ref))} vs {len(norm(new))} lines, {len(diff)} differ"
    )
    sys.exit(1 if diff or len(norm(ref)) != len(norm(new)) else 0)
