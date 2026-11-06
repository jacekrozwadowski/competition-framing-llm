"""Diagnostic cell: batched vs single-prompt scoring (numerical noise check).

NOT a standalone script. It reuses globals defined by the notebook's SCORING cell
(MODELS, FOUR_BIT, INSTRUCT_PREFILL, items, load, variant_ids, make_text, score, free_gpu, BATCH)
and needs a GPU. Run it in a Colab cell right after the scoring cell.

Reading the output: if the masses and the logit(A)-logit(B) differences agree (about 0.05 for fp16),
the batched scoring path is confirmed. Reported so far (see docs/models.md):
  qwen2.5-3b base (fp16):           max |diff| = 0.016
  mistral-7b instruct (4-bit):      max |diff| = 0.76
Not yet run for the other 4-bit models used in the main analysis
(qwen2.5-7b, qwen2.5-14b, mistral-7b base, bielik3-11b).
This code was run ad hoc in Colab and is archived here unchanged.
"""
import numpy as np, pandas as pd   # already imported in the notebook

def check(mkey, kind, n=24, seed=0):
    global BATCH
    base_name, inst_name = MODELS[mkey]
    tok, model = load(inst_name if kind == "instruct" else base_name, four_bit=(mkey in FOUR_BIT))
    idsA, idsB = variant_ids(tok, "A"), variant_ids(tok, "B")
    sub = items.sample(n, random_state=seed)
    pf = INSTRUCT_PREFILL.get(mkey, {}) if kind == "instruct" else {}
    texts = [make_text(tok, r, kind, pf.get(r.lang, "")) for r in sub.itertuples()]
    old = BATCH
    BATCH = 16; lp_b = score(model, tok, texts, idsA, idsB)
    BATCH = 1;  lp_1 = score(model, tok, texts, idsA, idsB)
    BATCH = old
    m_b, m_1 = np.exp(lp_b).sum(1), np.exp(lp_1).sum(1)
    d = (lp_b[:, 0] - lp_b[:, 1]) - (lp_1[:, 0] - lp_1[:, 1])
    print(mkey, kind, "| masa: paczki", m_b.mean().round(3), "pojedynczo", m_1.mean().round(3),
          "| max |Δ logit(A−B)|", np.abs(d).max().round(3))
    print(pd.DataFrame({"lang": sub.lang.values, "T": sub.template.values, "masa_pojedynczo": m_1})
            .groupby(["lang", "T"]).masa_pojedynczo.mean().round(2).unstack().to_string())
    del model, tok; free_gpu()


# calls that were run:
check("mistral-7b", "instruct")
check("qwen2.5-3b", "base")

# not yet run (uncomment in Colab):
# check("qwen2.5-7b", "base")
# check("qwen2.5-14b", "instruct")
# check("mistral-7b", "base")
# check("bielik3-11b", "base")
