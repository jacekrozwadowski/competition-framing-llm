# Data dictionary: `data/results_raw_v3.csv`

One row per scored prompt: 3456 prompts x 12 model variants = 41 472 rows.

| Column | Values | Meaning |
|---|---|---|
| `sid` | e.g. `en\|tribe\|charged\|coop_dominant\|scarce\|T0\|x1` | Scenario id (language, actor, wording, payoff structure, context, template, scale). The two option orders of one scenario share the same `sid`. |
| `lang` | `en`, `pl` | Prompt language. |
| `actor` | `tribe`, `team`, `agent` | How the two parties are named. |
| `wording` | `charged`, `plain`, `label` | Description of the two actions: charged ("seize control ... push the other side out" vs "join forces ..."), plain ("act independently" vs "act jointly"), or neutral labels ("Action X/Y"). |
| `payoff` | `comp_dominant`, `coop_dominant`, `equal_self`, `dom_comp`, `dom_coop` | Payoff structure (self, other). `equal_self` is the main test; `dom_comp`/`dom_coop` are dominance controls. Values: see Table 1 of the paper. |
| `ctx` | `scarce`, `abundant` | One context sentence: resources are scarce / abundant. Payoffs identical. |
| `template` | 0-5 | Paraphrase template index (6 per language). |
| `scale` | 1, 2 | Payoff scale (controls use only 1). |
| `order` | `comp_first`, `coop_first` | Whether the competitive option is presented first. |
| `comp_letter` | `A`, `B` | Letter assigned to the competitive option (follows from `order`). |
| `model` | `qwen2.5-3b`, `qwen2.5-7b`, `qwen2.5-14b`, `mistral-7b`, `bielik3-11b`, `gemma4-12b` | Model family. |
| `kind` | `base`, `instruct` | Base model (few-shot prefix) or instruction-tuned (chat template). |
| `lpA`, `lpB` | float | Natural-log of the total next-token probability of answer letter A / B (log-sum-exp over the single-token variants "A", " A", "▁A", "ĠA"). |

Derived quantities (computed by the notebook / `reproduce_tables.py`):

- `mass = exp(lpA) + exp(lpB)`: probability mass on the two valid answers (must be > 0.9).
- `pA = 1 / (1 + exp(lpB - lpA))`; `p_comp = pA` if `comp_letter == 'A'` else `1 - pA`;
  clipped to [1e-4, 1-1e-4].
- `z = logit(p_comp)`; the pair logit is the mean of `z` over the two option orders.
- Effects: template-level contrasts averaged over the two languages, Welch t interval over the
  6 templates per language; Bonferroni alpha = 0.05 / (4 x number of valid variants).
- `numbers = (m_coop + m_comp) / 2`, `lean = (m_coop - m_comp) / 2` from the dominance controls.
