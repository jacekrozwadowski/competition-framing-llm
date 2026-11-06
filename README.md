# Scarcity framing and semantic bias in competitive choices of open-weight LMs

Data, code and manuscript source for a small methodological study: does a sentence describing
resources as *scarce* (vs *abundant*), or the wording/actor of a payoff scenario, shift the
forced-choice (A/B) decisions of open-weight language models when the payoffs are unchanged?

**Opis po polsku:** repozytorium zawiera surowe wyniki (41 472 oceny, 12 wariantów modeli),
notebook Colab użyty do pomiaru, skrypt odtwarzający wszystkie tabele z artykułu bez GPU oraz
źródło polskiej wersji artykułu. Badanie jest wąskie i metodologiczne: nie testuje hipotezy
ewolucyjnej ani autonomicznych celów modeli.

## What is here

| Path | Content |
|---|---|
| `data/results_raw_v3.csv` | Raw output of the scoring run: one row per prompt (41 472 rows). See `docs/DATA_DICTIONARY.md`. |
| `data/derived/` | Tables recomputed from the raw file by `scripts/reproduce_tables.py` (effects, lean, diagnostics, absolute probabilities). |
| `logs/wyniki_v3.4.txt` | Original console log of the analysis run (reference for the reproduction check). |
| `notebooks/Rywalizacja_v3_4_colab.ipynb` | Colab notebook: scenario generator, GPU scoring, analysis. Written in Polish. |
| `scripts/reproduce_tables.py` | Re-runs the notebook's analysis cells on the CSV (CPU only) and checks them against the log. |
| `scripts/make_figures.py` | Regenerates the figures from `data/derived/`. |
| `paper/` | Manuscript source (Polish, LaTeX) and figures. |
| `docs/` | Data dictionary, version history, model list. |

## Reproduce the tables (no GPU, a few minutes)

```bash
pip install numpy pandas scipy matplotlib
python scripts/reproduce_tables.py --log logs/wyniki_v3.4.txt   # exits 0 if all 422 report lines match
python scripts/make_figures.py                                  # optional
```

The analysis recomputes everything from the raw log-probabilities in the CSV
(pair logits, template-level Welch intervals, Bonferroni threshold, lean/numbers decomposition).

## Re-run the measurement (GPU)

Open the notebook in Colab with a T4 or better GPU, set `QUICK = True` for a smoke test, then
`QUICK = False`. Cells must be run in order. The run is resumable (`results_partial_v3.csv`).
Gemma and Bielik are gated on Hugging Face: accept their licenses and add an `HF_TOKEN`
secret. No token or personal data is stored in this repository.
Exact package and model revisions: see `docs/models.md`.

## Key design facts

- Task: one-shot A/B choice, competitive vs cooperative option; both option orders for every
  scenario; score = logit of the competitive option from next-token log-probabilities.
- 12 model variants (Qwen2.5 3B/7B/14B, Mistral-7B, Bielik-11B-v3, Gemma-4-12B; Base and Instruct);
  7 pass the validity rule (A/B mass > 0.9 and label-wording control sign accuracy >= 0.9) and
  enter the main analysis. The rule was introduced post hoc (v3.2) and left unchanged afterwards.
  The scarcity contrast for all 12 is reported in the paper's appendix.
- Gemma-4-12B (Base and Instruct) is excluded, and its numbers should be read as unreliable
  measurement, not model behaviour: a batch-vs-single scoring check for Instruct showed a large
  discrepancy (see `docs/models.md`).
- Inference unit: the paraphrase template (6 per language), not the prompt.
- Validity rule and several design decisions were changed after seeing earlier results;
  see `docs/HISTORY.md` and the Limitations section of the paper.

## Citation and licenses

Cite via `CITATION.cff` (add the Zenodo DOI after the first release).
Code (`scripts/`, `notebooks/`): MIT, see `LICENSE`.
Data, logs, tables, figures, paper text, docs: CC BY 4.0, see `LICENSE-DATA.md` (full legal code in `LICENSE-CC-BY-4.0.txt`).
Model weights are not included and remain under their own licenses.