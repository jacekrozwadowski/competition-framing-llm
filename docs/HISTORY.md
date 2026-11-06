# Version history and researcher degrees of freedom

Transparent record of what changed after seeing results. Dates: TODO (fill from your notes).

| Version | Change | Reason |
|---|---|---|
| v1 | First pilot (Qwen2.5). | Base model did not read payoffs; a "(" prefill gave A/B mass 0.29. |
| v2 | Few-shot prefix for base models, no prefill; 3B and 7B runs. | Fix of v1. |
| v3 | Bigger design (templates, controls, Bonferroni). Bootstrap replaced by template-level Welch t interval. | Bootstrap by templates was anti-conservative (10-19% false positives under the null; Welch interval: 4-6%). |
| v3.1/v3.2 | Mistral-7B added after the Qwen runs. Token-budget batching, OOM recovery, resume. Validity rule changed to: mass > 0.9 **and** label-control sign accuracy >= 0.9. Prefill ("Answer:") for Mistral-Instruct and BOS de-duplication. Bonferroni computed over all valid models. | Rule changed **after** v3 results (post hoc). Mistral-Instruct did not return a letter as first token; it was finally excluded. |
| v3.4 | Qwen2.5-14B, Bielik-11B-v3 and Gemma-4-12B (Base and Instruct) added; prefill "Reply:" for Gemma-Instruct. Validity rule unchanged since v3.2. | Wider model coverage. 7 of 12 variants valid. |
