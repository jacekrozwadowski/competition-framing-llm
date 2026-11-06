# Models, precision and environment

| Family key | Base | Base revision | Instruct | Instruct revision | Precision | Notes |
|---|---|---|---|---|---|---|
| `qwen2.5-3b` | `Qwen/Qwen2.5-3B` | `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` | `Qwen/Qwen2.5-3B-Instruct` | `aa8e72537993ba99e69dfaafa59ed015b17504d1` | FP16 | |
| `qwen2.5-7b` | `Qwen/Qwen2.5-7B` | `d149729398750b98c0af14eb82c78cfe92750796` | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | 4-bit | |
| `qwen2.5-14b` | `Qwen/Qwen2.5-14B` | `97e1e76335b7017d8f67c08a19d103c0504298c9` | `Qwen/Qwen2.5-14B-Instruct` |  `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8` |4-bit | slow on T4 |
| `mistral-7b` | `mistralai/Mistral-7B-v0.1` | `27d67f1b5f57dc0953326b2601d68371d40ea8da` | `mistralai/Mistral-7B-Instruct-v0.2` | `63a8b081895390a26e140280378bc85ec8bce07a` | 4-bit | Instruct: prefill "Answer:" / "Odpowiedź:" |
| `bielik3-11b` | `speakleash/Bielik-11B-v3-Base-20250730` | `3cdb602aaced3e481761907cfb58434c27732f8b` | `speakleash/Bielik-11B-v3.0-Instruct` | `735bfee1125fe8b497ac2769de94822a11f77167` | 4-bit | gated on Hugging Face (token required) |
| `gemma4-12b` | `google/gemma-4-12B` | `1a2c7729392586f5ecddae8807ee2ef741eabb9a` | `google/gemma-4-12B-it` | `43c9d014b231474fa7212dbb48a762566c2b255d` | FP16 | gated; Instruct: prefill "Reply:" / "Odpowiedź:", empty reasoning block in chat template |

Base models: 8-example few-shot prefix (neutral "Move 1/2" actions). Instruct models: chat template plus a one-letter instruction.

## Environment
Google Colab, GPU: NVIDIA L4 (24 GB), Python 3.13.x, CUDA 13.0

| Package | Version |
|---|---|
| torch | 2.11.0 |
| transformers | 5.22.1 |
| bitsandbytes | 0.50.2 |
| accelerate | 1.15.0 |

4-bit loading: bitsandbytes, quant type nf4, compute dtype float16 (confirm in the model-loading cell).
Full package list: `requirements.txt`.

## Run parameters

| Parameter | Value | Meaning |
|---|---|---|
| `SEED` | 0 | Seed for scenario generation and ordering |
| `BATCH` | 16 | Maximum number of prompts per batch |
| `MAX_BATCH_TOKENS` | 8192 | Token budget per batch (prompts × longest prompt); halved automatically after an OOM |
| `SHOTS_N` | 8 | Few-shot examples for Base models |
| `QUICK` | False | Full run (6 templates per language) |
| `FORCE_4BIT` | False | Mixed precision: see the table above |


## Batch-vs-single scoring check

Function: `notebooks/check_batch_vs_single.py` (run ad hoc in a separate Colab cell). Sample: 24 random prompts per model
(`seed = 0`); the same prompts are scored in batches (`BATCH = 16`) and one by one (`BATCH = 1`).
"Max diff" is the maximum absolute difference of the logit(A) - logit(B) between the two modes. The mean difference was not printed.

| Variant | Precision | A/B mass, batched | A/B mass, single | Max diff | In main analysis |
|---|---|---|---|---|---|
| Qwen2.5-3B Base | FP16 | 0.951 | 0.951 | 0.016 | yes |
| Qwen2.5-7B Base | 4-bit | 0.959 | 0.958 | 0.188 | yes |
| Qwen2.5-14B Instruct | 4-bit | 0.958 | 0.958 | 0.75 | yes |
| Mistral-7B Instruct | 4-bit | 0.987 | 0.989 | 0.601 (earlier run: 0.76) | no (excluded by validity checks) |
| Mistral-7B Base | 4-bit | 0.959 | 0.958 | 0.187 | yes |
| Bielik-11B-v3 Base | 4-bit | 0.966 | 0.968 | 0.50 | yes |
| Gemma-4-12B Instruct | FP16 | 0.667 | 0.833 | **12.665** | no (excluded by validity checks) |

Notes:
- **Gemma-4-12B Instruct shows a serious batch-vs-single discrepancy**: A/B mass 0.667 (batched) vs 0.833 (single), max logit diff 12.665 on 24 prompts.
  In single mode some prompts also have mass far below 1 (per-template means 0.0-0.67 for several templates). So both the batched scoring path and the prompt ending
  (chat template with an empty reasoning block + "Reply:" prefill) are suspect for this model; this probably explains its below-chance control accuracy (0.29) in the main log.
  The Gemma results in `data/results_raw_v3.csv` should therefore be treated as unreliable measurement, not as model behaviour.
  Gemma-4-12B Base was not checked (its batching problem, if any, is unknown).
- Mistral-7B Instruct gave 0.76 in an earlier run and 0.601 now with the same seed; the difference between runs was not investigated.
- Batch and single masses agree to within 0.002 in all checked variants except Gemma-4-12B Instruct. The largest logit differences occur in the 4-bit models.
  For Qwen2.5-14B Instruct (0.75) and Bielik-11B-v3 Base (0.50) they are of the same order as or larger than the estimated scarcity effects
  (0.43 and 0.16 logit), so batch-composition noise cannot be excluded for these variants; only the maximum over 24 prompts was printed, not the mean.
