# Environment — verified facts

Durable properties of the models, chat templates, datasets and cluster. These are
**not results**: nothing here depends on an experimental finding, and nothing here
is invalidated when a run is thrown away.

**The rule for this file.** An entry may be recorded here only if it is a property
of an artifact we did not produce — a published config, a chat template, a dataset
schema, a library behaviour, a cluster policy — and it must carry how it was
established and how to re-check it. **No quantity that required a forward pass
through a model on our data belongs here.** Those go in `results/` behind a
manifest.

**Verification status.**

| Tag | Meaning |
|---|---|
| `published` | Read from a model config, dataset card, or library source. Re-check by reading it again |
| `offline` | Established by running a tokeniser or template locally. No GPU, seconds to re-check |
| `cluster` | Observed cluster behaviour |
| `unverified` | Asserted in the superseded playbook, cheap to check, **not yet re-checked** |

Everything tagged `offline` below was established by the removed code. The facts
are almost certainly right and the recipes are given, but each carries `offline`
rather than `confirmed` until the rebuilt code re-checks it. **Stage 0 re-checks
all of them as a matter of course.**

---

## Models

`published` — read from the HuggingFace configs.

| Model | Architecture | Layers | `d_model` | Experts |
|---|---|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | `Qwen2ForCausalLM` | **28** | 3584 | — |
| `Qwen/Qwen3.5-9B` | `Qwen3_5ForConditionalGeneration` | **32** | 4096 | — |
| `Qwen/Qwen3.5-35B-A3B` | `Qwen3_5MoeForConditionalGeneration` | **40** | 2048 | **256** |

### Qwen3.5's config is nested and its top-level shape fields are `None`

`published`. `config.num_hidden_layers` and `config.hidden_size` are **`None`** on
both Qwen3.5 checkpoints; the real values live under `config.text_config`. Code
doing `model.config.hidden_size` silently receives `None`.

**Requirement:** resolve shape through `config.text_config`, and locate the decoder
stack **by search**, raising if it cannot be found — never assume
`model.model.layers`. A hook on the wrong module produces plausible-looking
numbers, which is the worst failure mode available.

### Both Qwen3.5 checkpoints are multimodal wrappers

`published`. Both are `*ForConditionalGeneration` with a `vision_config`, whereas
the results we are building on (Zhao, NeuroStrike) are on text-only models. This
is why the roster carries a text-only model alongside: agreement between them
shows the vision tower is not driving a result, disagreement localises it.

PLAN-SCOPE lists **multimodal safety** as out of scope. Running a
vision-capable checkpoint in text-only mode is not multimodal safety research and
does not violate that; it does mean the vision tower is an uncontrolled difference
between the two dense models, which is the reason for the pairing.

### Layer indexing convention

`published` + convention. **Layer index `l` denotes the output of decoder block
`l`.** Index 0 is therefore *after* block 0's attention and MLP have run — it is
not the embedding output. Embeddings are not captured.

Consequence: high linear separability at index 0 is expected and is **not**
evidence that a probe is reading token identity. Any lexical-shortcut check must
use a length-only baseline and a matched random direction instead.

---

## Chat templates

All `offline`. Re-check with `tokenizer.apply_chat_template(msgs, tokenize=False)`
on each model.

### `tool` is not a role tag on either model

Both templates render a tool message as a **user** turn wrapped in a
`<tool_response>` block:

```
<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>
```

There is no `<|im_start|>tool`. Hand-building one would feed the model a token
sequence it never saw in training, so the measurement would be of its response to
an out-of-distribution string rather than of its role representation.

**Consequence for the design:** the tool/user contrast is a *structural wrapper
inside a user turn*, not a different tag. This is the real injection surface and a
sharper contrast than an invented tag would be. It also means the wrapper adds
tokens deterministically, which is a length confound that must be controlled.

### `enable_thinking=False` does not remove thinking on Qwen3.5

It inserts an **empty think block**. The prompt ends
`<|im_start|>assistant\n<think>\n\n</think>\n\n` rather than
`<|im_start|>assistant\n`. Qwen2.5 accepts the kwarg and ignores it.

Consequences: `t_post-inst` sits in a different textual context on the two models,
and an assistant-role instruction on Qwen3.5 carries this block ahead of it while
other role classes do not.

The kwarg must still be passed — Qwen3 emits `<think>` otherwise, and a short
generation would be judged on the reasoning preamble. NeuroStrike does the same
for Qwen3.

### Qwen3.5 forces the system message to position 0

`apply_chat_template` raises `"System message must be at the beginning"`. The
system class therefore **cannot occupy a common message slot** on that model;
`user`, `assistant` and `tool` can. System must be rendered in its natural
position and the deviation reported.

### Roles do not share a conversational position

Each role's natural slot differs — system opens, user follows, assistant follows a
user turn, tool follows an assistant turn. Left uncontrolled, a role probe can
separate the classes by **turn position and context length** without representing
role at all. This is the reason for the fixed-slot / natural-slot contrast in the
corpus design.

### Rendered length is not constant across roles, even in a fixed slot

The authentic markup differs: the `<tool_response>` wrapper adds tokens, and
Qwen3.5's empty think block adds more ahead of an assistant turn. **Length-only
baselines are mandatory** for anything fitted across role classes.

`unverified` — the specific per-role token counts recorded in the superseded
playbook are not carried forward. Stage 0 re-measures the rendered-length
distribution per (model, role, design) and writes it to the corpus report.

### Cross-role tokenisation of the instruction is not exactly invariant

`unverified`, previously measured at a small non-zero rate. In the tool rendering
the content is followed by `\n</tool_response>` rather than `<|im_end|>`, so the
final instruction token can merge differently — e.g. content ending in punctuation
followed by a newline.

**Requirement: this invariant is measured corpus-wide, never asserted.** Stage 0
sweeps every (instruction, design) pair, reports the mismatch rate, and writes the
affected uids to `tokenisation_mismatches.csv` so analyses needing exact
token-level matching can exclude them. The acceptable rate is pre-registered in
`EXPERIMENTS.md`, not chosen after seeing it.

*General lesson, and the reason this entry exists: the original check ran on a
single sample instruction and passed vacuously. **Every corpus-wide invariant is
checked corpus-wide.***

---

## Token positions

`offline`. Definitions used throughout, verified against rendered ChatML.

| Position | Definition |
|---|---|
| `t_inst` | Last content token of the instruction-bearing turn — that turn's `<\|im_end\|>` minus 1 — whatever role tag it carries |
| `t_post-inst` | Last token of the templated prompt, i.e. the position the first generated token attends from |
| content span | The instruction's own tokens, role tags and template markup excluded |

**Position indexing must be per-example.** `t_inst` is not index `-1`; with
left-padding it differs per row. Every extraction records the index it used, and
positions are verified against **real padded batches**, not single unpadded
examples.

---

## Datasets

`published` — availability and schemas verified against the Hub.

| Pool | Purpose | Notes |
|---|---|---|
| AdvBench | harmful | 520 rows, cols `prompt`, `target` |
| JBB-Behaviors | harmful | config `behaviors`. **Columns are capitalised** (`Goal`, not `goal`) — case-sensitive matching drops the source silently |
| Sorry-Bench | harmful | config `default`, 9,240 = 440 base × 21 styles. **Use `prompt_style == "base"` only** — see below |
| Alpaca | harmless | 52,002; use only `input == ""` rows so each is standalone |
| XSTest | benign-but-**sensitive** | `walledai/XSTest`, split `test`, 450 rows with an explicit `label` column (250 safe / 200 unsafe), read rather than inferred. Mirror `natolambert/xstest-v2-copy` derives safety from `type` — the eight `contrast_*` types are the unsafe ones, verified equivalent |
| StrongREJECT | held-out attack eval | 313 rows |
| C4 | role constant-content | config `en`. *Passing `en` as a split is the error that makes it look unavailable.* Streamed |
| Dolma3 | second role source | not checked; C4 suffices to start |

### Sorry-Bench must be filtered to `prompt_style == "base"`

This is a **scientific decision, not a measurement**, and it survives every purge.
The 9,240 rows are 440 base prompts × 21 mutation styles. Taking them
indiscriminately is wrong three ways:

- `role_play`, `authority_endorsement`, `expert_endorsement`, `logical_appeal`,
  `evidence-based_persuasion`, `misrepresentation` are **jailbreak framings**.
  Fitting `R_harm`/`R_control` on them makes RQ4's jailbreak family partly
  circular and violates PLAN-CPE's *"completely disjoint attack intents"*.
- `translate-fr|ml|mr|ta|zh-cn` are **multilingual**, explicitly out of scope.
- `ascii`, `atbash`, `caesar`, `morse` are **encoded strings**, not natural harmful
  instructions — the model may not parse them as harmful at all.

It also destroys the length distribution: unfiltered, harmful prompts run many
times longer than harmless ones, so `R_harm` would substantially encode *length*.

**The 20 excluded styles are a reserved resource, not waste.** They are held out,
already paired with their base prompt, and are the natural material for RQ4's
jailbreak family and RQ6's controlled context variants. They must never enter a
fitting set.

### The harmful class is deliberately heterogeneous

Even filtered to `base`, Sorry-Bench spans 44 categories across a wide severity
range — grooming and death threats sit beside retirement-account advice and zombie
fiction. It is built to probe *over*-refusal as much as refusal.

**This is not a defect to filter away.** Borderline items are what populate the
harmful-and-complied cell, without which `R_control` is unidentifiable. But the
positive class of `R_harm` is therefore not uniform, so `category` is recorded per
instruction (JBB `Category`, Sorry-Bench `cat*`, XSTest `type`) and **separation is
reported per source and per category, never only pooled**. A severity gradient is a
finding, not a bug.

### Harmful instructions run longer than harmless ones even after filtering

`unverified` in magnitude, structural in direction. `R_harm` needs the same
**length-only baseline** the role probe needs: fit a classifier on length alone and
require the direction to beat it. Length-*matching* is the wrong remedy here — it
would discard data and distort the source mix.

### Four sampling requirements

Each was found by running the sampler and each is a build requirement, not a
finding:

1. **Stratify the train/test split by (label, source), not label alone.** The
   sources differ in character, so a label-only split lets source proportions drift
   between the sides; held-out separation would then partly measure distribution
   shift rather than generalisation. `source` is also the Level-1 style proxy for
   the metadata-vs-style test, so the drift would contaminate that too.
2. **Sample round-robin across sources, not proportionally.** Pooling and shuffling
   is proportional-to-size, which nearly eliminates the smaller pools — including
   XSTest, the benign-but-sensitive negative that is the one thing preventing
   `R_harm` collapsing into a "sensitive topic" detector.
3. **Enforce attack-pool disjointness by construction**, not by checking after the
   fact. StrongREJECT overlaps AdvBench. Filter the attack pool against the
   fitting pool on exact match plus token-Jaccard, before use.
4. **Record `source_provenance`** — dataset id, config, split, filter and available
   count for every pool. The Sorry-Bench `base` filter and the XSTest
   `label`-vs-`type` decision are invisible in the resulting text and would
   otherwise be unrecoverable.

---

## Cluster

`cluster`.

- **Everything runs through Slurm, including smoke tests and CPU-only analysis.**
  A `FAST_DEV` run of a 0.5B model was `SIGKILL`ed on the login node seconds after
  weight loading with 119 GB free — a node policy kill, not OOM. Loading a few GB
  of cached activations is enough to trigger it too.
- Submission: `sbatch slurm/scripts/run_gpu.sh experiments/<x>.py` (GPU) or
  `run_cpu.sh` (corpus and analysis stages). `--export=ALL,MODELS=a,b` does NOT
  work — sbatch splits `--export` on commas; submit one job per model.
- Python: `/home/b6aj/jtelintelo.b6aj/miniforge3/envs/venv_causal_safety/bin/python`

---

## Determinism and reproducibility

`cluster` + measured by a full independent re-run (`results` vs `results_verify`,
2026-09-09).

**The corpus is byte-reproducible.** Two independent builds produced identical
`instructions.jsonl`, `attack_intents.jsonl` and `transfer_corpus.jsonl`,
including the streamed C4 draw.

**Activation capture and the difference-of-means estimator are exactly
deterministic.** Every label-independent concept (`R_harm`, `R_harm_user`,
`R_harm_at_post`) reproduced with max |ΔAUC| = **0.000000** across every layer on
both models.

**Greedy generation is NOT bit-reproducible across batch sizes.** Padding changes
bf16 numerics, which flips a handful of borderline greedy tokens. A run at batch 64
and one at batch 32 produced slightly different refusal labels, and therefore
slightly different control directions (≤0.011 AUC at the reported layer).

**Requirement:** `batch_size` is part of a run's identity. It is recorded in the
manifest, and exact reproduction requires matching it. Label-dependent quantities
carry run-to-run variation on top of their bootstrap CI, and that CI — being
conditional on the labels — does not cover it.

**One statistic is fragile.** The E1.5 onset depth is unstable where the
AUC-vs-depth curve is flat near its peak and the concept comes from a thin
label-dependent cell: `R_control_harmless` on Qwen3.5-9B moved 0.355 → 0.774 at the
90% threshold between runs, while every other concept on both models moved 0.000.
The 80% threshold moved only 0.064. Onset therefore needs a bootstrap CI before any
specific depth is claimed for such a concept.

## Traps that silently produce wrong numbers

Engineering facts. Each was hit at least once.

- **Register the steering hook BEFORE the capture hook.** Hooks fire in
  registration order; capture-first reads the pre-steering value and every delta is
  exactly `0.0`.
- **`torch.linalg.eigvalsh` crashes on bf16** — `"linalg_eigh_cpu" not implemented
  for 'BFloat16'`. Cast to float32.
- **Left padding is required** for `t_post-inst`; `t_inst` still needs per-example
  indices regardless.
- **`apply_chat_template(..., return_tensors="pt")` returns a `BatchEncoding`**, not
  a tensor. Guard with `if not torch.is_tensor(x): x = x["input_ids"]`.
- **Direction signs are asserted, never inferred.** A wrong sign produces "no
  effect", which is unfalsifiable rather than false. Each direction records which
  class is positive, and the assertion is checked at write time.
- **A judge scores incoherence as non-refusal.** A model broken by an over-large
  intervention will show high ASR even under a random direction. Utility gating is
  what separates an attack from a broken model, and no ASR is reportable without a
  utility number at the same intervention.
- **Residual norm grows with depth**, so a fixed absolute perturbation confounds
  depth with perturbation size. This is why the steering coefficient is denominated
  in class-mean separations rather than absolute norm — see the parameter register
  in `EXPERIMENTS.md`.
- **Thread caps must be set by the entry point, literally, above every project
  import.** Exposing a `cap_cpu_threads()` helper from a module that imports torch
  is self-defeating: the import initialises torch first.
- **Llama-Guard-3-8B is a second 8B model.** Label with the target model, release
  it, then run the cross-check — or place them on separate devices. Do not assume
  both fit alongside a 9B target.

---

## External results we rely on

`published` — from the cited papers, not from our runs. These are inputs to the
design, and they are the only numbers in this file.

- **Zhao et al. (2507.11878).** Harmfulness and refusal are encoded separately, as
  distinct directions. Steering harmfulness makes harmless instructions read as
  harmful. Steering refusal elicits refusal *without reversing the harmfulness
  judgment* — an **asymmetry**, not a symmetric dissociation. Certain jailbreaks
  reduce refusal signals without reversing the model's internal belief of
  harmfulness.
- **Arditi et al. (2406.11717).** Refusal is mediated by a single direction found by
  difference-of-means, added or ablated across all token positions at a validation-
  selected layer, with no coefficient beyond the vector's own magnitude. Selection
  by bypass/induce/KL scores. Behavioural readout by refusal-prefix substring match.
- **Role confusion (2603.12277).** LLMs internally represent the apparent role of an
  instruction; **style dominates tags**; role confusion predicts attack success
  before generation, and destyling substantially reduces it.
- **NeuroStrike / GateBreaker.** Alignment can depend on sparse sets of neurons, and
  in MoE models on a limited number of experts, neurons and routing decisions.
  NeuroStrike's reported effectiveness differs sharply across the checkpoints on
  our roster, which is one reason the roster is what it is.

**Each of these is re-read from the paper when the experiment that depends on it is
designed.** Restatements here are orientation, not a substitute.
