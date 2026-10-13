# Measuring attribute interference in text-to-image prompt following

This repository implements the data preparation, frozen-model generation, evaluation and descriptive analysis pipeline for project P40. The first submission target is **SD 1.5 on all 576 main prompts and four seeds: 2,304 images**. SDXL uses the same prompts and seeds for a later comparison. No models are trained or fine-tuned.

**Validation status:** CPU software tests and data preparation run locally. The pinned inference stack passed the SD 1.5 smoke test on a Colab T4 GPU, and the full SD 1.5 baseline has been generated and evaluated (see [Executed SD 1.5 baseline run](#executed-sd-15-baseline-run)). Those evaluation results are **preliminary**: they use an 8-bit quantized Qwen evaluator and no reviewed evaluator freeze. Synthetic test fixtures are not experiment results.

## Dataset and experimental boundaries

| Configuration | Purpose | Prompts | Seeds | Images |
|---|---|---:|---|---:|
| `smoke.yaml` | SD 1.5 technical integration test | 8 | 0 | 8 |
| `smoke_sdxl.yaml` | SDXL technical integration test | 8 | 0 | 8 |
| `pilot_sdxl.yaml` | SDXL development and evaluator review | 80 | 0, 1 | 160 |
| `main_sd15.yaml` | First baseline | 576 | 0, 1, 2, 3 | 2,304 |
| `main_sdxl.yaml` | Main comparison | 576 | 0, 1, 2, 3 | 2,304 |

The main vocabulary contains 16 everyday objects and four animals from the proposal. Binary factors are color (C), count (N), position (P), material (M), and a second colored object (O2). Object identity is always requested. Animals have no material factor: `16 × 32 + 4 × 16 = 576` prompts. The development objects are mug, basket and rabbit. The technical smoke object is teapot, disjoint from both sets.

A per-object RNG derived from assignment seed 42 assigns one fixed value per factor. This mapping does not depend on dataset ordering. Inspect `prompts.csv` for plausibility before generating. Human changes to vocabulary/assignment require a new dataset/run; never silently edit an existing prepared CSV. Singular wording is implicit when N is absent and is not scored as a count constraint. Material is blank for animals. Second-object color differs from the assigned main-object color, including when C is inactive.

Directed minimal pairs add exactly one factor, retaining all other assigned values. Within a generator, pairs use the same seed and inference settings. Seeds do not imply equivalent noise across SD 1.5 and SDXL, which use different resolutions. This resolution difference is part of the model comparison and must be disclosed.

## Installation

Python 3.10+ is required. Data preparation and tests do not require a GPU or model downloads:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

On a restricted macOS/Python 3.14 setup, hidden editable `.pth` files may be ignored. If the installed command cannot import the package, use a regular `pip install .` installation (reinstall after source edits), or run with `PYTHONPATH=src`.

For CUDA inference, use `notebooks/sd15_baseline_generation_evaluation.ipynb` in Colab (see below), or install on a GPU server:

```bash
pip install -r requirements-colab.txt
pip install --no-deps -e .
```

Restart a notebook runtime after installing/replacing Torch or Transformers. The notebook runs CLI subprocesses so models do not accumulate in notebook memory. Checkpoint access is validated by downloading model metadata before running. Use Hugging Face login interactively if access requires authentication; do not store tokens in this repository.

## Model configuration

- **SD 1.5:** `stable-diffusion-v1-5/stable-diffusion-v1-5`, 512 × 512, FP16.
- **SDXL:** `stabilityai/stable-diffusion-xl-base-1.0`, 1024 × 1024, FP16, no refiner.
- Both: 30 inference steps, guidance scale 7.5, batch size 1, checkpoint scheduler, no added negative prompt, CPU model offload and VAE slicing/tiling.
- **Qwen:** `Qwen/Qwen2.5-VL-7B-Instruct`, FP16 **without quantization**, deterministic decoding, SDPA, 16 output tokens per question, 256–1,280 visual tokens. CPU offload is explicit in configuration; the CPU memory cap defaults to 8 GiB.
- **OWLv2:** `google/owlv2-base-patch16-ensemble`, FP32, confidence threshold 0.2, class-specific NMS IoU 0.5. These are initial development choices, not calibrated quality guarantees.

The original proposal names `runwayml/stable-diffusion-v1-5`. Its [model-card mirror](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) identifies that repository as deprecated. The accessible mirror is used explicitly, not silently substituted at runtime. The requested model revision is resolved to an immutable Hub commit on first use and reused on resume. Main evaluators inherit the exact frozen pilot revisions.

Every run saves the model commits, scheduler config, package versions, CUDA/GPU details, source hash, input hashes and complete configuration. Runs are single-writer: do not use two notebook sessions on one output directory.

## Run in order

All commands accept `--config` and `--output`. Keep the output directory unchanged when resuming. In Colab, place it under mounted Google Drive.

### 1. Technical smoke tests

```bash
attribute-interference prepare --config configs/smoke.yaml --output outputs/smoke_sd15
attribute-interference generate --config configs/smoke.yaml --output outputs/smoke_sd15
attribute-interference evaluate --config configs/smoke.yaml --output outputs/smoke_sd15
attribute-interference analyze --config configs/smoke.yaml --output outputs/smoke_sd15
```

Repeat with `configs/smoke_sdxl.yaml` and `outputs/smoke_sdxl`. Inspect image quality, raw answers, boxes, metrics and resume behavior. An integration smoke pass does not establish evaluator accuracy. `--limit 2` caps pending inference jobs for a diagnostic invocation; it never changes the prepared dataset or its four-seed requirement.

### 2. Development pilot and manual review

```bash
attribute-interference prepare --config configs/pilot_sdxl.yaml --output outputs/pilot_sdxl
attribute-interference generate --config configs/pilot_sdxl.yaml --output outputs/pilot_sdxl
attribute-interference evaluate --config configs/pilot_sdxl.yaml --output outputs/pilot_sdxl
attribute-interference analyze --config configs/pilot_sdxl.yaml --output outputs/pilot_sdxl
attribute-interference export-annotations --config configs/pilot_sdxl.yaml --output outputs/pilot_sdxl
```

Open `review/gallery.html` and fill `review/annotations.csv`. Automatic labels are deliberately not shown in the review gallery. Fill `reviewer`, `human_object`, and every active `human_C/N/P/M/O2` with `true`, `false` or `unknown`; leave inactive factors blank. If the main object is absent, every active factor is false. Review sampling covers object, factor and complexity strata, including zero additional constraints.

Check `analysis/singletons.csv` against the proposal's 60% pilot criterion; these rows evaluate prompts with exactly one active factor. If a factor is below 60%, discuss the proposal's simplification/stronger-model fallback before freezing; do not tune on main results. Review false positives, counting errors and ambiguous parses. Modify evaluator configuration/questions in a **new development run**, rerun and review before freezing. The code does not claim that threshold 0.2 is optimal.

Once satisfied with the development decisions:

```bash
attribute-interference freeze-evaluator --config configs/pilot_sdxl.yaml --output outputs/pilot_sdxl
```

The freeze records settings, source, exact evaluator model commits, annotation hash and per-factor development agreement counts. It requires exactly the selected 24 reviewed images and at least one known human/automatic comparison per factor. This is development calibration, **not** the final 150-image held-out evaluator test or the proposal's full 200-image annotation exercise. Control-question yes-bias analysis and final inferential statistics remain follow-up work.

### 3. Full first baseline

```bash
attribute-interference prepare --config configs/main_sd15.yaml --output outputs/main_sd15
attribute-interference generate --config configs/main_sd15.yaml --output outputs/main_sd15
attribute-interference evaluate --config configs/main_sd15.yaml --output outputs/main_sd15 --frozen outputs/pilot_sdxl/evaluator_freeze.json
attribute-interference analyze --config configs/main_sd15.yaml --output outputs/main_sd15
```

Main evaluation requires the matching frozen pilot settings. Generation is complete only when all **2,304 images** exist. Scoring completion is reported separately; unknown answers remain visible and must be discussed rather than hidden. Repeat the same commands with `main_sdxl.yaml` and its own directory for the later model comparison.

## Executed SD 1.5 baseline run

The full SD 1.5 baseline was run in Google Colab with `notebooks/sd15_baseline_generation_evaluation.ipynb`. The notebook contains every stage in order: repository setup, dependencies, GPU check, Google Drive mount, smoke test, generation, generation summary, OWLv2 evaluation, Qwen evaluation, progress check and analysis. All stages are resumable, so rerunning the notebook on an existing output directory skips completed work.

**Environment.** Google Colab, NVIDIA Tesla T4 (14.6 GiB), Python 3.13, pinned stack from `requirements-colab.txt` plus `bitsandbytes==0.47.0`. Source commit `0de4f3a` (the `src/` and `configs/` directories are unchanged since). Outputs were written to Google Drive under `attribute_interference/v1/`.

### Stages

1. **Smoke test.** `configs/smoke.yaml`, 8 SD 1.5 images, generation only. Passed.
2. **Generation.** `configs/main_sd15.yaml`, 576 prompts × seeds 0–3. 2,301 of 2,304 images were generated; the mean generation time was 6.6 s per image. Three images were rejected by the SD 1.5 safety checker (false NSFW flags, all seed 1): `6e6d5e277f4477f4_s1`, `685d50e47cd34ca8_s1`, `d312e804d7899066_s1`. Seeds are fixed, so regenerating them reproduces the same output; they are kept as recorded generation errors rather than hidden.
3. **OWLv2 evaluation.** Unchanged FP32 detector. 1,727 of 1,727 applicable images (prompts with a count or position factor) evaluated, 0 errors, roughly one hour on the T4.
4. **Qwen2.5-VL-7B evaluation.** 2,301 of 2,301 images evaluated, 0 errors, 0 unparseable answers, roughly 2.5 hours on the T4 (about 15 images per minute).
5. **Analysis.** Metrics, figures and failure examples computed from the combined OWLv2 and Qwen labels.

### Deviations from the protocol

These results are preliminary and must be reported as such.

- **No evaluator freeze.** Main evaluation normally requires `evaluator_freeze.json` from the reviewed SDXL development pilot (section 2 of [Run in order](#run-in-order)). The pilot and its human review have not been done yet, so the notebook skips the freeze check without modifying any file in `src/`; the source hash therefore still matches the generation run. The run directory is marked with `PRELIMINARY_UNFROZEN.json`. Evaluator accuracy has not been validated against human labels.
- **8-bit quantized Qwen.** In FP16 with CPU offload, Qwen does not fit on a T4 and is too slow for 2,301 images in one session. It was loaded with bitsandbytes 8-bit quantization (vision encoder and output head kept in FP16), which fits entirely on the GPU. A spot check of raw answers showed well-formed Yes/No responses consistent with the images. The run directory is marked with `QWEN_INT8_NOTE.json`.
- **Separate storage.** Quantized answers are stored in `qwen_int8_records/`, not `qwen_records/`, so they never mix with a future FP16 evaluation. Final results are copied to `analysis_qwen_int8/` with a `NOTE.txt`. The `analysis/` directory and `labels.csv` in the run root are regenerated by any later `analyze` call and should not be cited.

### Outputs

| Location (Google Drive, `attribute_interference/v1/main_sd15/`) | Contents |
|---|---|
| `images/`, `generation_records/` | 2,301 PNG images and per-image generation records |
| `owl_records/` | OWLv2 boxes and scores |
| `qwen_int8_records/` | Raw and parsed 8-bit Qwen answers |
| `analysis_qwen_int8/` | `summary.json`, `attributes.csv`, `by_complexity.csv`, `interference.csv`, `singletons.csv`, `labels.csv`, figures, `failure_cases.html`, `NOTE.txt` |

The image collection stays on Drive and is not committed to Git.

### Reproducing in Colab

1. Open the notebook in Colab, select a T4 GPU runtime and run cells 1 and 2.
2. Restart the session, then run cell 1 and all following cells.
3. Mount the Google Drive account that holds `attribute_interference/v1/` (or a shortcut to it with editor access).

With existing outputs, generation and evaluation report that all jobs are complete and only the analysis is recomputed.

### Next steps

To replace the preliminary evaluation: run the SDXL development pilot, review the 24 exported images, run `freeze-evaluator`, then run the standard FP16 evaluation of `main_sd15` with `--frozen`. Images do not need to be regenerated. Evaluator settings in `pilot_sdxl.yaml` and `main_sd15.yaml` are identical, so a freeze from the pilot applies directly. The full SDXL comparison remains future work.

## Resuming and memory

Generation and evaluation save an atomic JSON record per image to persistent storage. The generation manifest is a derived JSONL export, refreshed when the generator returns; after a hard notebook termination the per-image records remain authoritative. Rerunning generation reconstructs the manifest. A saved PNG without its completed record is regenerated. Missing/corrupt/hash-mismatched images are regenerated; evaluation caches are keyed by image hash.

Qwen and OWLv2 are loaded sequentially. Run `evaluate --backend qwen` and `evaluate --backend owl` separately if desired. CPU offload is slower and still needs sufficient RAM. If Qwen cannot fit, retain the saved artifacts and use a larger GPU. No automatic quantization or model substitution occurs. Changing inference settings, source files, Python or package versions requires a separate output directory. A replacement GPU may continue the run with unchanged software/settings: each image records its hardware environment ID. Such hardware changes can affect reproducibility and must be disclosed; the summary lists generation environment IDs.

Colab [does not guarantee a GPU type or uninterrupted resources](https://research.google.com/colaboratory/faq.html). Do not assume the full experiment finishes in one session. Record timings from the actual run; no runtime estimate is presented as measured evidence.

## Data loading, preprocessing and labeling

`prepare` writes `prompts.csv`, `pairs.csv`, `jobs.csv` and immutable `run.json`. `generate` reads these records, generates images and logs settings. `evaluate` loads RGB images and applies the model's official processor. Qwen receives one templated question per requested factor; OWLv2 receives the main-object query. Bounding boxes are clipped, confidence-filtered and de-duplicated with NMS.

Only an unambiguous Yes/No answer is parsed as Boolean. Count answers accept integer digits or words zero–ten; other outputs remain unknown. Raw questions/answers and raw/filtered detector boxes are retained for audit. Qwen's N/P scores are diagnostic; the combined evaluator uses OWLv2 for N/P. Object presence comes from Qwen. An unknown presence answer leaves all combined labels unknown. A negative presence answer fails every requested constraint. This evaluator choice must be checked in the pilot.

Position is correct only if **all detected main-object centers** lie strictly inside the requested half; a center exactly at the midpoint fails both sides. Zero boxes fails position. Count is the number of remaining main-object boxes. Color/material questions require every visible instance to satisfy the attribute. O2 requires a matching colored second object.

## Output and metric definitions

| Output | Contents |
|---|---|
| `prompts.csv`, `pairs.csv`, `jobs.csv` | Reproducible inputs and 2,304 main generation jobs |
| `generation_records/`, `generations.jsonl` | Per-image generation status, hashes, times and provenance |
| `qwen_records/`, `owl_records/` | Raw evaluator responses, boxes, errors and parsed values |
| `labels.csv` | Combined and individual evaluator labels; blank means unknown/inactive |
| `review/` | Blind gallery, human annotation sheet and selected-image hashes |
| `analysis/summary.json` | Completion flags, error counts and all numeric summaries |
| `analysis/*.csv`, `analysis/*.png` | Attribute/complexity results and interference matrix |
| `analysis/failure_cases.html` | Up to six actual failures with prompts, for visual verification |

- **Attribute accuracy:** correct / known applicable labels. Coverage is known / all requested jobs, so missing generations and unknown answers remain visible.
- **Strict accuracy:** all requested factors **and object presence** true, among completely evaluated images. Its coverage denominator includes all scheduled images.
- **Complexity:** number of active C/N/P/M/O2 factors; the object is always present. Zero-factor prompts evaluate object presence only.
- **Interference Δ(A,B):** mean `A_after − A_before` over same-seed pairs adding B, where A is already requested, object presence is true in both images and A labels are known. Negative values indicate interference. Counts expose exclusions.
- **Flip rate:** correct→incorrect transitions / all eligible pairs. `conditional_loss_rate` also reports transitions / initially correct eligible pairs; these denominators are intentionally distinct.
- **Timing:** mean successful generation wall time, including CPU offload and synchronized GPU computation but excluding model loading and PNG saving.

Unknown results are never replaced by zeros. Missing metrics use JSON null/blank CSV cells. Plots do not fabricate observations. Failure examples are automatically identified and need human visual confirmation before a report. The summary marks incomplete runs explicitly.

## Submission and next stage

Before submission: finish the full SD 1.5 run, inspect coverage/errors and failure examples, and attach a real team Git repository link to the report. No remote repository is configured by this implementation. Keep code/configs/tests and compact reviewed result tables in Git; copy chosen tables and plots to a versioned results directory after a real run. Do not commit checkpoints, tokens or the full image collection.

The report should identify the executed checkpoint revisions, frozen evaluation settings, actual GPU/runtime, observed problems and real metrics. Describe training setup as “No training or fine-tuning; frozen pretrained models used zero-shot.” Future work includes the full SDXL comparison, held-out human agreement/Cohen's kappa, control questions, corrected McNemar tests and mixed-effects analysis.

Official implementation references: [Diffusers CPU offload](https://huggingface.co/docs/diffusers/main/en/optimization/memory), [Qwen2.5-VL model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), [OWLv2 processor and inference](https://huggingface.co/docs/transformers/model_doc/owlv2).
