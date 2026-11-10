# Attribute interference in text-to-image prompts

Project P40. When one more constraint is added to a prompt, which of the constraints already in it stop being satisfied?

We compare SD 1.5 (baseline) and SDXL on 576 factorial prompts with 4 seeds each, 2,304 images per model. Prompts vary color, count, position, material and a second object. All models are pretrained and nothing is trained.

## Main results

| Metric | SD 1.5 | SDXL |
|---|---:|---:|
| Strict accuracy | 16.1% | 24.5% |
| Color | 48.4% | 65.4% |
| Material | 37.7% | 61.8% |
| Second object | 37.3% | 64.1% |
| Count | 30.8% | 33.8% |
| Position | 3.6% | 4.3% |
| Generation time on Tesla T4 | 6.6 s per image | 33.1 s per image |

SDXL is better overall. Neither model handles count or position well. Adding a second object lowers color accuracy by 27 points for SD 1.5 and 33 points for SDXL.

All labels come from Qwen2.5-VL in 8-bit. We first used OWLv2 for count and position, but replaced it after comparing both with 150 human-labeled SDXL images. OWLv2 results are kept as a comparison.

## What do you want to reproduce?

| Goal | What to do | Needs |
|---|---|---|
| The tables of the report | Steps 1, 2 and 5 | Any computer, about a minute after the download |
| Our exact images | Steps 1 and 2, then check them with `verify_images.py` | Any computer |
| The whole experiment from the start | Steps 1, 3, 4 and 5 | GPU, more than a day |

## 1. Install

You need Python 3.13. Steps 3 and 4 also need an NVIDIA GPU with at least 15 GB of memory and CUDA 12.8 drivers.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements-colab.txt
pip install -e . --no-deps
pip install "scikit-learn>=1.5,<2" "statsmodels>=0.14,<0.15"
pytest
```

## 2. Download our runs

Each Drive folder holds the images, raw model answers and labels of one run:

- [SD 1.5 run](https://drive.google.com/drive/folders/1M5MIvXCSDvjzcyqrCS1Tk4d-9o-fkZDL)
- [SDXL run](https://drive.google.com/drive/folders/1LnJclVA_sjJCcCs-FVpK7MialOcHSeaz)

Set their paths and check the images:

```bash
SD15=path/to/main_sd15
SDXL=path/to/main_sdxl

python scripts/verify_images.py --run "$SD15" --hashes results/image_hashes/main_sd15.csv
python scripts/verify_images.py --run "$SDXL" --hashes results/image_hashes/main_sdxl.csv
```

You should see 2,301 matched and 3 filtered images for SD 1.5, and 1,415 T4 plus 889 RTX 3090 matched images for SDXL. Then go to step 5.

## 3. Generate images

Prompts, seeds and model revisions are fixed in `configs/`. Each run has 2,304 jobs and takes about 4 hours for SD 1.5 and 21 hours for SDXL on a Tesla T4. An interrupted run continues from where it stopped when you repeat the command.

```bash
SD15=outputs/main_sd15
SDXL=outputs/main_sdxl

attribute-interference prepare --config configs/main_sd15.yaml --output "$SD15"
attribute-interference generate --config configs/main_sd15.yaml --output "$SD15"

attribute-interference prepare --config configs/main_sdxl.yaml --output "$SDXL"
attribute-interference generate --config configs/main_sdxl.yaml --output "$SDXL"
```

As an alternative, run `notebooks/sd15_baseline_generation_evaluation.ipynb` and `notebooks/sdxl_generation_evaluation.ipynb` in Colab.

## 4. Label images

This runs OWLv2 and 8-bit Qwen on every image, writes `labels.csv` and computes basic metrics. An interrupted run continues when you repeat the command.

```bash
PYTHONPATH=src python scripts/evaluate_labels.py --config configs/main_sd15.yaml --run "$SD15"
PYTHONPATH=src python scripts/evaluate_labels.py --config configs/main_sdxl.yaml --run "$SDXL"
```

Add `--limit 2` to try a few images first. The configs mention NF4 quantization, but this script always uses 8-bit, as our final runs did.

## 5. Compute the report tables

No GPU is needed. Human labels are already in `notebooks/review/`. They were made on our images, so with newly generated images the comparisons with humans do not apply.

```bash
python scripts/evaluator_agreement.py --annotations notebooks/review/annotations_test150.csv --run "$SDXL" \
  --first-annotator notebooks/review/annotations_anna.csv --second-annotator notebooks/review/annotations_saida.csv \
  --output results/evaluator_agreement
python scripts/evaluator_dependence.py --base "$SD15" --second "$SDXL" --output results/evaluator_dependence
python scripts/owl_threshold_sensitivity.py --base "$SD15" --second "$SDXL" --output results/owl_threshold_sensitivity
python scripts/interaction_analysis.py --base "$SD15" --second "$SDXL" --output results/interaction_analysis
python scripts/report_tables.py --base "$SD15" --second "$SDXL"
python scripts/select_failure_candidates.py --base "$SD15" --second "$SDXL" \
  --annotations results/evaluator_agreement/annotations_clean.csv --output results/failure_candidates
```

| Report table or section | File |
|---|---|
| Table 1, attribute and strict accuracy | `results/report_tables/accuracy.csv` |
| Table 2, strict accuracy by number of constraints | `results/report_tables/strict_by_constraints.csv` |
| Table 3 and Table 5, agreement with human labels | `results/evaluator_agreement/agreement.csv` |
| Agreement between human annotators | `results/evaluator_agreement/inter_annotator.csv` |
| Table 4, interference between constraints | `results/report_tables/interference.csv` |
| Regression on added constraints | `results/report_tables/regression_terms.csv` |
| Table 6, results under each evaluator | `results/evaluator_dependence/summary.csv` and `interference.csv` |
| Table 7, OWLv2 thresholds | `results/report_tables/owl_thresholds_against_humans.csv` and `results/owl_threshold_sensitivity/` |
| OWLv2 and Qwen counts against human counts | `results/report_tables/count_owl_against_qwen.csv` |
| Results on objects both models generated on the T4 | `results/report_tables/t4_objects_accuracy.csv` and `t4_objects_interference.csv` |
| Failure cases | `results/failure_candidates/` and `results/report_figures/failure_cases.png` |
| Hash of every image | `results/image_hashes/` |

Generation times in Table 8 are the `seconds` field of `generation_records/` in each run.

## Notebooks

These are the Colab notebooks of our original runs.

| Notebook | Purpose |
|---|---|
| `notebooks/sd15_baseline_generation_evaluation.ipynb` | SD 1.5 generation and evaluation |
| `notebooks/sdxl_generation_evaluation.ipynb` | SDXL pilot, generation and evaluation |
| `notebooks/sdxl_pilot_freeze_owl_calibration.ipynb` | Pilot human labels and OWLv2 calibration |
| `notebooks/qwen_yes_bias_control_questions.ipynb` | Qwen control questions on SD 1.5 |
| `notebooks/demo.ipynb` | Live demo: three prompts through the full pipeline on a Colab T4 |

## Team

Saida Musaeva, Alina Pestova, Anna Mingaleva, Maxim Iziumov
