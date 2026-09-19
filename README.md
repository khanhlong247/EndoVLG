# EndoVLG: Region-to-Region Vision-Language Modeling for Endoscopy Report Generation

EndoVLG is a multi-stream vision-language (VLM) framework for automated diagnostic report generation from gastrointestinal (GI) endoscopy images. Instead of extracting a single global feature from the whole image — which can dilute small pathological cues such as polyps — EndoVLG decomposes each endoscopic image into three parallel feature streams (Global Context, Local Texture, Geometry) under a **Fine-Grained Region-to-Region** mechanism, then fuses them with **Llama-2-7b-chat** (LoRA fine-tuned via [Unsloth](https://github.com/unslothai/unsloth)) to generate natural-language reports.


**Keywords:** Endoscopy Report Generation, Vision-Language Model, Region-level Feature Extraction, Multimodal Fusion, LoRA, Kvasir-VQA-x1.

## Table of Contents

- [Highlights](#-highlights)
- [EndoVLG Model Architecture](#-endovlg-model-architecture)
- [Experimental Results](#-experimental-results)
- [Installation](#️-installation)
- [Directory Structure](#-directory-structure)
- [How to Run](#-how-to-run)
  - [Part 1 — Building the Kvasir-VQA-x1 Dataset](#part-1--building-the-kvasir-vqa-x1-dataset)
  - [Part 2 — Training](#part-2--training)
- [Evaluation](#-evaluation)
- [Notes](#-notes)

## Highlights

- **Multi-stream feature-extraction architecture** — separates Global Context, Local Texture, and Geometry via Hadamard-product masking between the raw image and the segmentation mask, letting the model concentrate computation on the lesion region instead of treating the whole image as one undifferentiated tensor.
- **Progressive three-stage training strategy** — a balanced malignant/benign Custom Score guides the diagnostic-alignment stage, and a four-phase progressive-unfreezing schedule stabilizes LoRA fine-tuning of the LLM in the final stage.
- **Large-scale Kvasir-VQA-x1 supervision** for region-grounded report generation, exploiting its built-in per-sample complexity score for curriculum-style sampling.
- **Outperforms the strongest published baseline (Qwen2.5-VL-ft) on 5/8 NLG metrics** on the Kvasir-VQA-x1 test set, with the highest BERT-F1 among compared methods.

## EndoVLG Model Architecture

`EndoRegHD` ([`src/models/endoreghd.py`](src/models/endoreghd.py)) processes each endoscopic image $I \in \mathbb{R}^{H \times W \times 3}$ together with its segmentation mask $M \in \{0,1\}^{H \times W}$ through three parallel vision streams, fused with the LLM to generate an answer:

1. **Global Context Branch** — the full image → a UNet ([`src/models/unet.py`](src/models/unet.py)) encoder only, 1024-d bottleneck, global-average-pooled → `global_adapter` (`Linear → LayerNorm → GELU → Linear`) → one global token: $f_g = E_\theta(I)$.
2. **Local Texture Branch** — up to `max_regions=6` cropped ROIs from $I \odot M$ (Hadamard product — background suppressed to zero, lesion surface texture preserved), through the **same** UNet encoder (shared weights with the Global branch) → `texture_adapter`: $f_t = E_\theta(\text{Crop}(I \odot M))$.
3. **Geometry Branch** — the corresponding mask crop, through `mask_encoder` (`GeometryMaskEncoder`, [`src/models/mask_encoder.py`](src/models/mask_encoder.py) — an ImageNet-pretrained **ResNet-34** backbone with frozen BatchNorm and a `SpatialAttentionPooling` head instead of plain average pooling) → `geometry_projector`: $f_{geo} = \text{Encoder}_{\text{Res34}}(M)$.

`texture_adapter` and `geometry_projector` outputs per region are concatenated into $f_{local} = [f_t; f_{geo}]$ and passed through `fusion_layer` (Linear + LayerNorm) to produce one embedding per region. All tokens (1 global + up to 6 region tokens) are interleaved into the prompt as embeddings (`prepare_multimodal_prompt`): `BOS + "Question: {question} " + "\nContext: " + [global] + " Details: " + [regions] + "\nAnswer: "`, fed into **Llama-2-7b-chat-hf** (loaded 4-bit via Unsloth, LoRA fine-tuned with `r=32` on `q/v/k/o/gate/up/down_proj`) to generate the answer/report.

Per-sample input data (`EndoDataset`, [`src/data/dataset.py`](src/data/dataset.py)) consists of the raw image, the binary mask (used to auto-derive lesion bounding boxes when absent from the JSON), the question, and the reference answer.

Training follows **three progressive stages**:

| Stage | Objective | Data | Script (new location) |
|:-:|---|---|---|
| 1 — Vision Encoder Pre-training | Train UNet on segmentation + coarse classification (DSC, IoU) | curated polyp dataset | `training/train_stage1.py` |
| 2 — Vision-Diagnostic Alignment | Train ResNet-34 + UNet to recognize malignant/benign, optimizing the **Custom Score** $S = (ACC_{mal}+ACC_{ben})\times(1-\lvert ACC_{mal}-ACC_{ben}\rvert)$ | curated polyp dataset | `training/train_stage2.py` |
| 3 — Semantic Alignment & Report Generation | LoRA fine-tune Llama-2-7b under a 4-phase progressive-unfreezing schedule (Texture/Geometry adapter → +Global/Fusion → +local LoRA → full LoRA + LR reset) | Kvasir-VQA-x1 (complexity-score curriculum) | `training/train.py` |

## Experimental Results

Report-generation performance of EndoVLG (Stage 3 checkpoint, epoch 24) measured on the Kvasir-VQA-x1 test set (4,058 images, 15,955 question-answer pairs), against two published baselines on the same benchmark — **FL2_VQA_MT** (multi-task fine-tuned Florence-2) and **Qwen2.5-VL-ft** (denoted Q-VL-ft-3333):

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR | CHRF++ | BLEU-4 | BERT-F1 | BLEURT |
|:-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| FL2_VQA_MT | 70.76 | 52.34 | 68.02 | 68.04 | -- | 47.26 | 94.79 | -- |
| Q-VL-ft-3333 | 71.60 | 53.40 | 69.00 | **68.90** | **67.91** | 47.80 | 95.40 | **0.404** |
| **EndoVLG (Ours)** | **73.15** | **54.46** | **70.07** | 68.78 | 67.33 | **48.22** | **95.78** | 0.39 |

EndoVLG surpasses the strongest baseline (Q-VL-ft-3333) on **5/8 metrics** (ROUGE-1/2/L, BLEU-4, BERT-F1), trailing only slightly (under 0.6 points) on METEOR, CHRF++, and BLEURT, and outperforms every published metric of FL2_VQA_MT.

## Installation

- **Python 3.10**, package management via [`uv`](https://github.com/astral-sh/uv) ([`pyproject.toml`](pyproject.toml) + [`uv.lock`](uv.lock)).
- An NVIDIA GPU with enough VRAM to load Llama-2-7b-chat 4-bit + LoRA (≥16GB recommended).

```bash
git clone <repo-url>
cd EndoVLG_clean

uv sync
```

Key dependencies: `torch`, `transformers`, `unsloth` (4-bit LLM + LoRA loading), `peft`, `evaluate`, `bert-score`, `rouge-score`, `sacrebleu`, `opencv-python`, `scikit-learn`, `timm`, `wandb` (training logs). `training/run_bleurt.py` additionally needs the `bleurt` package (not in `pyproject.toml`):

```bash
uv pip install git+https://github.com/google-research/bleurt.git
```

## Directory Structure

```
EndoVLG_clean/
├── data/                     # Kvasir-VQA-x1 build pipeline + debug-dataset splitters for Stage 1/2
├── training/                 # Train/inference entrypoints: train_stage1.py, train_stage2.py, train.py, run_eval.py, run_bleurt.py
├── scripts/                  # .sh launchers for every script in data/ and training/
├── src/                      # Core library: model (endoreghd.py, unet.py, mask_encoder.py), dataset (dataset.py), utils
└── pyproject.toml, uv.lock, requirements.txt, README.md
```

Every script under `data/` and `training/` uses **paths relative to the project root** (not to the script's own location) for datasets/checkpoints. Always run them via the matching launcher in `scripts/` (which `cd`s to the project root and sets `PYTHONPATH` automatically), or make sure your current working directory is the project root when calling `python data/...` / `python training/...` directly.

## How to Run

The full pipeline has **2 sequential parts**: (1) building the Kvasir-VQA-x1 dataset (including training the UNet that generates masks — a prerequisite for building the dataset), then (2) training EndoVLG through Stage 2 and Stage 3. The tables below list the exact run order.

### Part 1 — Building the Kvasir-VQA-x1 Dataset

The raw dataset comes from HuggingFace [`SimulaMet/Kvasir-VQA-x1`](https://huggingface.co/datasets/SimulaMet/Kvasir-VQA-x1). Since lesion masks are not included in the raw dataset, the pipeline must **auto-generate masks with a fine-tuned UNet segmentation model** before the dataset build is complete — so the UNet training step (Stage 1) must run first, before any Kvasir-VQA-x1 download/processing step.

#### Step 0 — Train the mask-generating UNet (prerequisite)

Requires a pretrained UNet checkpoint (`checkpoints/kvasir_seg/UNet_IoUBCELoss_augmented.pth`) and a malignant/benign-labeled polyp dataset (BKAI-IGH — prepared/collected separately, not produced by any script in this repo). Then run:

```bash
bash scripts/run_stage1.sh          # python training/train_stage1.py
```

→ Output: `checkpoints/stage1_seg_unet_full/unet_kvasir_best.pth` (best validation Dice checkpoint).

#### Steps 1-5 — Build Kvasir-VQA-x1 (run in this exact order)

Each step below has a parallel `_test` version that builds the official test split (used by `run_eval.py`) — same logic, only the HF `dataset['test']` split is used instead of `dataset['train']`.

| # | Command | Input | Output |
|:-:|---|---|---|
| 1 | `bash scripts/kvasir_download.sh` <br> `bash scripts/kvasir_download_test.sh` | `SimulaMet/Kvasir-VQA-x1` (HF Hub) | `kvasir_vqa_ready/images/`, `kvasir_vqa_test_ready/images/` |
| 2 | `bash scripts/kvasir_create_metadata.sh` <br> `bash scripts/kvasir_create_metadata_test.sh` | HF dataset (questions/answers) | `kvasir_vqa_ready/dataset.json`, `kvasir_vqa_test_ready/dataset.json` |
| 3 | `bash scripts/kvasir_generate_masks.sh` <br> `bash scripts/kvasir_generate_masks_test.sh` | images + UNet checkpoint (Step 0) | `kvasir_vqa/{images,masks,dataset.json}`, `kvasir_vqa_test/{images,masks,dataset.json}` |
| 4 | `bash scripts/enrich_vqa_metadata.sh` <br> `bash scripts/enrich_vqa_metadata_test.sh` | generated masks (K-Means lesion bounding-box extraction) | overwrites `regions` in `kvasir_vqa/dataset.json`, `kvasir_vqa_test/dataset.json` |
| 5 | `bash scripts/split_kvasir_vqa.sh` | `kvasir_vqa/` (train split only; the test split uses `kvasir_vqa_test/` directly) | `kvasir_vqa_train/{train,val}/{images,masks,dataset.json}` (90/10 split by image, seed 42) |

`kvasir_vqa_train/{train,val}` and `kvasir_vqa_test/` are the exact data used directly by Stage 3 (Part 2 below) and `run_eval.py`.

### Part 2 — Training

Once Part 1 is complete (UNet checkpoint from Step 0 + the Kvasir-VQA-x1 dataset), training EndoVLG consists of 2 main stages plus 1 feature-precompute step, run in this order:

#### Stage 2: Vision-Language Region Alignment

- **Objective:** train `mask_encoder` (ResNet-34), `geometry_projector`, `global_adapter`, `texture_adapter` to recognize malignant/benign lesions from visual features, optimizing the Custom Score $S$ (the LLM is **not** involved in this stage).
- **Load weights:** `volume_encoder` initialized from the Step 0 UNet checkpoint.
- **Data:** BKAI-IGH (same set as Stage 1, not `kvasir_vqa_train/`).

```bash
bash scripts/run_stage2.sh          # python training/train_stage2.py
```

→ Output: `checkpoints/stage2_vision_alignment/vision_alignment_best.pth`.

#### Precompute vision cache (optional, speeds up Stage 3)

Precomputes frozen backbone features for every unique image in `kvasir_vqa_train/train`, avoiding recomputation of UNet/ResNet-34 every epoch:

```bash
bash scripts/precompute_vision_cache.sh   # python data/precompute_vision_cache.py
```

→ Output: `checkpoints/stage3_vision_cache_K1.pt`.

#### Stage 3: SFT — Fine-tune the LLM to Generate Answers (VQA)

- **Objective:** teach Llama-2-7b-chat to generate VQA answers grounded in the vision features aligned in Stage 2.
- **Load weights:** all non-LLM components (adapters/encoders) from the Stage 2 checkpoint; the LLM is loaded 4-bit with a fresh LoRA (`r=32`).
- **Progressive unfreezing curriculum** (configured via `STAGE31_EPOCH`, `STAGE32_EPOCH`, `STAGE33_EPOCH` in [`training/train.py`](training/train.py)): texture/geometry adapters → + global adapter/fusion layer → + local LoRA (`q/v/k/o_proj`) → + full LoRA (`gate/up/down_proj`, LR reset).
- **Data:** `kvasir_vqa_train/{train,val}` (built in Part 1).
- Can resume from a prior Stage 3 checkpoint via `RESUME_CKPT_PATH` in `training/train.py`.

```bash
bash scripts/train.sh                # python -u training/train.py, logs to logs/train_run_<timestamp>.log
```

→ Output: `checkpoints/<OUTPUT_DIR>/EndoRegHD_Epoch<N>.pth` (saved whenever validation loss improves)

## Evaluation

Evaluated on the official `kvasir_vqa_test/` test set (built in Part 1), generating answers via greedy decoding and computing standard NLG metrics:

```bash
bash scripts/run_eval.sh             # python -u training/run_eval.py -- edit CHECKPOINT_PATH in the file to point at the desired Stage 3 checkpoint
```

→ Output: `results/<OUTPUT_DIR>/predictions_debug.csv` (per-sample Reference/Prediction) and `overall_metrics.csv` (BLEU-1/2/4, ROUGE-1/2/L with `use_stemmer=True`, METEOR, CHRF++, BERT-F1 — all computed on lowercased text to avoid unfairly penalizing case inconsistencies already present in the dataset itself).

`overall_metrics.csv` leaves the `BLEURT` column at `0.0` by default — computed separately (requires the `bleurt` package, see [Installation](#️-installation)):

```bash
python training/run_bleurt.py        # reads the existing predictions_debug.csv, does not re-run model inference
                                      # uses the bleurt-base-128 checkpoint (matches the original baseline's scoring)
```
