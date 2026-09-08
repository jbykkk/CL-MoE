# UCIT benchmark adapter

This branch uses the task order published by Octopus:

1. ImageNet-R
2. ArxivQA
3. VizWiz
4. IconQA
5. CLEVR-Math
6. Flickr30k

Octopus instruction files use fields such as `text`, `answer`, and `image`.
CL-MoE's legacy LLaVA loader expects `image` plus a two-turn
`conversations` list. `prepare_ucit.py` validates and converts between these
formats without copying any image files.

## Paths

Local development defaults:

```text
CL-MoE:  /Users/lyk/Documents/Projects/CL-MoE
Octopus: /Users/lyk/Documents/Projects/Octopus
```

Server paths:

```text
CL-MoE:  /home/data1/lyk/CL-MoE
Octopus: /home/data1/lyk/Octopus
```

Converted JSON is written under `CL-MoE/data/UCIT/` and is ignored by Git. By
default, training files use exactly the sample counts and seeded random subsets
selected by Octopus's `path.json#N` syntax (`data_seed=42`). Test files always
retain all 3,000 examples. Image paths remain relative to the Octopus checkout,
so CL-MoE training uses the Octopus root as `--image_folder`; images are not
copied or symlinked into CL-MoE.

The matched training sizes are 9,600 examples for ImageNet-R, ArxivQA, VizWiz,
and IconQA; 16,000 for CLEVR-Math; and 3,200 for Flickr30k.

The independent full-data variant contains 23,998 ImageNet-R examples, 40,000
ArxivQA examples, 40,000 VizWiz examples, 29,859 IconQA examples, 40,000
CLEVR-Math examples, and 40,000 Flickr30k examples. Prepare it without
overwriting the matched subset:

```bash
python scripts/UCIT/prepare_ucit.py \
  --split train \
  --train-sampling full \
  --output-root data/UCIT_full \
  --octopus-root /home/data1/lyk/Octopus
```

## After the download finishes

Run a full schema and image check locally:

```bash
python scripts/UCIT/prepare_ucit.py --check-images
```

Run the equivalent command on the server:

```bash
python scripts/UCIT/prepare_ucit.py \
  --octopus-root /home/data1/lyk/Octopus \
  --check-images
```

A non-zero exit means an instruction file is missing, its schema is
incompatible, or at least one referenced image cannot be found. Once this
passes, use `data/UCIT/train/<task>.json` as `--data_path` and the Octopus root as
`--image_folder`.

UCIT mixes accuracy tasks and captioning tasks. The existing CL4VQA evaluator
only implements exact-answer VQA accuracy, so benchmark-comparable evaluation
still needs a UCIT-specific evaluator for VizWiz/Flickr30k caption metrics.

## Training

The server defaults match the sibling checkout layout. Override any path with
an environment variable when needed:

```bash
export OCTOPUS_ROOT=/home/data1/lyk/Octopus
export CLMOE_GPUS=0,1
source /home/benke/miniconda3/etc/profile.d/conda.sh
conda activate clmoe
bash scripts/UCIT/train_all.sh
```

To train CL-MoE once on each task's complete UCIT training set, use the
dedicated wrapper:

```bash
bash scripts/UCIT/train_all_full.sh
```

This is still CL-MoE's normal six-task continual-learning chain. Each task is
trained directly for one epoch on its complete training set; there is no
Octopus Stage 1, historical-gradient estimation, or Stage 2. Checkpoints and
router statistics are isolated under `checkpoints/UCIT/full_train/` and
`results/UCIT/full_train/`. The existing `octopus_matched_seed42` experiment is
not modified.

The full-data wrapper supports the same task-boundary continuation syntax:

```bash
bash scripts/UCIT/train_all_full.sh arxivqa
```

If a complete task boundary has been reached, restart from the next task by
passing its name. The wrapper verifies the skipped models and router indices:

```bash
bash scripts/UCIT/train_all.sh arxivqa
```

This is task-level continuation: an interrupted start task is trained again
from step zero because the current configuration does not save intra-task ZeRO
checkpoints.

The default experiment name is `octopus_matched_seed42`. Its artifacts are
kept out of the project root:

```text
checkpoints/UCIT/octopus_matched_seed42/       trained adapters
results/UCIT/octopus_matched_seed42/router_stats/  router count/index files
results/UCIT/octopus_matched_seed42/eval/      predictions and metrics
```

Set `UCIT_RUN_NAME` to keep another configuration in a separate namespace.

The training entry points also accept `MODEL_ROOT`, `UCIT_DATA_ROOT`,
`OUTPUT_ROOT`, `MASTER_PORT`, and `CLMOE_RUNTIME_DIR`. Run one task with:

```bash
bash scripts/UCIT/train_task.sh imagenet_r
bash scripts/UCIT/train_task.sh arxivqa imagenet_r
```

Validate the first task's paths without starting DeepSpeed or loading a model:

```bash
CLMOE_DRY_RUN=1 bash scripts/UCIT/train_task.sh imagenet_r
```

The scripts export the CL-MoE repository root through `PYTHONPATH` before
launching DeepSpeed workers, so the top-level `llava` package remains visible
to every distributed rank.

The training wrapper also uses a project-specific Torch extension cache at
`.cache/torch_extensions_clmoe` and, when available, `/usr/bin/gcc` and
`/usr/bin/g++`. This prevents DeepSpeed CPUAdam binaries built on another host
or against a newer glibc from being reused accidentally. Override
`TORCH_EXTENSIONS_DIR`, `CC`, `CXX`, or `MAX_JOBS` when needed.

## Evaluation

Evaluate every UCIT task using the final continual-learning adapter:

```bash
bash scripts/UCIT/evaluate_all.sh \
  checkpoints/UCIT/octopus_matched_seed42/flickr30k/llava-1.5-7b-lora
```

Accuracy scoring requires `mathruler`. Caption scoring additionally requires
`pycocotools` and `pycocoevalcap`, matching Octopus's evaluation dependencies.
Greedy decoding defaults to 16 new tokens, matching Octopus's vLLM evaluation
configuration. Set `UCIT_MAX_NEW_TOKENS` only for an explicitly named variant.
On the current server these are already installed in the Octopus environment,
so run evaluation from the `clmoe` environment with:

```bash
export UCIT_SCORER_PYTHON=/home/student2/.conda/envs/octopus/bin/python
```

Validate evaluation paths without loading the model:

```bash
UCIT_DRY_RUN=1 bash scripts/UCIT/evaluate_task.sh imagenet_r checkpoint/vicuna-7b-v1.5
```
