# SPARC

Official implementation of **SPARC**, an exemplar-free class-incremental learning framework based on Vision Transformers.

SPARC learns new classes without storing samples from previous tasks. The implementation combines task-specific prompt-expert adaptation, current-task classification constraints, cross-stage feature distillation, and dual-space prototype-based inference.

## Main components

- **SPEA**: task-specific prompt-expert adaptation.
- **MPVA**: multi-scale prompt-visual adapter.
- **SPIC / CTCR**: classification constraints over the classes of the current task.
- **CSFD**: cross-stage feature distillation.
- **DPHC**: dual-space prototype-based task relevance and logit calibration.

## Repository structure

```text
SPARC/
├── main_sparc.py                  # Training and evaluation entry point
├── sparc.py                       # SPARC model and incremental-learning components
├── datasets.py                    # Dataset loading and class filtering
├── engine.py                      # Evaluation utilities
├── samplers.py                    # Data samplers
├── utils.py                       # Logging, metrics, and distributed utilities
└── SDT2Net/
    └── EAB_FTF_vit_fixed.py       # ViT backbone and incremental classifier
```

## Requirements

- Python 3.8 or later
- PyTorch and torchvision with CUDA support
- An NVIDIA GPU

Create an environment and install the dependencies:

```bash
conda create -n sparc python=3.9 -y
conda activate sparc

pip install torch torchvision
pip install timm==0.6.13 numpy termcolor
```

The current training loop calls CUDA operations directly, so CPU-only execution is not supported without modifying the code.

## Pretrained model

SPARC uses the ImageNet-pretrained **DeiT-Small/16** checkpoint:

[`deit_small_patch16_224-cd65a155.pth`](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth)

Download the checkpoint and place it at:

```text
SPARC/model/pretrain/deit_small_patch16_224-cd65a155.pth

## Dataset preparation

The code supports the following dataset identifiers:

| Dataset | Argument | Number of classes |
| --- | --- | ---: |
| UC Merced Land Use | `UCM` | 21 |
| AID | `AID` | 30 |
| RSI-CB256 | `RSI-CB256` | 35 |
| NWPU-RESISC45 | `NWPU` | 45 |

For UCM, AID, RSI-CB256, NWPU-RESISC45 arrange the data in ImageFolder format:

```text
DATASET_ROOT/
├── train/
│   ├── class_000/
│   │   ├── image_001.jpg
│   │   └── ...
│   ├── class_001/
│   └── ...
└── val/
    ├── class_000/
    ├── class_001/
    └── ...
```



## Training

Run SPARC on UCM with 7 classes per incremental task:

```bash
python main_sparc.py \
  --data-set UCM \
  --data-path /path/to/UCM \
  --classes-per-task 7 \
  --epochs 50 \
  --batch-size 64 \
  --output_dir ./log/ucm_sparc_exemplar_free
```

The default backbone is `vit_deit_SDT2Net_small_patch16_224`. When `--finetune` is not provided, the code attempts to download pretrained ViT weights on the first run.

To initialize the backbone from a local checkpoint:

```bash
python main_sparc.py \
  --data-set UCM \
  --data-path /path/to/UCM \
  --classes-per-task 7 \
  --finetune /path/to/checkpoint.pth \
  --output_dir ./log/ucm_sparc_exemplar_free
```

## Important arguments

| Argument | Default | Description |
| --- | ---: | --- |
| `--batch-size` | `64` | Training batch size |
| `--epochs` | `50` | Training epochs for each incremental task |
| `--classes-per-task` | `7` | Number of new classes introduced per task |
| `--num-prompts` | `10` | Number of prompt tokens |
| `--expert-rank` | `16` | Rank of the prompt-expert adapter |
| `--lambda-fd` | `1.0` | Weight of cross-stage feature distillation |
| `--lr` | `1e-3` | Learning rate |
| `--weight-decay` | `0.05` | AdamW weight decay |
| `--data-set` | `UCM` | Dataset identifier |
| `--data-path` | empty | Dataset root directory |
| `--finetune` | empty | Optional local pretrained checkpoint |
| `--output_dir` | `./log/ucm_sparc_exemplar_free/` | Output directory |
| `--seed` | `0` | Random seed; values greater than 0 also shuffle class order |

Use the help command to view all available options:







