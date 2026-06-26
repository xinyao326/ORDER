# Learning Ordinality-Aware Multimodal Representations for Composite Materials Design [![DOI](https://zenodo.org/badge/1281292743.svg)](https://doi.org/10.5281/zenodo.20929222)

PyTorch implementation of **ORDER** — a framework for learning unified representations from material microstructure images and tabular mechanical properties. It supports cross-modal retrieval, property prediction, and conditional microstructure generation.


## Requirements

```bash
conda env create -f environment.yml
conda activate order
```

Key dependencies: PyTorch 2.8.0 (CUDA 11.7), timm, transformers, open-clip-torch, dalle2-pytorch, cvxpy/cvxopt.

## Datasets

**Composite** (`datasets_composite/`): ~436 samples. 9 tabular features (NumFibers, MMA, Vf, A11–A13, A22, A23, A33), 2 targets (yield strength, elongation). Unzip `datasets_composite/processed.zip` first.

**Nanofiber** (`datasets_fiber/`): ~200 samples. 7 features including categorical fiber direction, 5 targets (fracture, elongation, elastic modulus, tangent modulus, yield).

> **Note on fiber images**: `datasets_fiber/images/preprocessed` should be created by pointing to the actual data location:
```
ln -s /your/path/to/preprocessed datasets_fiber/images/preprocessed
```
The `preprocessed` folder can be obtained from `https://figshare.com/s/0cad763a26f928b70840` (link from [Wu et al.]) under path `images/preprocessed`. 

Wu, Yuhui, et al. "A versatile multimodal learning framework bridging multiscale knowledge for material design." npj Computational Materials 11.1 (2025): 276.

## Quick Start (Interactive Launchers)

All three pipelines are driven by interactive shell scripts. Run from the repo root:

```bash
cd scripts

# Pretraining + cross-modal retrieval evaluation
bash run_pretrain_retrieval.sh

# Property prediction (requires a pretrained checkpoint)
bash run_predict.sh

# Conditional microstructure generation (requires a pretrained checkpoint)
bash run_generation.sh
```

Each script presents a menu to select methods, datasets, and tasks. For full run, simply use the default settings in the scripts. All scripts also support fully non-interactive mode via CLI flags (see below).

## Non-Interactive Usage

### 1. Pretraining & Retrieval


```bash
cd scripts

# Composite dataset — optimal ORDER-α
bash run_pretrain_retrieval.sh \
    --methods "order_dyn order_alpha:0.2" \
    --datasets "composite" \
    --tasks "pretrain retrieval" \
    --device cuda:0 --seed 0

# Nanofiber dataset — optimal ORDER-α
bash run_pretrain_retrieval.sh \
    --methods "order_dyn order_alpha:0.9" \
    --datasets "fiber" \
    --tasks "pretrain retrieval" \
    --device cuda:0 --seed 0
```

Available methods: `order_dyn`, `order_alpha:0.0/0.2/0.5/0.9`, and `_surr` / `_vit16` suffixed variants.

Checkpoints are saved to `scripts/save/<method>/<backbone>/<setting>/`.

### 2. Property Prediction

Requires a pretrained checkpoint from step 1.

```bash
cd scripts

# Composite dataset — optimal ORDER-α
bash run_predict.sh \
    --methods "order_dyn order_alpha:0.2" \
    --datasets "composite" \
    --modalities "tab image fusion" \
    --device cuda:0 --seed 0

# Nanofiber dataset — optimal ORDER-α
bash run_predict.sh \
    --methods "order_dyn order_alpha:0.9" \
    --datasets "fiber" \
    --modalities "tab image fusion" \
    --device cuda:0 --seed 0
```

Modalities: `tab` (tabular only), `image` (image only), `fusion` (both).

Per-task hyperparameters (epochs, lr, dropout, weight decay) are configured in `predict_hparams.sh`.

### 3. Conditional Image Generation

Requires a pretrained checkpoint from step 1. Runs the full pipeline: prior training → decoder training → sampling → evaluation.

```bash
cd scripts
bash run_generation.sh \
    --methods "order_dyn" \
    --datasets "composite fiber" \
    --tasks "train_prior train_decoder generate eval_generate physics_eval" \
    --device cuda:0
```

Generated images are saved to `scripts/save/<method>/<backbone>/<setting>/gen-<split>/`.



## Repository Structure

```
ORDER/
├── src/
│   ├── models/          # OrderModel, FT-Transformer, LoRA-CLIP, MLP heads
│   ├── trainer/         # Pre-training (EPO-LP), fine-tuning, losses, evaluator
│   ├── data/            # Dataset classes for composite and fiber
│   ├── model_config.py  # All hyperparameter defaults
│   └── utils.py         # EPO-LP multi-objective solver
├── scripts/
│   ├── run_pretrain_retrieval.sh
│   ├── run_predict.sh
│   ├── run_generation.sh
│   ├── predict_hparams.sh   # Fine-tuning hyperparameter overrides
│   ├── train_order_dyn.py / train_order_alpha.py
│   ├── train_order_dyn_surr.py / train_order_alpha_surr.py
│   ├── predict.py / fusion_predict.py
│   ├── train_prior.py / train_decoder.py
│   ├── generate.py / eval_generate.py
│   └── demo_physics_metrics.py
├── datasets_composite/
├── datasets_fiber/
└── environment.yml
```
