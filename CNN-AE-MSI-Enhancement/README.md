# CNN-AE-MSI Enhancement

## Overview

This repository provides a clean and reproducible implementation of a convolutional neural network autoencoder (CNN-AE) framework for registration-oriented enhancement of low-SNR multispectral transmission images.

The framework is designed for multispectral transmission breast-imaging sequences acquired at four wavelength bands: **600 nm, 620 nm, 670 nm, and 760 nm**. It improves degraded image frames by reducing noise, preserving structural information, and supporting more reliable frame accumulation and registration-oriented assessment.

The code is organized for journal submission, reproducibility, and GitHub release.

## Main features

- CNN-autoencoder architecture with encoder, bottleneck, and decoder stages.
- Conv2D + BatchNorm2D + ReLU encoder blocks.
- ConvTranspose2D decoder blocks with sigmoid output normalization.
- Wavelength-specific image organization for 600 nm, 620 nm, 670 nm, and 760 nm data.
- Training, inference, preprocessing, and evaluation scripts.
- Evaluation using coefficient correlation (CC), mutual information (MI), root mean square error (RMSE), peak signal-to-noise ratio (PSNR), and registration-oriented processing time (RT).
- Structured folders for data, checkpoints, results, figures, and documentation.

## Repository structure

```text
CNN-AE-MSI-Enhancement/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── CITATION.cff
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── README.md
│   └── sample/
│       ├── 600nm/
│       ├── 620nm/
│       ├── 670nm/
│       └── 760nm/
│
├── src/
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── preprocess.py
│   ├── metrics.py
│   └── utils.py
│
├── checkpoints/
│   └── README.md
│
├── results/
│   ├── README.md
│   ├── enhanced_images/
│   ├── figures/
│   └── metrics/
│
├── scripts/
│   ├── run_training.sh
│   └── run_evaluation.sh
│
└── docs/
    └── methodology.md
```

## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

Recommended Python version: **Python 3.9 or later**.

## Dataset preparation

Organize the extracted image frames as follows:

```text
data/
├── train/
│   ├── 600nm/
│   ├── 620nm/
│   ├── 670nm/
│   └── 760nm/
├── val/
│   ├── 600nm/
│   ├── 620nm/
│   ├── 670nm/
│   └── 760nm/
└── test/
    ├── 600nm/
    ├── 620nm/
    ├── 670nm/
    └── 760nm/
```

The images are resized to **256 × 256** pixels and normalized to the range `[0, 1]` during training and evaluation.

The full dataset used in the manuscript can be linked here:

```text
Zenodo DOI: https://doi.org/10.5281/zenodo.14037985
```

If additional restricted experimental data are required, contact the corresponding author.

## Configuration

Edit the experiment settings in:

```text
configs/config.yaml
```

Important fields include:

```yaml
data:
  train_dir: data/train
  val_dir: data/val
  test_dir: data/test
  image_size: 256

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
```

If clean paired targets are not available, leave the target directories empty. The code will train the autoencoder in self-reconstruction mode.

## Training

Run:

```bash
python src/train.py --config configs/config.yaml
```

or:

```bash
bash scripts/run_training.sh
```

Training outputs are saved to:

```text
checkpoints/
results/metrics/training_history.csv
results/figures/training_loss.png
```

## Evaluation

Run:

```bash
python src/evaluate.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth
```

Optional Non-Local Means baseline comparison:

```bash
python src/evaluate.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth --compare_nlm
```

Evaluation outputs are saved to:

```text
results/enhanced_images/
results/metrics/evaluation_metrics.csv
results/metrics/summary_metrics.json
```

## Inference on new images

Run:

```bash
python src/inference.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --input_dir data/test \
  --output_dir results/enhanced_images
```

## Preprocessing

To resize images, generate grayscale outputs, and save histograms:

```bash
python src/preprocess.py \
  --input_dir data/train \
  --output_dir results/preprocessed \
  --image_size 256
```

## Model architecture

The CNN-AE follows an encoder--bottleneck--decoder design:

- Input: RGB image, 256 × 256 × 3.
- Encoder: three Conv2D blocks with BatchNorm2D and ReLU.
- Bottleneck: 32 × 32 × 256 latent representation.
- Decoder: three ConvTranspose2D reconstruction stages.
- Output: sigmoid activation, normalized to `[0, 1]`.

## Evaluation metrics

The repository computes the following metrics:

- **CC**: coefficient correlation.
- **MI**: mutual information.
- **RMSE**: root mean square error.
- **PSNR**: peak signal-to-noise ratio.
- **RT**: registration-oriented processing time per image.

These metrics should be interpreted jointly because each evaluates a different aspect of restoration and registration-oriented enhancement.

## Citation

If you use this repository, please cite the associated manuscript:

```text
Muhammad Fahad et al. CNN-AE-Based Registration-Oriented Enhancement of Low-SNR Multispectral Transmission Images for Breast Imaging. Submitted to The Visual Computer.
```

This citation will be updated after publication.

## License

This project is released under the MIT License. See the `LICENSE` file for details.
