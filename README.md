# CNN-AE-MSI Enhancement

## Overview

This repository provides a reproducible implementation of a convolutional neural network autoencoder (CNN-AE) framework for registration-oriented enhancement of low-SNR multispectral transmission images.

The framework is designed for multispectral transmission breast optical imaging sequences acquired at four wavelength bands: **600 nm, 620 nm, 670 nm, and 760 nm**. The CNN-AE improves degraded image frames by reducing noise, preserving structural information, and supporting frame accumulation and registration-oriented assessment.

This repository supports the manuscript:

**CNN-AE-Based Registration-Oriented Enhancement of Low-SNR Multispectral Transmission Images for Breast Optical Imaging**

The proposed CNN-AE should be interpreted as an image restoration and registration-oriented preprocessing framework. It does **not** directly estimate deformation fields, displacement vectors, or transformation matrices.

## Repository purpose

This repository was prepared to improve reproducibility and transparency for journal review. It includes source code, configuration files, dependency information, training and inference commands, evaluation scripts, representative sample images, and instructions for testing the workflow.

The public sample images allow users to verify that the preprocessing, inference, and metric-computation pipeline runs correctly. Exact reproduction of the manuscript’s quantitative tables requires access to the complete experimental dataset, which is not fully released publicly and may be requested from the corresponding author subject to institutional approval and data-use conditions.

## Main features

* CNN-autoencoder architecture with encoder, bottleneck, and decoder stages.
* Conv2D + BatchNorm2D + ReLU encoder blocks.
* ConvTranspose2D decoder blocks with sigmoid output normalization.
* Wavelength-specific organization for 600 nm, 620 nm, 670 nm, and 760 nm images.
* Self-reconstruction training mode when paired clean ground-truth images are unavailable.
* Preprocessing, training, inference, and evaluation scripts.
* Evaluation using coefficient correlation (CC), mutual information (MI), root mean square error (RMSE), peak signal-to-noise ratio (PSNR), and registration-oriented processing time (RT).
* Representative sample images for independent workflow testing.
* Structured folders for configuration files, data, source code, scripts, results, and documentation.

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
├── docs/
│   └── methodology.md
│
├── results/
│   ├── README.md
│   ├── enhanced_images/
│   │   ├── processed_600nm/
│   │   ├── processed_620nm/
│   │   ├── processed_670nm/
│   │   └── processed_760nm/
│   ├── figures/
│   └── metrics/
│
├── scripts/
│   ├── run_training.sh
│   └── run_evaluation.sh
│
└── src/
    ├── dataset.py
    ├── evaluate.py
    ├── inference.py
    ├── metrics.py
    ├── model.py
    ├── preprocess.py
    ├── train.py
    └── utils.py
```

Note: If your local results folders are named `processed 600nm`, `processed 620nm`, `processed 670nm`, and `processed 760nm`, the code will still work if the paths are handled consistently. However, folder names without spaces, such as `processed_600nm`, are recommended for reproducibility.

## Installation

Create a Python environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

Recommended Python version:

```text
Python 3.9 or later
```

The main dependencies are listed in `requirements.txt` and may include:

```text
torch
torchvision
numpy
opencv-python
scikit-image
matplotlib
pandas
tqdm
PyYAML
```

## Data availability

This repository includes **representative sample multispectral transmission images** for independent testing of the preprocessing, inference, and evaluation workflow.

The complete experimental dataset used to generate the quantitative results reported in the manuscript is **not fully released publicly** because it forms part of an institutional experimental breast optical imaging dataset and may be subject to institutional, ethical, or data-use restrictions.

Access to the complete experimental dataset may be requested from the corresponding author and will be considered upon reasonable request, subject to institutional approval and relevant data-use conditions.

Corresponding author:

```text
Tao Zhang
Email: zhangtao@tju.edu.cn
```

Important note:

The sample images provided in this repository allow users to verify the functionality of the code pipeline. However, exact reproduction of the manuscript’s quantitative tables requires access to the complete experimental dataset used in the study.

## Dataset preparation

For testing with the provided sample images, use the following structure:

```text
data/sample/
├── 600nm/
├── 620nm/
├── 670nm/
└── 760nm/
```

For full training and evaluation, organize the extracted image frames as follows:

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

All images are resized to **256 × 256 pixels** and normalized to the range `[0, 1]` during preprocessing, training, and evaluation.

## Configuration

Experiment settings are defined in:

```text
configs/config.yaml
```

Important configuration fields include:

```yaml
data:
  train_dir: data/train
  val_dir: data/val
  test_dir: data/test
  sample_dir: data/sample
  image_size: 256

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  mode: self_reconstruction

output:
  checkpoint_dir: checkpoints
  result_dir: results
```

If paired clean ground-truth images are not available, the model can be trained in **self-reconstruction mode**, where the normalized input frame is used as the reconstruction target.

## Preprocessing

To resize images, generate grayscale outputs, and save histograms for full training data:

```bash
python src/preprocess.py \
  --input_dir data/train \
  --output_dir results/preprocessed \
  --image_size 256
```

To preprocess the provided sample images:

```bash
python src/preprocess.py \
  --input_dir data/sample \
  --output_dir results/preprocessed \
  --image_size 256
```

## Training

To train the CNN-AE model:

```bash
python src/train.py --config configs/config.yaml
```

Alternatively, use the shell script:

```bash
bash scripts/run_training.sh
```

Training outputs are saved to:

```text
checkpoints/
results/metrics/training_history.csv
results/figures/training_loss.png
```

## Inference

To run inference using a trained checkpoint:

```bash
python src/inference.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --input_dir data/test \
  --output_dir results/enhanced_images
```

To test the workflow using the provided sample images:

```bash
python src/inference.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --input_dir data/sample \
  --output_dir results/enhanced_images
```

## Evaluation

To evaluate the trained model:

```bash
python src/evaluate.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best_model.pth
```

Optional Non-Local Means baseline comparison:

```bash
python src/evaluate.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --compare_nlm
```

Evaluation outputs are saved to:

```text
results/enhanced_images/
results/metrics/evaluation_metrics.csv
results/metrics/summary_metrics.json
```

The evaluation script computes:

* **CC**: coefficient correlation.
* **MI**: mutual information.
* **RMSE**: root mean square error.
* **PSNR**: peak signal-to-noise ratio.
* **RT**: registration-oriented enhancement/inference time per image.

## Reproducing the reported quantitative workflow

The following steps describe the workflow used to reproduce the evaluation pipeline.

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Prepare the dataset

Arrange the complete experimental dataset into wavelength-specific folders:

```text
data/train/600nm/
data/train/620nm/
data/train/670nm/
data/train/760nm/

data/val/600nm/
data/val/620nm/
data/val/670nm/
data/val/760nm/

data/test/600nm/
data/test/620nm/
data/test/670nm/
data/test/760nm/
```

For sample testing only, use:

```text
data/sample/600nm/
data/sample/620nm/
data/sample/670nm/
data/sample/760nm/
```

### Step 3: Run preprocessing

```bash
python src/preprocess.py \
  --input_dir data/train \
  --output_dir results/preprocessed \
  --image_size 256
```

### Step 4: Train the CNN-AE

```bash
python src/train.py --config configs/config.yaml
```

### Step 5: Run inference

```bash
python src/inference.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --input_dir data/test \
  --output_dir results/enhanced_images
```

### Step 6: Compute metrics

```bash
python src/evaluate.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best_model.pth
```

The computed metrics are saved under:

```text
results/metrics/
```

### Reproducibility note

The public sample images can be used to test whether preprocessing, inference, and metric computation run correctly. Exact reproduction of the manuscript’s quantitative tables requires access to the complete experimental dataset used in the study. The complete dataset may be requested from the corresponding author, subject to institutional approval and data-use conditions.

## Model architecture

The CNN-AE follows an encoder--bottleneck--decoder design:

* Input: RGB image, 256 × 256 × 3.
* Encoder: three Conv2D blocks with BatchNorm2D and ReLU activation.
* Bottleneck: 32 × 32 × 256 latent representation.
* Decoder: three ConvTranspose2D reconstruction stages.
* Output: sigmoid activation, normalized to `[0, 1]`.

The model is optimized using mean squared error loss. When paired clean ground-truth images are unavailable, the model uses a self-reconstruction strategy in which the normalized input image is used as the reconstruction target.

## Target-image definition

No independently acquired clean or noise-free ground-truth multispectral transmission images are included in this repository. Therefore, the CNN-AE can be trained using a self-reconstruction strategy. In this mode, the target image is the corresponding preprocessed and normalized input frame.

This means that RMSE and PSNR should be interpreted as reconstruction-fidelity metrics relative to the defined target or reference image, not as full-reference denoising metrics against an independently clean ground-truth image.

## Evaluation metrics

The repository computes the following metrics.

### Coefficient correlation

CC measures the linear similarity between a reference image and an enhanced image. Higher values indicate stronger structural similarity.

### Mutual information

MI measures the shared information between two images. Higher values generally indicate greater shared intensity information, but MI can be sensitive to changes in intensity distribution.

### Root mean square error

RMSE measures pixel-level reconstruction difference. Lower values indicate lower reconstruction error.

### Peak signal-to-noise ratio

PSNR measures reconstruction quality in decibels. Higher values indicate lower distortion.

### Registration-oriented processing time

RT measures the CNN-AE enhancement/inference time per image for the proposed method. It does not represent deformation-field estimation, displacement-vector estimation, or transformation-matrix estimation time.

These metrics should be interpreted jointly because each evaluates a different aspect of image restoration and registration-oriented enhancement. The proposed method should not be interpreted as universally superior for every metric.

## Interpretation of registration-oriented enhancement

The proposed CNN-AE does not directly perform geometric registration. It does not estimate a deformation field, displacement vector, or transformation matrix. Instead, it enhances low-SNR multispectral transmission images by improving structural visibility and reducing noise. The enhanced images can support registration-oriented assessment, frame accumulation, and downstream visual analysis.

## Limitations

* The complete experimental dataset is not fully released publicly.
* Public sample images are provided for workflow testing, not for exact reproduction of all manuscript tables.
* Exact reproduction of reported quantitative values requires access to the complete experimental dataset.
* The CNN-AE does not directly estimate geometric transformations.
* The model is evaluated as a preprocessing and image-enhancement framework, not as a clinical diagnostic or screening system.
* Broader validation requires larger independent datasets, repeated acquisitions, and more diverse imaging conditions.

## Citation

If you use this repository, please cite the associated manuscript:

```text
Muhammad Fahad et al. CNN-AE-Based Registration-Oriented Enhancement of Low-SNR Multispectral Transmission Images for Breast Optical Imaging. Submitted to The Visual Computer.
```

This citation will be updated after publication.

## Code archive

The source code is archived on Zenodo:

```text
https://doi.org/10.5281/zenodo.20710396
```

## License

This project is released under the MIT License. See the `LICENSE` file for details.
