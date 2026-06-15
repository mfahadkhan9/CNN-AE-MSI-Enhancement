# Methodology summary

This repository implements a CNN-autoencoder-assisted framework for registration-oriented enhancement of low-SNR multispectral transmission images.

The model follows an encoder--bottleneck--decoder architecture:

1. Encoder: Conv2D + BatchNorm2D + ReLU blocks.
2. Bottleneck: compact latent representation for structural feature preservation.
3. Decoder: ConvTranspose2D reconstruction blocks.
4. Output: sigmoid activation to constrain image intensities to `[0, 1]`.

The default training mode is self-reconstruction, where the input image is also used as the target. This is appropriate when clean paired targets are not available. If paired targets are available, specify the corresponding target directories in `configs/config.yaml`.

Evaluation metrics include coefficient correlation (CC), mutual information (MI), root mean square error (RMSE), peak signal-to-noise ratio (PSNR), and registration-oriented processing time (RT).
