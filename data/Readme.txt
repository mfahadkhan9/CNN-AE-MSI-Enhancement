README for Dataset
Project: Optimizing Multispectral Transmission Images for Early Breast Cancer Screening using CNN-AE
Dataset Overview
This dataset is part of the project aimed at denoising multispectral transmission images to improve early breast cancer screening. The images in this dataset are provided at various wavelengths to support testing and replication of the CNN-AE model's effectiveness in image denoising.

Contents
Raw Images: Multispectral images stored in subdirectories based on wavelength.
Sample Images: Example images to demonstrate data format and structure.
Processed Output Samples: Example output images showing denoised results.
Folder Structure
raw_data/ - Contains the original multispectral images used as input.
processed_data/ - Contains sample denoised images produced by the CNN-AE model.
Data Format
Image Format: All images are in .png format.
Resolution: 512x512 pixels (resized in preprocessing).
Usage Instructions
Data Loading: The images can be loaded using standard libraries such as OpenCV or PIL.
Preprocessing: Follow the guidelines in the main project README to resize and preprocess the images before feeding them into the model.
Citation

Contact
For access to the full dataset or additional questions, please contact the corresponding author at zhangtao@tju.edu.cn.