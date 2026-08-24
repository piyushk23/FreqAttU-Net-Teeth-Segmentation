FreqAttU-Net: Frequency-Enhanced Attention U-Net for Dental Panoramic X-Ray Segmentation

A deep learning project for binary semantic segmentation of teeth in panoramic dental X-ray images using a frequency-enhanced Attention U-Net architecture.

Overview

Dental panoramic X-rays contain important structural information such as tooth boundaries, but accurate segmentation is challenging because of intensity variations, dental artifacts, overlapping anatomical structures, and foreground-background imbalance.

This project introduces FreqAttU-Net, an Attention U-Net enhanced with a lightweight Frequency Enhancement Module (FEM). The FEM uses a 2D Fast Fourier Transform (FFT) to extract frequency-domain information and learn boundary-aware features. These features are fused with spatial features in the encoder.

The project uses 2,885 image-mask pairs from the Children's Dental Panoramic Radiographs dataset and compares FreqAttU-Net against a standard Attention U-Net baseline.

Key Results

Model

Dice

IoU

Attention U-Net

0.8971

0.8395

FreqAttU-Net

0.9131

0.8554

The proposed model improves the baseline by:

+1.6% Dice

+1.6% IoU

Only approximately 37K additional parameters for the FEM

Approximately 7.88M total trainable parameters

The validation Dice score reaches approximately 0.924, with the metrics plateauing around epoch 30.

Architecture

FreqAttU-Net follows a 4-level U-Net encoder-decoder architecture with attention gates on all skip connections.

Main Components

Frequency Enhancement Module (FEM)

Applies a 2D FFT to the input X-ray.

Extracts the FFT magnitude.

Applies log1p compression to stabilize the frequency representation.

Bilinearly resizes the frequency map to the original image resolution.

Uses two convolutional layers to learn frequency-aware features.

Frequency-Spatial Fusion

Frequency features are additively fused with the first encoder level.

Downsampled frequency features are concatenated with the second encoder level.

Attention Gates

Applied to all four skip connections.

Use decoder context to emphasize relevant tooth regions and suppress irrelevant background such as jawbone and soft tissue.

U-Net Encoder-Decoder

Base filter count: 32

Encoder channels: 32, 64, 128, 256

Bottleneck: 512 channels

Transposed convolutions are used for decoder upsampling.

Frequency Enhancement Module

The FEM follows this pipeline:

Input X-Ray
    |
    v
2D FFT
    |
    v
Magnitude Extraction
    |
    v
Log-Magnitude Compression
    |
    v
Bilinear Upsampling
    |
    v
Two Convolutional Layers
    |
    v
Frequency-Aware Features
    |
    +----------------------+
    |                      |
    v                      v
Encoder Level 1       Encoder Level 2
(Addition)            (Concatenation)

Tooth boundaries contain abrupt intensity transitions and therefore correspond to high-frequency components in the Fourier domain. The FEM provides this frequency information as a complementary boundary prior to spatial convolution features.

Dataset

The project uses the Children's Dental Panoramic Radiographs Dataset from Kaggle.

After filtering for image-mask pairs with binary segmentation masks:

Total pairs: 2,885

Train/Validation/Test split: 70/15/15

Split method: stratified random sampling

Random seed: 42

Splits are patient-disjoint

Dataset:
https://www.kaggle.com/datasets/truthisneverlinear/childrens-dental-panoramic-radiographs-dataset

Preprocessing and Augmentation

Images are:

Resized to 512 × 512

Normalized to [-1, 1]

Training augmentation includes:

Random horizontal flips

Random vertical flips

Rotation up to ±15°

Brightness jitter

Contrast jitter

Gaussian noise

Training Configuration

Parameter

Value

Optimizer

AdamW

Initial Learning Rate

1e-4

Weight Decay

1e-5

Scheduler

Cosine Annealing

Epochs

40

Batch Size

4

GPU

NVIDIA T4 16GB

Base Filters

32

Loss Function

A combined Binary Cross-Entropy and Dice loss is used:

L = 0.5 × BCE + 0.5 × Dice Loss

Dice loss helps address the foreground-background class imbalance, while BCE provides pixel-level classification supervision.

Evaluation Metrics

Dice Similarity Coefficient

Dice = 2TP / (2TP + FP + FN)

Intersection over Union

IoU = TP / (TP + FP + FN)

Both metrics are computed after applying a sigmoid activation followed by a threshold of 0.5.

Ablation Study

The effect of the Frequency Enhancement Module was evaluated by comparing the baseline Attention U-Net with FreqAttU-Net.

Variant

FEM

Dice

IoU

Attention U-Net

✗

0.8971

0.8395

FreqAttU-Net

✓

0.9131

0.8554

The results show that adding FEM alone improves both Dice and IoU without changing the remaining architecture.

Qualitative Results

The model successfully segments both upper and lower dental arches while excluding jawbone and soft tissue. Test examples achieved Dice scores of 0.925 and 0.940.

The report also notes that the model remained accurate on an example containing bright dental metalwork, indicating robustness to imaging artifacts.

Repository Structure

A recommended repository structure is:

FreqAttU-Net-Dental-X-Ray-Segmentation/
│
├── EE655_Project_Report.ipynb
├── README.md
├── requirements.txt
│
├── data/
│   ├── images/
│   └── masks/
│
├── checkpoints/
│
└── results/
    ├── training_curves/
    ├── qualitative_results/
    └── visualizations/

The complete implementation is provided in the accompanying Jupyter notebook:

EE655_Project_Report.ipynb

Running the Project

1. Clone the repository

git clone https://github.com/atits23/-FreqAttU-Net-Dental-X-Ray-Segmentation.git
cd -FreqAttU-Net-Dental-X-Ray-Segmentation

2. Install dependencies

The implementation uses Python and PyTorch along with common scientific and image-processing libraries.

Typical dependencies include:

pip install torch torchvision numpy pandas matplotlib scikit-learn opencv-python

3. Download the dataset

Download the Children's Dental Panoramic Radiographs Dataset from Kaggle and organize the image-mask pairs according to the notebook's expected data paths.

4. Run the notebook

Open:

EE655_Project_Report.ipynb

and execute the cells sequentially for preprocessing, model construction, training, evaluation, and visualization.

Key Implementation Details

The FEM performs:

fft = torch.fft.rfft2(x, norm='ortho')
magnitude = torch.abs(fft)
magnitude = torch.log1p(magnitude)
magnitude = F.interpolate(
    magnitude,
    size=(x.shape[2], x.shape[3]),
    mode='bilinear',
    align_corners=False
)
freq_feat = self.conv(magnitude)

The frequency features are then fused with the first encoder level:

e1 = self.enc1(x)
e1 = e1 + freq_feat

and injected again at encoder level 2:

e2_in = torch.cat(
    [self.pool(e1), self.pool(freq_feat)],
    dim=1
)

References

Ronneberger, O. et al. U-Net: Convolutional Networks for Biomedical Image Segmentation, MICCAI, 2015.

Oktay, O. et al. Attention U-Net: Learning Where to Look for the Pancreas, MIDL, 2018.

Chi, L., Jiang, B., and Mu, Y. Fast Fourier Convolution, NeurIPS, 2020.

truthisneverlinear. Children's Dental Panoramic Radiographs Dataset, Kaggle, 2023.

Reference implementations:

Attention U-Net: https://github.com/ozan-oktay/Attention-Gated-Networks

U-Net: https://github.com/milesial/Pytorch-UNet

Team

This project was developed as part of EE655 – Computer Vision & Deep Learning.

Member

Contribution

Deepak Kumar Meena

FEM design and implementation, FFT feature extraction, frequency-domain literature review and FEM visualization

Piyush Kumar

Attention U-Net baseline, attention gates, combined BCE + Dice loss and ablation study

Anjan Das

Dataset acquisition, preprocessing, augmentation, normalization and dataset analysis

Atit Kumar Satsangi

FreqAttU-Net integration, training loop, hyperparameter tuning and GPU training

Amey Dikshit

Quantitative evaluation, visualization, report editing and repository documentation

Acknowledgements

We acknowledge the authors of the Children's Dental Panoramic Radiographs Dataset for making the dataset publicly available on Kaggle.

We also thank Professor Koteswar Rao Jerripothula for teaching the foundational concepts of Computer Vision and Deep Learning in EE655.
