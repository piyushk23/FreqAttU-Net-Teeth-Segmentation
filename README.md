# FreqAttU-Net: Dental X-Ray Segmentation

FreqAttU-Net is a deep learning model designed for **semantic segmentation of teeth in panoramic dental X-ray images**.

The project enhances the standard **Attention U-Net** architecture with a lightweight **Frequency Enhancement Module (FEM)**. The FEM uses a **2D Fast Fourier Transform (FFT)** to extract frequency-domain features, particularly high-frequency information associated with tooth boundaries. These features are fused with spatial features in the U-Net encoder to improve segmentation quality.

The model was trained and evaluated on **2,885 panoramic dental X-ray image-mask pairs**.

Compared with the Attention U-Net baseline, FreqAttU-Net improved:

- **Dice Score:** 0.8971 → **0.9131**
- **IoU:** 0.8395 → **0.8554**

The project was developed as part of **EE655 – Computer Vision & Deep Learning** at IIT Kanpur. :contentReference[oaicite:0]{index=0}
