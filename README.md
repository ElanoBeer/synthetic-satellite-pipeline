# Synthetic Optical Satellite Imagery Generation Pipeline  

## 📌 Overview  
This repository contains the implementation of a **synthetic data generation pipeline for optical satellite imagery**, developed as part of my master's thesis. The project explores **generative models** for creating high-fidelity satellite images, aiming to enhance AI-driven maritime surveillance. The goal is to provide **diverse, high-quality training data** to improve deep learning models for vessel detection.
## 🎯 Objectives  
- Develop a **synthetic data generation framework** tailored for maritime satellite imagery.  
- Utilize **Generative Adversarial Networks (GANs) and diffusion models** to create realistic images.  
- Improve **deep learning model robustness** to variations in weather, lighting, and vessel types.  
- Bridge the **data gap in maritime surveillance** by generating high-fidelity, annotated training datasets.  

## 🏗️ Pipeline Components  
The **synthetic imagery generation pipeline** consists of the following key components:  

### 1️⃣ **Data Collection & Preprocessing**  
- Sources: Real-world **optical and SAR satellite imagery**.  
- Preprocessing: Image **normalization, augmentation, and segmentation**.  
- Object extraction: **Ship segmentation models (e.g., YOLACT, Mask R-CNN)** for dataset preparation.  

### 2️⃣ **Synthetic Image Generation**  
- **GAN-based approach**: U-Net GANs, StyleGAN, and CycleGAN.  
- **Diffusion models**: Alternative generative approach for high-resolution realism.  
- **Object insertion techniques**: Context-aware ship placement (guided by object detection strategies).  

### 3️⃣ **Evaluation & Validation**  
- **Fidelity metrics**: Fréchet Inception Distance (FID), Structural Similarity Index (SSIM).  
- **Realism assessment**: Human expert evaluation and histogram comparison (HOG).  
- **Model performance**: Training deep learning-based vessel detection models with synthetic data.  

### 4️⃣ **Maritime Anomaly Simulation**  
- **Dark fleet behavior modeling**: Simulating AIS-spoofing and untracked vessel movement.  
- **Adverse weather simulation**: Generating images with occlusion factors (clouds, fog, reflections).  

---

## 📂 Repository Structure  
This repository is organized as follows:  

```plaintext
📦 synthetic-satellite-imagery-pipeline  
 ┣ 📂 data               # Raw and processed dataset (real & synthetic images)  
 ┣ 📂 models             # Pre-trained and custom deep learning models  
 ┣ 📂 notebooks          # Jupyter notebooks for training & evaluation  
 ┣ 📂 src                # Core pipeline implementation (GANs, diffusion models, object insertion)  
 ┃ ┣ 📜 data_loader.py   # Scripts for loading and preprocessing satellite images  
 ┃ ┣ 📜 train_gan.py     # Training script for generative models  
 ┃ ┣ 📜 evaluate.py      # Metrics evaluation (FID, SSIM, etc.)  
 ┣ 📂 results            # Generated synthetic images & validation results  
 ┣ 📂 docs               # Project documentation & methodology details  
 ┣ 📜 requirements.txt   # Dependencies & environment setup  
 ┣ 📜 README.md          # Project overview  
 ┗ 📜 LICENSE            # License information  
