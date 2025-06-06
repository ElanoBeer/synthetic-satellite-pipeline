# Synthetic Optical Satellite Imagery Generation Pipeline
## 📌 Overview
This repository contains the implementation of a **synthetic data generation pipeline for optical satellite imagery**, developed as part of a research project focused on maritime surveillance. The pipeline leverages various generative models to create realistic satellite imagery, with a specific focus on maritime vessel detection and anomaly identification.
## 🎯 Objectives
- Create a versatile **synthetic data generation framework** for maritime satellite imagery
- Implement and evaluate multiple generative approaches including **GANs, VAEs, and object insertion techniques**
- Generate diverse training data to improve **vessel detection performance** across varying conditions
- Simulate realistic maritime scenarios including **vessel anomalies** and challenging weather conditions

## 🏗️ Pipeline Components
### 1️⃣ **Data Processing & Preparation**
- **Image preprocessing**: Normalization, patching, and augmentation
- **Object extraction**: Vessel segmentation from source imagery
- **Dataset management**: Tools for organizing real and synthetic datasets (MASATI-v2, S2Ships)

### 2️⃣ **Synthetic Generation Methods**
- **Generative models**:
    - VAE implementation for satellite imagery reconstruction
    - Wasserstein GAN with gradient penalty for satellite image generation
    - Context Encoder with PatchGAN for generative inpainting

- **Image Augmentation**:
  - Fast and effective dataset augmentation using Albumentations

- **Object insertion techniques**:
    - Context-aware vessel placement
    - Seamless cloning for realistic integration

- **Environmental simulation**:
    - Cloud generation and weather effects
    - Maritime condition variations

### 3️⃣ **Vessel Detection & Evaluation**
- **YOLOv8 implementation**: Optimized for maritime vessel detection
- **Cross-validation frameworks**: Comprehensive testing across datasets
- **Performance metrics**: Precision, recall, and specialized maritime detection metrics

## 📂 Repository Structure
``` 
📦 synthetic-satellite-pipeline
 ┣ 📂 data                      # Dataset storage (not tracked in git)
 ┣ 📂 evaluation                # Directory for evaluation scripts
 ┃ ┣ 📜 cloud_mask_generator.py # Generate cloud masks
 ┃ ┣ 📜 eval_cloud_obstruction.py # Evaluate simulated cloud obstruction
 ┃ ┣ 📜 vessel_count_eval.py    # Evaluation of object insertion counts
 ┣ 📂 images                    # Generated imagery and evaluation thesis results
 ┣ 📂 models                    # Model implementations
 ┃ ┣ 📂 yolov8-n                # YOLOv8 vessel detection implementation
 ┃ ┣ 📂 context_encoder         # Context Encoder models
 ┃ ┣ 📂 wgan_gp                 # Wasserstein GAN with Gradient Penalty models
 ┃ ┗ 📂 vae                     # Variational Autoencoder models
 ┣ 📂 notebooks                 # Jupyter notebooks for experiments and visualizations
 ┃ ┣ 📜 cv_evaluations.ipynb    # Main CV Results and visualizations
 ┃ ┣ 📜 eurosat_gan.ipynb       # GAN training on EuroSAT data
 ┃ ┣ 📜 image_augmentation_utils.ipynb # Testing utilities for image augmentation
 ┃ ┣ 📜 masativ2_testing.ipynb  # Exploratory data analysis and testing with Masati-v2
 ┃ ┣ 📜 object_exploration.ipynb # Vessel object analysis
 ┃ ┣ 📜 s2ships_testing.ipynb   # Exploratory data analysis and testing with S2Ships
 ┃ ┣ 📜 seamless_cloning.ipynb  # Object insertion experiments
 ┃ ┣ 📜 test_cloud_generator.ipynb # Cloud simulation testing
 ┃ ┣ 📜 test_pipeline_components.ipynb # Component testing
 ┃ ┗ 📜 vae_testing.ipynb       # Extensive testing of VAE's generative capabilities
 ┣ 📂 src                       # Core implementation
 ┃ ┣ 📜 basic_augmentation.py   # Data augmentation techniques
 ┃ ┣ 📜 cloud_generator.py      # Cloud and weather simulation
 ┃ ┣ 📜 detection_attempt.py    # Prior object detection attempt with other models
 ┃ ┣ 📜 object_insertion.py     # Vessel placement algorithms
 ┃ ┣ 📜 pipeline.py             # Run pre-generation pipeline
 ┃ ┗ 📜 synthetic_generation.py # Main pipeline orchestration
 ┣ 📂 utils                     # Utility scripts
 ┃ ┣ 📜 annotation_converter.py # Converting annotation formats
 ┃ ┣ 📜 boat_extraction.py      # Vessel extraction utilities
 ┃ ┣ 📜 dataset_creator.py      # Create dataset configurations
 ┃ ┣ 📜 image_patching.py       # Script to patch large satellite imagery
 ┗ 📜 .gitignore                # Intentionally untracked files
 ┗ 📜 .python-version           # Specified Python version for pyenv
 ┗ 📜 pyproject.toml            # Dependencies and configurations for project
 ┗ 📜 README.md                 # Project documentation
 ┗ 📜 uv.lock                   # Dependency lock file generated by uv
```
## 🔧 Key Features
### Vessel Extraction & Insertion
The pipeline implements sophisticated object extraction and insertion techniques to create realistic vessel placements in maritime scenes:
- Contextual vessel positioning based on water/land segmentation
- Seamless blending with background maritime environments
- Scale and orientation adjustments for realism

### Synthetic Environment Generation
Environmental variations are simulated to enhance dataset diversity:
- Cloud and atmospheric condition generation
- Maritime surface variations (waves, reflections)
- Time-of-day and seasonal variations

### YOLOv8 Vessel Detection
An optimized YOLOv8 implementation provides state-of-the-art vessel detection:
- Fine-tuned for maritime vessel detection scenarios
- Comprehensive evaluation across synthetic and real datasets
- Performance analysis with various training data combinations

## 🚀 Getting Started
### Requirements
The project requires Python 3.11 and the following key dependencies:
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Ultralytics YOLOv8

Although the project was created using a virtual environment, the dependencies have been exported to requirements.txt and poetry files.

## ❗ Running Image Generation Models
Important: The Pytorch implementation were performed using a separate environment, which utilizes CUDA. It is inspired by the following GitHub repository: https://github.com/eriklindernoren/PyTorch-GAN
Nevertheless, its dependencies can not be integrated with packages used within this project and thus should be performed in a new environment as well.
The integration within this repo is solely to have all utilized scripts in one place. More information on these dependencies:

- torch: 2.6.0+cu126
- torchaudio: 2.6.0+cu126
- torchvision: 2.6.0+cu126
- CUDA: 12.6

### Machine Specifications
The experiments in this thesis were performed in the following machine. However, with the resource allocation being maximized, an improved machine configurations would be preferred.
- GPU: Nvidia RTX 2070 8GB VRAM
- CPU: AMG Ryzen 7 2700X 8-core processor
- RAM: 16 GB

### Usage Examples
#### Generating Synthetic Images
``` python
from src.synthetic_generation import SyntheticGenerator

generator = SyntheticGenerator(background_dir='data/backgrounds',
                               vessel_dir='data/vessels')
synthetic_images = generator.generate_batch(num_images=100)
```
#### Running Vessel Detection
``` python
from models.yolov8

-n.optimized_vessel_detection
import VesselDetector

detector = VesselDetector(model_path='models/yolov8-n/best.pt')
results = detector.detect('path/to/image.jpg')
```
## 📊 Results
The pipeline has demonstrated significant improvements in vessel detection performance, particularly in challenging scenarios with:
- Adverse weather conditions including cloud coverage
- Complex backgrounds with boat clusters
- Robust performance under varying lighting conditions

For detailed performance metrics and visualizations, see the evaluation notebooks in the repository.

## 🔍 Future Work
Ongoing development is focused on:
- Expanding the diversity of synthetic vessel types
- Implementing more sophisticated diffusion models
- Integrating API and geospatial input data
- Improving computational efficiency for large-scale dataset generation

## 📝 Citation
If you use this code in your research, please cite:
``` 
@misc{synthetic-satellite-pipeline,
  author = {Beer, Elano},
  title = {Synthetic Optical Satellite Imagery Generation Pipeline},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/ElanoBeer/synthetic-satellite-pipeline}
}
```
