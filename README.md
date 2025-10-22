# ⚙️ VigorVision

![PyPI](https://img.shields.io/pypi/v/vigorvision?color=brightgreen)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Torch](https://img.shields.io/badge/pytorch-2.0%2B-red)
![License](https://img.shields.io/badge/license-MIT-yellow)
![Build](https://img.shields.io/github/actions/workflow/status/VigorIndustries/vigorvision/tests.yml?branch=main)

> **VigorVision** — an advanced, modular AI vision framework built by **Vigor Industries Limited**, designed for **object detection**, **segmentation**, and **industry-specific visual intelligence**.

---

## 🧭 Table of Contents

1. [Overview](#-overview)
2. [Features](#-features)
3. [Installation](#-installation)
4. [Quick Start](#-quick-start)
5. [Model Architecture](#-model-architecture)
6. [Configuration](#-configuration)
7. [Training](#-training)
8. [Visualization & Logging](#-visualization--logging)
9. [Project Structure](#-project-structure)
10. [Contributing](#-contributing)
11. [License](#-license)
12. [Contact](#-contact)

---

## 🧠 Overview

**VigorVision** provides a fully-featured, modular AI framework for vision applications such as:

- Industrial anomaly detection  
- Object tracking and classification  
- Manufacturing process vision systems  
- Smart quality control and inspection  

It leverages **PyTorch**, **Albumentations**, and **Weights & Biases** for maximum performance, extensibility, and interpretability.

---

## ✨ Features

| Category | Highlights |
|-----------|-------------|
| 🧩 **Architecture** | Modular design with customizable backbone, neck, and detection head |
| 🧠 **Training Engine** | Optimized multi-threaded dataloaders with adaptive batch handling |
| 🧮 **Anchors** | Auto-anchor analysis and smart anchor selection for detection models |
| 📈 **Visualization** | Native support for **TensorBoard** and **Weights & Biases (wandb)** |
| 🔧 **Data Augmentation** | Albumentations-based pipeline for advanced transformations |
| ⚡ **Performance** | FP16 (mixed precision) and multi-GPU training support |
| 🧰 **Utilities** | Built-in tools for dataset validation and metrics computation |

---

## ⚙️ Installation

### ✅ From PyPI
```bash
pip install vigorvision
pip install torch>=2.0.0
pip install torchvision>=0.15.0
pip install albumentations>=1.3.0
pip install opencv-python>=4.8.0
pip install numpy>=1.24.0
pip install tqdm>=4.65.0
pip install pyyaml>=6.0
pip install tensorboard>=2.13.0
pip install wandb>=0.16.0
pip install scikit-learn>=1.3.0
pip install scipy>=1.11.0
pip install matplotlib>=3.8.0
```
