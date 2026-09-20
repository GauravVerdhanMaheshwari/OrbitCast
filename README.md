# 🚀 OrbitCast

> **An end-to-end satellite imagery processing and deep learning pipeline for cyclone tracking using INSAT-3D data.**

![Python Version](https://shields.io)
![Framework](https://shields.io)
![UI](https://shields.io)
![License](https://shields.io)

## 📌 Overview

OrbitCast is a complete machine learning workflow designed to ingest, process, and visualize **INSAT-3D meteorological satellite data**. The system extracts **4-channel spatial patches** from raw satellite `.h5` files, formats them for PyTorch deep learning models to track or predict cyclonic activity, and serves the predictions via an interactive Streamlit web dashboard.

## ✨ Key Features

- **Raw HDF5 Ingestion:** Efficiently parses heavy `.h5` scientific data structures directly from INSAT-3D downloads.
- **4-Channel Patch Generation:** Preprocesses and exports multi-spectral data into standardized, model-ready `.npz` patches.
- **PyTorch Integration:** Includes a custom dataset pipeline natively built for training spatial-temporal or convolutional models.
- **Interactive Dashboard:** Features a Streamlit-powered UI to visualize processed satellite frames and monitor model predictions.

## 🛠️ Tech Stack

- **Deep Learning Framework:** PyTorch, TorchVision
- **Data Processing:** H5Py, NumPy, SciPy, OpenCV (cv2)
- **Visualization & UI:** Streamlit, Matplotlib

---

## 📂 Project Structure

```text
orbitcast_project/
├── raw_data/               # Downloaded raw INSAT-3D .h5 files (grouped by event)
│   ├── cyclone_01/
│   └── cyclone_02/
├── processed_patches/      # Preprocessed & exported 4-channel .npz patches
├── models/                 # Saved PyTorch model checkpoints (.pth)
├── dataset.py              # Custom PyTorch Dataset module for training
├── export_patches.py       # Data preprocessing and patch extraction script
├── train.py                # Deep learning model training pipeline
└── app.py                  # Streamlit dashboard interface
```

---

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have Python 3.8 or higher installed on your system.

### 2. Installation

Clone this repository and install the required dependencies using `pip`:

```bash
# Clone the repository
git clone https://github.com
cd orbitcast

# Install core dependencies
pip install torch torchvision numpy scipy h5py streamlit matplotlib opencv-python
```

### 3. Workflow Execution

#### Step 1: Preprocess the Data

Extract 4-channel image patches from your raw INSAT-3D `.h5` files:

```bash
python export_patches.py
```

#### Step 2: Train the Model

Run the PyTorch training pipeline using the preprocessed dataset:

```bash
python train.py
```

#### Step 3: Launch the Dashboard

Run the interactive Streamlit application to visualize results:

```bash
streamlit run app.py
```

---

## 💡 Code Architecture Quick View

### Data Loading (`dataset.py`)

The pipeline uses a custom PyTorch dataset to load the `.npz` files efficiently on the fly during training:

```python
import numpy as np
from torch.utils.data import Dataset

class OrbitCastDataset(Dataset):
    def __init__(self, patch_dir):
        # Initialises file paths for 4-channel patches
        pass
```
