# OrbitCast-XAI: Spatio-Temporal Cyclone Nowcasting & Explainable AI Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://orbitcast-xai.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Live Interactive Dashboard:** [orbitcast-xai.streamlit.app](https://orbitcast-xai.streamlit.app/)

OrbitCast-XAI is an end-to-end deep learning framework and operational dashboard designed for **real-time satellite-based cyclone nowcasting, intensity stage classification, and explainable AI (XAI) feature activation**.

By processing multi-spectral INSAT-3D satellite imagery paired with NOAA IBTrACS best-track data, OrbitCast-XAI forecasts future storm frames ($T+1$) and provides transparent, automated decision-support narratives aligned with **India Meteorological Department (IMD)** operational standards.

---

## Key Features

- **Spatio-Temporal Sequence Forecasting:** Utilizes `HybridOrbitCastNet` (CNN + ConvLSTM) to ingest multi-channel satellite frame sequences ($T=0$) and forecast short-term storm evolution ($T+1$).
- **IMD Stage Classification & Operational Metrics:** Automatically estimates sustained wind speeds (kts) and categorizes tropical cyclones into official IMD intensity stages (LPA, DD, CS, VSCS, SuCS).
- **Explainable AI (Grad-CAM & SHAP):** Generates gradient-weighted class activation heatmaps to highlight cloud-top convective density and eyewall structural focus areas.
- **Automated Text Narratives:** Translates complex neural network feature activations into human-readable risk narratives for meteorological decision-makers.
- **Interactive Streamlit UI:** A clean, cloud-hosted dashboard supporting pre-loaded `.npz` INSAT-3D patches or user-uploaded satellite files.

---

## Tech Stack & Architecture

- **Deep Learning Engine:** PyTorch, Torchvision, OpenCV, Albumentations
- **Data Processing & Pipelines:** Python, NumPy, Pandas, xarray, h5py, netCDF4
- **Baselines & ML Benchmark:** scikit-learn (Random Forest, XGBoost), Persistence & Linear Extrapolation Tracks
- **Explainability (XAI):** Grad-CAM, SHAP, Custom Natural Language Narrative Generator
- **Frontend & Deployment:** Streamlit Cloud, Git LFS (Large File Storage for `.pth` model checkpoints)

---

## Pipeline Workflow

```
┌────────────────────────┐    ┌────────────────────────┐    ┌────────────────────────┐
│  INSAT-3D Satellite    │───>│ Geospatial Alignment   │───>│ HybridOrbitCastNet     │
│  (MOSDAC HDF5/netCDF)  │    │  & IBTrACS Pairing     │    │  (CNN + ConvLSTM)      │
└────────────────────────┘    └────────────────────────┘    └────────────────────────┘
                                                                        │
                                                                        ▼
┌────────────────────────┐    ┌────────────────────────┐    ┌────────────────────────┐
│ Interactive Streamlit  │<───│  Grad-CAM Activation   │<───│  Metrics, Intensity,   │
│ Dashboard Deployment   │    │  Heatmaps & XAI Text   │    │  & Spatio-temporal T+1 │
└────────────────────────┘    └────────────────────────┘    └────────────────────────┘
```

---

## Dataset & Validation Strategy

- **Data Sources:**
  - [MOSDAC (ISRO)](https://www.mosdac.gov.in) — INSAT-3D / 3DR meteorological satellite archives.
  - [NOAA IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive) — Historical North Indian Ocean cyclone tracks and intensity records.
  - [India Meteorological Department (IMD)](https://mausam.imd.gov.in) — RSMC cyclone stage classification standards.
- **Validation Methodology:** Evaluated strictly using **Leave-One-Cyclone-Out (LOCO) cross-validation** (e.g., train on Cyclones A, B, C; evaluate on unseen Cyclone D) to prevent data leakage and guarantee real-world generalization across unseen storms.

---

## Quickstart & Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/GauravVerdhanMaheshwari/OrbitCast.git
cd OrbitCast
```

### 2. Set Up Virtual Environment & Dependencies

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Fetch Model Weights via Git LFS

```bash
git lfs install
git lfs pull
```

### 4. Run the Streamlit Dashboard Locally

```bash
streamlit run app.py
```

---

## Project Directory Structure

```text
OrbitCast/
├── orbitcast_project/
│   ├── models/
│   │   └── hybrid_orbitcast.pth      # Model weights (tracked via Git LFS)
│   ├── processed_patches/
│   │   └── patch_0040.npz            # Pre-loaded sample INSAT-3D patches
│   ├── src/
│   │   ├── model.py                  # PyTorch HybridOrbitCastNet definition
│   │   ├── xai.py                    # Grad-CAM and narrative engine
│   │   └── utils.py                  # Data loading and satellite utilities
├── app.py                            # Main Streamlit dashboard script
├── .gitattributes                    # Git LFS configuration
├── requirements.txt                  # Python dependencies
└── README.md                         # Documentation
```

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
