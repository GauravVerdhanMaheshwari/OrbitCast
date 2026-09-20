orbitcast_project/
├── raw_data/ # Downloaded INSAT-3D .h5 files
│ ├── cyclone_01/
│ └── cyclone_02/
├── processed_patches/ # Exported 4-channel .npz patches
├── models/ # Saved PyTorch checkpoints
├── dataset.py # PyTorch Dataset module
├── export_patches.py # Data preprocessing script
├── train.py # Training pipeline script
└── app.py # Streamlit Dashboard UI
