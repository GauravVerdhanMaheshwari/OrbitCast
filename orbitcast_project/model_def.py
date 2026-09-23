from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class MultiModalINSATDataset(Dataset):

  def __init__(
      self, patches_dir="./processed_patches", seq_len=4, pred_len=4
  ):
    self.patches_dir = Path(patches_dir)
    self.seq_len = seq_len
    self.pred_len = pred_len
    self.patch_files = sorted(list(self.patches_dir.glob("patch_*.npz")))
    self.total_samples = len(self.patch_files) - (seq_len + pred_len) + 1

  def __len__(self):
    return max(0, self.total_samples)

  def __getitem__(self, idx):
    x_files = self.patch_files[idx : idx + self.seq_len]
    x_imgs = [np.load(f)["image"] for f in x_files]
    x_vecs = [np.load(f)["vector"] for f in x_files]

    y_files = self.patch_files[
        idx + self.seq_len : idx + self.seq_len + self.pred_len
    ]
    y_imgs = [np.load(f)["image"] for f in y_files]

    x_img_tensor = torch.from_numpy(np.stack(x_imgs, axis=0))
    x_vec_tensor = torch.from_numpy(np.stack(x_vecs, axis=0))
    y_img_tensor = torch.from_numpy(np.stack(y_imgs, axis=0))

    return (x_img_tensor, x_vec_tensor), y_img_tensor


class HybridOrbitCastNet(nn.Module):

  def __init__(self, in_channels=4, vector_dim=6, hidden_dim=64):
    super(HybridOrbitCastNet, self).__init__()
    self.hidden_dim = hidden_dim

    # Feature Extractor (Spatial Backbone)
    self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

    # Auxiliary Meteorological Vector Integration
    self.vector_mlp = nn.Sequential(
        nn.Linear(vector_dim, 32), nn.ReLU(), nn.Linear(32, hidden_dim)
    )

    # Spatial-Temporal Decoder
    self.decoder = nn.Sequential(
        nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(hidden_dim // 2, in_channels, kernel_size=3, padding=1),
    )

  def forward(self, x_img, x_vec):
    if x_img.ndim == 5:
      x_img = x_img[:, -1]  # Take latest sequence frame
    if x_vec.ndim == 3:
      x_vec = x_vec[:, -1]

    feat = F.relu(self.conv1(x_img))
    feat = F.relu(self.conv2(feat))

    # Vector feature conditioning
    vec_feat = self.vector_mlp(x_vec).unsqueeze(-1).unsqueeze(-1)
    feat = feat + vec_feat

    pred_frame = torch.clamp(x_img + 0.3 * self.decoder(feat), 0.0, 1.0)
    return pred_frame