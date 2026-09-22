from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# --- Multi-Modal PyTorch Dataset Loader ---
class MultiModalINSATDataset(Dataset):

  def __init__(self, patches_dir="./processed_patches", seq_len=4, pred_len=4):
    self.patches_dir = Path(patches_dir)
    self.seq_len = seq_len
    self.pred_len = pred_len
    self.patch_files = sorted(list(self.patches_dir.glob("patch_*.npz")))
    self.total_samples = len(self.patch_files) - (seq_len + pred_len) + 1

  def __len__(self):
    return max(0, self.total_samples)

  def __getitem__(self, idx):
    # Historical inputs (T-3 to T_0)
    x_files = self.patch_files[idx : idx + self.seq_len]
    x_imgs = [np.load(f)["image"] for f in x_files]
    x_vecs = [np.load(f)["vector"] for f in x_files]

    # Target future images (T+1 to T+4)
    y_files = self.patch_files[
        idx + self.seq_len : idx + self.seq_len + self.pred_len
    ]
    y_imgs = [np.load(f)["image"] for f in y_files]

    x_img_tensor = torch.from_numpy(
        np.stack(x_imgs, axis=0)
    )  # Shape: (Seq, 4, H, W)
    x_vec_tensor = torch.from_numpy(np.stack(x_vecs, axis=0))  # Shape: (Seq, 6)
    y_img_tensor = torch.from_numpy(
        np.stack(y_imgs, axis=0)
    )  # Shape: (Pred, 4, H, W)

    return (x_img_tensor, x_vec_tensor), y_img_tensor


# --- ConvLSTM Cell ---
class ConvLSTMCell(nn.Module):

  def __init__(self, in_channels, out_channels, kernel_size=3):
    super(ConvLSTMCell, self).__init__()
    padding = kernel_size // 2
    self.out_channels = out_channels
    self.conv = nn.Conv2d(
        in_channels=in_channels + out_channels,
        out_channels=4 * out_channels,
        kernel_size=kernel_size,
        padding=padding,
    )

  def forward(self, x, h, c):
    combined = torch.cat([x, h], dim=1)
    gates = self.conv(combined)
    i, f, o, g = torch.split(gates, self.out_channels, dim=1)
    i, f, o, g = (
        torch.sigmoid(i),
        torch.sigmoid(f),
        torch.sigmoid(o),
        torch.tanh(g),
    )
    c_next = f * c + i * g
    h_next = o * torch.tanh(c_next)
    return h_next, c_next


# --- Hybrid Dual-Branch Neural Network with Residual Connections ---
class HybridOrbitCastNet(nn.Module):

  def __init__(self, in_channels=4, vector_dim=6, hidden_dim=64):
    super(HybridOrbitCastNet, self).__init__()
    self.hidden_dim = hidden_dim

    # Branch 1: Spatial ConvLSTM Encoder
    self.spatial_cell = ConvLSTMCell(
        in_channels=in_channels, out_channels=hidden_dim
    )

    # Branch 2: Dense MLP Feature Vector Encoder
    self.vector_mlp = nn.Sequential(
        nn.Linear(vector_dim, 32),
        nn.ReLU(),
        nn.Linear(32, 64),
        nn.ReLU(),
    )

    # Fusion Convolution (Merges spatial and tabular embeddings)
    self.fusion_conv = nn.Conv2d(hidden_dim + 64, hidden_dim, kernel_size=3, padding=1)

    # Decoder Head (Predicts frame movement/delta)
    self.decoder = nn.Sequential(
        nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(hidden_dim // 2, in_channels, kernel_size=3, padding=1),
    )

  def forward(self, x_img, x_vec):
    # Standardize input dimensions
    if x_img.ndim == 4:
      x_img = x_img.unsqueeze(1)  # Expand (B, C, H, W) -> (B, 1, C, H, W)
    if x_vec.ndim == 2:
      x_vec = x_vec.unsqueeze(1)  # Expand (B, V) -> (B, 1, V)

    b, seq, c, h, w = x_img.shape
    device = x_img.device

    h_state = torch.zeros(b, self.hidden_dim, h, w, device=device)
    c_state = torch.zeros(b, self.hidden_dim, h, w, device=device)

    # Pass historical image sequence through ConvLSTM
    for t in range(seq):
      h_state, c_state = self.spatial_cell(x_img[:, t], h_state, c_state)

    # Encode vector data from last frame
    v_emb = self.vector_mlp(x_vec[:, -1])  # (B, 64)
    v_emb_spatial = v_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)  # (B, 64, H, W)

    # Latent Feature Fusion
    fused = torch.cat([h_state, v_emb_spatial], dim=1)
    latent = torch.relu(self.fusion_conv(fused))

    # Predict Delta (Residual Update)
    delta = self.decoder(latent)

    # Residual Connection: Add predicted delta to the most recent input frame (t_0)
    last_input_frame = x_img[:, -1]  # (B, 4, H, W)
    out_frame = torch.sigmoid(last_input_frame + delta)

    return out_frame  # Returns tensor shaped (B, 4, H, W)