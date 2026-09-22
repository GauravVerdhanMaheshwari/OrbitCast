import gc
from pathlib import Path
from model_def import HybridOrbitCastNet, MultiModalINSATDataset
from pytorch_msssim import SSIM
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# Edge/Gradient Loss to force sharp high-frequency details
class GradientLoss(nn.Module):

  def __init__(self):
    super().__init__()
    kernel_x = (
        torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    kernel_y = (
        torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    self.register_buffer("kernel_x", kernel_x)
    self.register_buffer("kernel_y", kernel_y)

  def forward(self, pred, target):
    # Expand kernels across 4 channels
    kx = self.kernel_x.repeat(pred.shape[1], 1, 1, 1)
    ky = self.kernel_y.repeat(pred.shape[1], 1, 1, 1)

    pred_grad_x = F.conv2d(pred, kx, padding=1, groups=pred.shape[1])
    pred_grad_y = F.conv2d(pred, ky, padding=1, groups=pred.shape[1])
    target_grad_x = F.conv2d(target, kx, padding=1, groups=target.shape[1])
    target_grad_y = F.conv2d(target, ky, padding=1, groups=target.shape[1])

    return F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(
        pred_grad_y, target_grad_y
    )


class SharpnessLoss(nn.Module):

  def __init__(self):
    super().__init__()
    self.l1 = nn.L1Loss()
    # win_size=3 captures fine micro-textures on cropped patches
    self.ssim = SSIM(data_range=1.0, channel=4, spatial_dims=2, win_size=3)
    self.grad = GradientLoss()

  def forward(self, pred, target):
    if pred.ndim == 5:
      b, s, c, h, w = pred.shape
      pred = pred.view(b * s, c, h, w)
      target = target.view(b * s, c, h, w)

    l1_loss = self.l1(pred, target)
    ssim_loss = 1.0 - self.ssim(pred, target)
    grad_loss = self.grad(pred, target)

    # Balanced weight blend
    return (0.3 * l1_loss) + (0.4 * ssim_loss) + (0.3 * grad_loss)


def train_hybrid_orbitcast(
    epochs=25, batch_size=4, lr=1e-3, forecast_steps=4
):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(
      f"--- Training Hybrid OrbitCast-XAI (Multi-Step t+{forecast_steps}) on:"
      f" {device} ---"
  )

  dataset = MultiModalINSATDataset(seq_len=1, pred_len=forecast_steps)
  if len(dataset) == 0:
    print("[ERROR] No valid patches found in ./processed_patches/")
    return

  loader = DataLoader(
      dataset, batch_size=batch_size, shuffle=True, drop_last=True
  )

  model = HybridOrbitCastNet(in_channels=4, vector_dim=6, hidden_dim=64).to(
      device
  )
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  criterion = SharpnessLoss().to(device)

  step_weights = [1.0, 0.8, 0.6, 0.4]

  model.train()
  for epoch in range(1, epochs + 1):
    running_loss = 0.0
    for (x_img, x_vec), y_img in loader:
      x_img, x_vec, y_img = (
          x_img.to(device),
          x_vec.to(device),
          y_img.to(device),
      )

      optimizer.zero_grad()
      current_img = x_img
      total_loss = 0.0

      for t in range(forecast_steps):
        pred_step = model(current_img, x_vec)
        target_step = y_img[:, t] if y_img.ndim == 5 else y_img

        step_loss = criterion(pred_step, target_step)
        total_loss += step_weights[t] * step_loss

        # Detach gradient on rollout to keep step-wise training fast
        current_img = pred_step.detach()

      total_loss.backward()
      optimizer.step()
      running_loss += total_loss.item()

    avg_loss = running_loss / len(loader)
    print(f"Epoch [{epoch:02d}/{epochs:02d}] - Multi-Step Loss: {avg_loss:.6f}")

  save_dir = Path("./models")
  save_dir.mkdir(exist_ok=True)
  save_path = save_dir / "hybrid_orbitcast.pth"
  torch.save(model.state_dict(), save_path)
  print(f"\n[SUCCESS] Model checkpoint saved to: {save_path.resolve()}")

  gc.collect()
  if torch.cuda.is_available():
    torch.cuda.empty_cache()


if __name__ == "__main__":
  train_hybrid_orbitcast(epochs=25, batch_size=4, lr=1e-3, forecast_steps=4)