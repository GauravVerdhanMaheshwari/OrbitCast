from pathlib import Path
from model_def import HybridOrbitCastNet, MultiModalINSATDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class CombinedEdgeLoss(nn.Module):

  def __init__(self, l1_weight=0.8, grad_weight=0.2):
    super(CombinedEdgeLoss, self).__init__()
    self.l1 = nn.L1Loss()

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
    self.grad_weight = grad_weight

  def forward(self, pred, target):
    l1_loss = self.l1(pred, target)

    b, c, h, w = pred.shape
    kx = self.kernel_x.repeat(c, 1, 1, 1).to(pred.device)
    ky = self.kernel_y.repeat(c, 1, 1, 1).to(pred.device)

    pred_grad_x = F.conv2d(pred, kx, padding=1, groups=c)
    pred_grad_y = F.conv2d(pred, ky, padding=1, groups=c)
    target_grad_x = F.conv2d(target, kx, padding=1, groups=c)
    target_grad_y = F.conv2d(target, ky, padding=1, groups=c)

    grad_loss = self.l1(pred_grad_x, target_grad_x) + self.l1(
        pred_grad_y, target_grad_y
    )

    return l1_loss + (self.grad_weight * grad_loss)


def train_model():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Training on device: {device}")

  if torch.cuda.is_available():
    torch.cuda.empty_cache()

  dataset = MultiModalINSATDataset(
      patches_dir="./processed_patches", seq_len=4, pred_len=4
  )
  if len(dataset) == 0:
    print("No patch samples found in ./processed_patches!")
    return

  dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
  model = HybridOrbitCastNet(in_channels=4, vector_dim=6, hidden_dim=64).to(
      device
  )

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = CombinedEdgeLoss()
  scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

  models_dir = Path("./models")
  models_dir.mkdir(exist_ok=True)

  epochs = 10
  model.train()

  for epoch in range(epochs):
    running_loss = 0.0
    for (x_img, x_vec), y_img in dataloader:
      x_img = x_img.float().to(device)
      x_vec = x_vec.float().to(device)
      y_img = y_img.float().to(device)

      optimizer.zero_grad()
      curr_seq = x_img
      total_loss = 0.0

      if torch.cuda.is_available():
        with torch.amp.autocast("cuda"):
          for step in range(y_img.shape[1]):
            target_frame = y_img[:, step]
            pred_frame = model(curr_seq, x_vec)

            loss = criterion(pred_frame, target_frame)
            total_loss = total_loss + loss

            next_frame_input = pred_frame.detach().unsqueeze(1)
            curr_seq = torch.cat([curr_seq[:, 1:], next_frame_input], dim=1)

          total_loss = total_loss / y_img.shape[1]

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
      else:
        for step in range(y_img.shape[1]):
          target_frame = y_img[:, step]
          pred_frame = model(curr_seq, x_vec)

          loss = criterion(pred_frame, target_frame)
          total_loss = total_loss + loss

          next_frame_input = pred_frame.detach().unsqueeze(1)
          curr_seq = torch.cat([curr_seq[:, 1:], next_frame_input], dim=1)

        total_loss = total_loss / y_img.shape[1]
        total_loss.backward()
        optimizer.step()

      running_loss += total_loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.5f}")

  torch.save(model.state_dict(), models_dir / "hybrid_orbitcast.pth")
  print("Model saved to ./models/hybrid_orbitcast.pth")


if __name__ == "__main__":
  train_model()