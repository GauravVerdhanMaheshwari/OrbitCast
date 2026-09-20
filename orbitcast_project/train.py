import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model_def import HybridOrbitCastNet, MultiModalINSATDataset
from pathlib import Path
import gc

def train_hybrid_orbitcast(epochs=15, batch_size=2, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training Hybrid OrbitCast-XAI Engine on: {device} ---")
    
    dataset = MultiModalINSATDataset(seq_len=4, pred_len=4)
    if len(dataset) == 0:
        print("[ERROR] No valid sequence patches found in processed_patches/.")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = HybridOrbitCastNet(in_channels=4, vector_dim=6, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for (x_img, x_vec), y_img in loader:
            x_img, x_vec, y_img = x_img.to(device), x_vec.to(device), y_img.to(device)
            
            optimizer.zero_grad()
            
            # Autoregressive Prediction Loop
            current_img_seq = x_img.clone()
            loss = 0.0
            
            for t in range(4):
                pred_step = model(current_img_seq, x_vec)
                loss += criterion(pred_step, y_img[:, t:t+1])
                # Roll window forward
                current_img_seq = torch.cat([current_img_seq[:, 1:], pred_step], dim=1)
                
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - MSE Loss: {avg_loss:.6f}")

    # Save trained weights
    save_path = Path("./orbitcast_project/models/hybrid_orbitcast.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n[SUCCESS] Model checkpoint saved to: {save_path.resolve()}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    train_hybrid_orbitcast(epochs=15)