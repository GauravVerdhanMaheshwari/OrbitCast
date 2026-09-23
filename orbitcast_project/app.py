from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from model_def import HybridOrbitCastNet

# 1. Page Config
st.set_page_config(
    page_title="OrbitCast-XAI Portal",
    page_icon="🌀",
    layout="wide",
)

st.title("🌀 OrbitCast-XAI: Satellite Nowcasting & Motion Analysis")

# 2. Model Loader
@st.cache_resource
def load_model():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = HybridOrbitCastNet(in_channels=4, vector_dim=6, hidden_dim=64).to(
      device
  )
  model_path = Path("./models/hybrid_orbitcast.pth")
  if model_path.exists():
    model.load_state_dict(torch.load(model_path, map_location=device))
  model.eval()
  return model, device

model, device = load_model()

# 3. Sidebar Controls & File Upload
st.sidebar.header("🕹️ Controls & Data Input")

uploaded_file = st.sidebar.file_uploader("Upload custom .npz patch", type=["npz"])

patch_dir = Path("./processed_patches")
patch_files = sorted(list(patch_dir.glob("*.npz")))

# Forecast Step Slider (t + 1 to t + 4)
forecast_step = st.sidebar.slider(
    "Forecast Time Horizon (t + N)", min_value=1, max_value=4, value=1
)

target_gt_path = None

if uploaded_file is not None:
  curr_data = np.load(uploaded_file)
  selected_name = uploaded_file.name
elif patch_files:
  selected_patch_path = st.sidebar.selectbox(
      "Select Dataset Patch (t_0)", patch_files, format_func=lambda p: p.name
  )
  curr_data = np.load(selected_patch_path)
  selected_name = selected_patch_path.name

  # Search for Ground Truth matching t + forecast_step
  match = re.search(r"(\d+)", selected_name)
  if match:
    curr_idx = int(match.group(1))
    target_idx = curr_idx + forecast_step
    possible_gt = patch_dir / f"patch_{target_idx:04d}.npz"
    if possible_gt.exists():
      target_gt_path = possible_gt
else:
  st.error("No `.npz` files found! Upload a patch or run preprocessing.")
  st.stop()

colormap = st.sidebar.selectbox(
    "Visualization Mode",
    ["Natural Cloud (Gray)", "Thermal IR (Gist Heat)", "Water Vapor (Blues)"]
)

cmap_name = "gray"
if "Heat" in colormap:
  cmap_name = "gist_heat"
elif "Blues" in colormap:
  cmap_name = "Blues"

channel_idx = 1 if "Blues" in colormap else 0

# 4. Input Preprocessing
img_raw = curr_data["image"]
vec_raw = curr_data["vector"]

img_tensor = torch.tensor(img_raw, dtype=torch.float32)
vec_tensor = torch.tensor(vec_raw, dtype=torch.float32)

while img_tensor.ndim < 5:
  img_tensor = img_tensor.unsqueeze(0)

if vec_tensor.ndim == 1:
  vec_tensor = vec_tensor.unsqueeze(0).unsqueeze(0)
elif vec_tensor.ndim == 2:
  vec_tensor = vec_tensor.unsqueeze(0)

if vec_tensor.shape[-1] != 6:
  vec_padded = torch.zeros((vec_tensor.shape[0], vec_tensor.shape[1], 6))
  vec_padded[..., : min(6, vec_tensor.shape[-1])] = vec_tensor[..., : min(6, vec_tensor.shape[-1])]
  vec_tensor = vec_padded

# Autoregressive Multi-Step Inferences
curr_img_seq = img_tensor.to(device)
v_input = vec_tensor.to(device)

pred_frames = []
with torch.no_grad():
  for step in range(forecast_step):
    pred_step = model(curr_img_seq, v_input)
    if pred_step.ndim == 4:
      next_frame = pred_step.unsqueeze(1)
    else:
      next_frame = pred_step
    
    pred_frames.append(next_frame.squeeze(1))
    # Slide the input window forward with new prediction
    curr_img_seq = torch.cat([curr_img_seq[:, 1:], next_frame], dim=1)

# Extract frame corresponding to current slider position
pred_tensor = pred_frames[-1]
pred_np = pred_tensor.cpu().numpy()

def process_slice(data_array, ch_idx=0):
  arr = data_array.squeeze()
  if arr.ndim == 4:
    arr = arr[-1, ch_idx]
  elif arr.ndim == 3:
    arr = arr[ch_idx] if arr.shape[0] > ch_idx else arr[0]
  p_low, p_high = np.percentile(arr, (1, 99))
  if p_high - p_low > 1e-6:
    return np.clip((arr - p_low) / (p_high - p_low), 0.0, 1.0)
  return arr

input_slice = process_slice(img_raw, channel_idx)
pred_slice = process_slice(pred_np, channel_idx)

# Explainable Motion Heatmap (|Pred - Input|)
motion_heatmap = np.abs(pred_slice - input_slice)

# 5. Dashboard Layout
col1, col2, col3, col4 = st.columns(4)

with col1:
  st.markdown(f"**1. Input Patch ($t_0$)**\n`{selected_name}`")
  fig1, ax1 = plt.subplots(figsize=(3.5, 3.5))
  ax1.imshow(input_slice, cmap=cmap_name)
  ax1.axis("off")
  st.pyplot(fig1)

with col2:
  st.markdown(f"**2. Predicted ($t+{forecast_step}$)**")
  fig2, ax2 = plt.subplots(figsize=(3.5, 3.5))
  ax2.imshow(pred_slice, cmap=cmap_name)
  ax2.axis("off")
  st.pyplot(fig2)

with col3:
  st.markdown(f"**3. Motion Heatmap (XAI)**")
  fig3, ax3 = plt.subplots(figsize=(3.5, 3.5))
  ax3.imshow(motion_heatmap, cmap="magma")
  ax3.axis("off")
  st.pyplot(fig3)

with col4:
  if target_gt_path:
    st.markdown(f"**4. Ground Truth ($t+{forecast_step}$)**\n`{target_gt_path.name}`")
    gt_data = np.load(target_gt_path)
    gt_slice = process_slice(gt_data["image"], channel_idx)
    fig4, ax4 = plt.subplots(figsize=(3.5, 3.5))
    ax4.imshow(gt_slice, cmap=cmap_name)
    ax4.axis("off")
    st.pyplot(fig4)
  else:
    st.markdown(f"**4. Ground Truth ($t+{forecast_step}$)**")
    st.info(f"No patch file found for index t+{forecast_step}.")