from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from model_def import HybridOrbitCastNet

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="OrbitCast - Patch Comparison",
    page_icon="🌀",
    layout="wide",
)
st.title("🌀 OrbitCast: 3-Way Patch Comparison")

# -----------------------------------------------------------------------------
# 2. Model Loader
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 3. Data Selection
# -----------------------------------------------------------------------------
patch_dir = Path("./processed_patches")
patch_files = sorted(list(patch_dir.glob("*.npz")))

if not patch_files:
  st.error("No `.npz` files found in `./processed_patches/`!")
  st.stop()

st.sidebar.header("Controls")
selected_patch_path = st.sidebar.selectbox(
    "Select Input Patch (t_0)", patch_files, format_func=lambda p: p.name
)

# Render Style Choice
colormap = st.sidebar.selectbox(
    "Visualization Mode",
    options=[
        "Natural Cloud (Gray)",
        "Thermal IR (Gist Heat)",
        "Water Vapor (Blues)",
    ],
)

cmap_name = "gray"
if "Heat" in colormap:
  cmap_name = "gist_heat"
elif "Blues" in colormap:
  cmap_name = "Blues"

channel_idx = 1 if "Blues" in colormap else 0

# -----------------------------------------------------------------------------
# 4. Load Data & Next Frame
# -----------------------------------------------------------------------------
curr_data = np.load(selected_patch_path)

match = re.search(r"(\d+)", selected_patch_path.name)
next_patch_path = None

if match:
  curr_idx = int(match.group(1))
  next_filename = f"patch_{curr_idx + 1:04d}.npz"
  possible_next = patch_dir / next_filename
  if possible_next.exists():
    next_patch_path = possible_next

# Helper function to extract 2D spatial channel slice without destructive min-max scaling
def extract_slice(data_array, ch_idx=0):
  arr = data_array.squeeze()
  if arr.ndim == 4:
    arr = arr[-1, ch_idx]
  elif arr.ndim == 3:
    if arr.shape[0] > ch_idx:
      arr = arr[ch_idx]
    else:
      arr = arr[0]
  return arr


# -----------------------------------------------------------------------------
# 5. Model Inference & Visual Processing
# -----------------------------------------------------------------------------
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
  vec_padded[..., : min(6, vec_tensor.shape[-1])] = vec_tensor[
      ..., : min(6, vec_tensor.shape[-1])
  ]
  vec_tensor = vec_padded

with torch.no_grad():
  pred_tensor = model(img_tensor.to(device), vec_tensor.to(device))
  pred_np = pred_tensor.cpu().numpy()


# Robust display-scaling function
def process_slice_for_display(data_array, ch_idx=0):
  arr = data_array.squeeze()
  if arr.ndim == 4:
    arr = arr[-1, ch_idx]
  elif arr.ndim == 3:
    if arr.shape[0] > ch_idx:
      arr = arr[ch_idx]
    else:
      arr = arr[0]

  # Percentile scaling (1st to 99th) stretches dynamic contrast without blowouts
  p_low, p_high = np.percentile(arr, (1, 99))
  if p_high - p_low > 1e-6:
    norm_arr = np.clip((arr - p_low) / (p_high - p_low), 0.0, 1.0)
  else:
    norm_arr = arr
  return norm_arr


# Process each panel independently for display
input_slice = process_slice_for_display(img_raw, channel_idx)
pred_slice = process_slice_for_display(pred_np, channel_idx)

gt_slice = None
if next_patch_path:
  next_data = np.load(next_patch_path)
  gt_slice = process_slice_for_display(next_data["image"], channel_idx)

# -----------------------------------------------------------------------------
# 6. Side-by-Side 3-Column Display
# -----------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
  st.markdown(f"### 1. Input Patch (`{selected_patch_path.name}`)")
  fig1, ax1 = plt.subplots(figsize=(4, 4))
  ax1.imshow(input_slice, cmap=cmap_name)
  ax1.axis("off")
  st.pyplot(fig1)

with col2:
  st.markdown("### 2. Predicted ($t+1$)")
  fig2, ax2 = plt.subplots(figsize=(4, 4))
  ax2.imshow(pred_slice, cmap=cmap_name)
  ax2.axis("off")
  st.pyplot(fig2)

with col3:
  if gt_slice is not None:
    st.markdown(f"### 3. Actual Next Patch (`{next_patch_path.name}`)")
    fig3, ax3 = plt.subplots(figsize=(4, 4))
    ax3.imshow(gt_slice, cmap=cmap_name)
    ax3.axis("off")
    st.pyplot(fig3)
  else:
    st.markdown("### 3. Actual Next Patch")
    st.warning(
        f"No matching file (`patch_{curr_idx + 1:04d}.npz`) found in"
        " directory."
    )