import glob
from pathlib import Path
import h5py
import numpy as np
from scipy.ndimage import zoom

RAW_DIR = Path("./raw_data")
OUTPUT_DIR = Path("./processed_patches")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def process_multimodal_dataset():
  h5_files = sorted(glob.glob(str(RAW_DIR / "**/*.h5"), recursive=True))
  print(f"Found {len(h5_files)} raw MOSDAC H5 files.")

  # 1. Detect existing patches to prevent overwriting
  existing_patches = list(OUTPUT_DIR.glob("patch_*.npz"))
  if existing_patches:
    existing_indices = [
        int(p.stem.split("_")[1])
        for p in existing_patches
        if p.stem.split("_")[1].isdigit()
    ]
    patch_count = max(existing_indices) + 1
  else:
    patch_count = 0

  print(f"Starting patch export at index: {patch_count:04d}\n")

  for file_path in h5_files:
    try:
      with h5py.File(file_path, "r") as f:
        # 2. Read Raw Datasets (handling 2D vs 3D shapes)
        tir1 = np.squeeze(
            f["IMG_TIR1"][()] if "IMG_TIR1" in f else f["TIR1"][()]
        )
        wv = np.squeeze(f["IMG_WV"][()] if "IMG_WV" in f else f["WV"][()])
        mir = np.squeeze(f["IMG_MIR"][()] if "IMG_MIR" in f else f["MIR"][()])

        # 3. Match Resolutions (WV/MIR to TIR1)
        if wv.shape != tir1.shape:
          zoom_factors = (
              tir1.shape[0] / wv.shape[0],
              tir1.shape[1] / wv.shape[1],
          )
          wv = zoom(wv, zoom_factors, order=1)

        if mir.shape != tir1.shape:
          zoom_factors = (
              tir1.shape[0] / mir.shape[0],
              tir1.shape[1] / mir.shape[1],
          )
          mir = zoom(mir, zoom_factors, order=1)

        # 4. Normalize Channels to [0, 1]
        tir1_n = (tir1 - np.nanmin(tir1)) / (
            np.nanmax(tir1) - np.nanmin(tir1) + 1e-8
        )
        wv_n = (wv - np.nanmin(wv)) / (np.nanmax(wv) - np.nanmin(wv) + 1e-8)
        mir_n = (mir - np.nanmin(mir)) / (np.nanmax(mir) - np.nanmin(mir) + 1e-8)

        # 5. Physical Channel Derivation (Delta Water Vapor: TIR1 - WV)
        dwv_n = np.clip(tir1_n - wv_n, 0, 1)

        # 6. Stack into 4-Channel Spatial Tensor (C, H, W)
        stacked_img = np.stack([tir1_n, wv_n, mir_n, dwv_n], axis=0).astype(
            np.float32
        )

        # 7. Crop Bounding Box Region
        if stacked_img.shape[1] >= 400 and stacked_img.shape[2] >= 400:
          stacked_img = stacked_img[:, 800:1200, 600:1000]

        # 8. Extract Tabular Feature Vector (6-D)
        eye_y, eye_x = np.unravel_index(
            np.argmin(stacked_img[0]), stacked_img[0].shape
        )
        mean_temp = float(np.mean(stacked_img[0]))
        max_convection = float(np.max(stacked_img[3]))
        temp_std = float(np.std(stacked_img[0]))

        vector_feat = np.array(
            [
                eye_y / 400.0,
                eye_x / 400.0,
                mean_temp,
                max_convection,
                temp_std,
                1.0,
            ],
            dtype=np.float32,
        )

        # 9. Export Multi-Modal NPZ Patch
        patch_file = OUTPUT_DIR / f"patch_{patch_count:04d}.npz"
        np.savez_compressed(patch_file, image=stacked_img, vector=vector_feat)
        print(
            f"[{patch_count:04d}] Saved: {patch_file.name} (from"
            f" {Path(file_path).name})"
        )
        patch_count += 1

    except Exception as e:
      print(f"Skipping corrupted file {Path(file_path).name}: {e}")

  total_patches = len(list(OUTPUT_DIR.glob("patch_*.npz")))
  print(
      f"\n[SUCCESS] Preprocessing finished. Total patches now available in"
      f" '{OUTPUT_DIR}': {total_patches}"
  )


if __name__ == "__main__":
  process_multimodal_dataset()