import glob
import h5py
import numpy as np
from pathlib import Path

RAW_DIR = Path("./orbitcast_project/raw_data")
OUTPUT_DIR = Path("./orbitcast_project/processed_patches")

def process_multimodal_dataset():
    h5_files = sorted(glob.glob(str(RAW_DIR / "**/*.h5"), recursive=True))
    print(f"Found {len(h5_files)} raw MOSDAC H5 files.")
    
    patch_count = 0
    for file_path in h5_files:
        try:
            with h5py.File(file_path, 'r') as f:
                # 1. Extract Satellite Channels
                tir1 = f['IMG_TIR1'][()] if 'IMG_TIR1' in f else f['TIR1'][()]
                wv   = f['IMG_WV'][()] if 'IMG_WV' in f else f['WV'][()]
                mir  = f['IMG_MIR'][()] if 'IMG_MIR' in f else f['MIR'][()]
                
                # Normalize images to [0, 1]
                tir1_n = (tir1 - np.nanmin(tir1)) / (np.nanmax(tir1) - np.nanmin(tir1) + 1e-8)
                wv_n   = (wv - np.nanmin(wv)) / (np.nanmax(wv) - np.nanmin(wv) + 1e-8)
                mir_n  = (mir - np.nanmin(mir)) / (np.nanmax(mir) - np.nanmin(mir) + 1e-8)
                
                # Physical Channel Derivation: Delta Water Vapor (TIR1 - WV)
                dwv_n  = np.clip(tir1_n - wv_n, 0, 1)
                
                # Stack into 4-Channel Spatial Tensor
                stacked_img = np.stack([tir1_n, wv_n, mir_n, dwv_n], axis=0).astype(np.float32)
                
                # Crop to 400x400 Storm Bounding Box if full disk
                if stacked_img.shape[1] > 400 and stacked_img.shape[2] > 400:
                    stacked_img = stacked_img[:, :400, :400]

                # 2. Extract Tabular Feature Vector (6-D)
                # Estimate cyclone eye location (minimum brightness temp in TIR1 channel)
                eye_y, eye_x = np.unravel_index(np.argmin(stacked_img[0]), stacked_img[0].shape)
                mean_temp = float(np.mean(stacked_img[0]))
                max_convection = float(np.max(stacked_img[3]))
                temp_std = float(np.std(stacked_img[0]))
                
                vector_feat = np.array([
                    eye_y / 400.0,       # Normalized Eye Y-coordinate
                    eye_x / 400.0,       # Normalized Eye X-coordinate
                    mean_temp,           # Mean Cloud Top Temp
                    max_convection,      # Max Updraft Index
                    temp_std,            # Temperature Variance
                    1.0                  # Placeholder for Wind/Pressure if IBTrACS merged
                ], dtype=np.float32)

                # 3. Export Multi-Modal NPZ Patch
                patch_file = OUTPUT_DIR / f"patch_{patch_count:04d}.npz"
                np.savez_compressed(patch_file, image=stacked_img, vector=vector_feat)
                patch_count += 1
                print(f"[{patch_count}/{len(h5_files)}] Saved: {patch_file.name}")

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    print(f"\n[SUCCESS] Preprocessing finished. Saved {patch_count} multi-modal samples.")

if __name__ == "__main__":
    process_multimodal_dataset()