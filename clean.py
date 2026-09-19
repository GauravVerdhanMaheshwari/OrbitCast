import shutil
from pathlib import Path

# Define Root Directory
BASE_DIR = Path("./orbitcast_project")

# Clear old files if restarting
if BASE_DIR.exists():
    shutil.rmtree(BASE_DIR)
    print("Cleaned up old workspace.")

# Create structured folders
(BASE_DIR / "raw_data" / "cyclone_01").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "raw_data" / "cyclone_02").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "processed_patches").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "models").mkdir(parents=True, exist_ok=True)

print(f"[SUCCESS] Environment ready at: {BASE_DIR.resolve()}")