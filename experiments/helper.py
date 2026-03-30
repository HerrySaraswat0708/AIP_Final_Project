import os
import shutil
from pathlib import Path

# Paths - Update these to your actual server paths
source_dir = Path("data/raw/PET/oxford-iiit-pet/images")
target_dir = Path("data/raw/PET/oxford-iiit-pet/classes")

if not source_dir.exists():
    print(f"Error: Source directory {source_dir} not found.")
else:
    target_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for img_path in source_dir.glob("*.jpg"):
        # Extract breed: "Abyssinian_123" -> "Abyssinian"
        # "yorkshire_terrier_1" -> "yorkshire_terrier"
        parts = img_path.stem.split('_')
        breed_name = "_".join(parts[:-1])
        
        # Create breed folder
        breed_folder = target_dir / breed_name
        breed_folder.mkdir(exist_ok=True)
        
        # Move file
        shutil.move(str(img_path), str(breed_folder / img_path.name))
        count += 1

    print(f"Success! Organized {count} images into {len(os.listdir(target_dir))} breed folders.")