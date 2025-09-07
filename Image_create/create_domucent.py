import os
import shutil
from glob import glob


base_dir = r''


png_files = glob(os.path.join(base_dir, '*.png'))

for file_path in png_files:
    filename = os.path.basename(file_path)                # '26.png'
    name_without_ext = os.path.splitext(filename)[0]      # '26'

    target_dir = os.path.join(base_dir, name_without_ext) # '.../results/26'
    target_path = os.path.join(target_dir, filename)      # '.../results/26/26.png'

    os.makedirs(target_dir, exist_ok=True)
    shutil.move(file_path, target_path)

    print(f"Moved: {file_path} -> {target_path}")

subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

for subdir in subdirs:
    subdir_path = os.path.join(base_dir, subdir)
    png_in_subdir = glob(os.path.join(subdir_path, '*.png'))

    if png_in_subdir:
        old_path = png_in_subdir[0]
        new_path = os.path.join(subdir_path, 'image.png')
        os.rename(old_path, new_path)
        print(f"Renamed: {old_path} -> {new_path}")
