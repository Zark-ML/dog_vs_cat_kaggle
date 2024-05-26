import os
import shutil
import random
from collections import defaultdict

def split_dataset(root_dir, output_dir, split_size):
    """
    Splits the dataset into training and validation sets.

    Args:
        root_dir (string): Directory with all the images.
        output_dir (string): Directory where train/val folders will be created.
        split_size (float): Proportion of the dataset to include in the validation set.
    """
    # Create train and validation directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # List all image files and their labels
    image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
    label_to_files = defaultdict(list)
    
    for file in image_files:
        label = 'dog' if 'dog' in file else 'cat'
        label_to_files[label].append(file)
    
    train_files = []
    val_files = []

    # Split each class separately to maintain homogeneity
    for label, files in label_to_files.items():
        random.shuffle(files)
        split_idx = int(len(files) * split_size)
        val_files.extend(files[:split_idx])
        train_files.extend(files[split_idx:])
    
    # Copy files to the respective directories
    for file in train_files:
        shutil.copy(os.path.join(root_dir, file), os.path.join(train_dir, file))
    
    for file in val_files:
        shutil.copy(os.path.join(root_dir, file), os.path.join(val_dir, file))

    print(f"Train files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")


if __name__ == "__main__":
    split_dataset("/home/edgar/dev/zark_ml/dog_cat/train", "/home/edgar/dev/zark_ml/dog_cat/data/", split_size=0.2)