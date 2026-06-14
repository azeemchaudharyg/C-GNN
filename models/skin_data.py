# Importing libraries
import os
import glob
import cv2

from torch.utils.data import Dataset


class SkinCancerDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # Ensure consistent order
        self.image_paths = sorted(  
            glob.glob(os.path.join(img_dir, "**", "*.jpg"), recursive=True) +
            glob.glob(os.path.join(img_dir, "**", "*.png"), recursive=True)
        )
        
        self.labels = [1 if "malignant" in img_path else 0 for img_path in self.image_paths]  

        if not self.image_paths:
            raise ValueError("No images found in dataset. Check the path!")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))  # Resize to ResNet input size

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        filenames = os.path.splitext(os.path.basename(img_path))[0]
        
        return img, label, filenames