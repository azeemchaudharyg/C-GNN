# Importing libraries
import numpy as np
import pickle

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.skin_data import SkinCancerDataset
from models.feature_extractor import ResNetFeatureExtractor, EfficientNetFeatureExtractor, VGG19FeatureExtractor, InceptionV3FeatureExtractor

from sklearn.preprocessing import StandardScaler


# Set to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load and Preprocess Image
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = SkinCancerDataset(img_dir="data/PAD-Process", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Get one batch from the DataLoader
for images, labels, filenames in dataloader:
    print("Batch of Images Shape:", images.shape)  # Shape of images
    print("Batch of Labels Shape:", labels.shape)  # Shape of labels
    break  # Print only the first batch

print("Total Images: ", len(dataset))
print("Total Batches: ", len(dataloader))

# Initialize resnet model
#model = ResNetFeatureExtractor().to(device).eval()
#model = EfficientNetFeatureExtractor().to(device).eval()
#model = VGG19FeatureExtractor().to(device).eval()
model = InceptionV3FeatureExtractor().to(device).eval()


all_features = []
all_labels = []
all_filenames = []

with torch.no_grad():
    for images, labels, filenames in dataloader:
        images = images.to(device)
        features = model(images)

        all_features.append(features.cpu().numpy())
        all_labels.append(labels.numpy())
        all_filenames.extend(filenames) 
        
# Stack all batches into single arrays
all_features = np.vstack(all_features)
all_labels = np.hstack(all_labels)

scaler = StandardScaler()
all_features = scaler.fit_transform(all_features)

with open("data/PAD Files/pad_inception_features.pkl", "wb") as f:
    pickle.dump({
        "features": np.vstack(all_features),
        "labels": np.hstack(all_labels),
        "filenames": all_filenames
    }, f) 

print("Features, labels, and filenames saved successfully.")


'''
# Save the scaler to apply same transformation on test set later
with open('isic_tbp_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    
# Load and apply for test and validation set
with open('isic_tbp_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

X_test_scaled = scaler.transform(X_test_raw)
'''


# Load the features, labels, and filenames from the file
with open('data/PAD Files/pad_inception_features.pkl', 'rb') as f:
    data = pickle.load(f)

all_features = data['features']
all_labels = data['labels']
filenames = data['filenames']

print("Features and labels have been loaded successfully.")

print("Feature Shape: ", all_features.shape)
print("Label Shape: ", all_labels.shape)
print("Number of Filenames: ", len(filenames))

print("Type of features loaded:", type(all_features))
