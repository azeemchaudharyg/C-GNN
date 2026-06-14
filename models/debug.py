import pickle
import torch
import numpy as np

import matplotlib.pyplot as plt


def load_features_labels(pickle_path):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    features = data['features']
    labels = data['labels']
    filenames = data.get('filenames', None)  # if available
    return features, labels, filenames

test_features, test_labels, test_filenames = load_features_labels('data/ISIC-TBP-2024-Process/isic_tbp_test_features_labels_filenames.pkl')

print(f"Test features shape: {test_features.shape}")
print(f"Test labels shape: {len(test_labels)}")
print(f"Example filenames: {test_filenames[:5]}")
print(f"Unique labels in test set: {np.unique(test_labels)}")

# Check feature-label alignment
assert test_features.shape[0] == len(test_labels), "Mismatch in features and labels count!"

# Check for NaN or Inf values
print("NaN in features:", np.isnan(test_features).any())
print("Inf in features:", np.isinf(test_features).any())

# Check label distribution
from collections import Counter
print("Label distribution in test set:", Counter(test_labels))


train_features, train_labels, _ = load_features_labels('data/ISIC-TBP-2024-Process/isic_tbp_train_features_labels_filenames.pkl')

print("Train features mean:", train_features.mean(axis=0)[:5])
print("Test features mean:", test_features.mean(axis=0)[:5])

print("Train features std:", train_features.std(axis=0)[:5])
print("Test features std:", test_features.std(axis=0)[:5])

for i in range(100):
    print(f"File: {test_filenames[i]} Label: {test_labels[i]}")