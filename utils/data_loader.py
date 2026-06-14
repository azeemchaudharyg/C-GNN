import numpy as np
import pickle
import sys

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split



features_per_node = 0


def load_features_labels(pickle_path):
    with open(pickle_path, 'rb') as f:
            data = pickle.load(f)    # features, labels = pickle.load(f)  

    
    features = data['features']
    labels = data['labels']
    filenames = data['filenames']
    

    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).float()
    else:
        raise ValueError("Features must be a NumPy array.")

    labels = torch.tensor(labels, dtype=torch.long)

    return features, labels, filenames


def pad_features(features, num_nodes):
    total_features = features.shape[1]
    remainder = total_features % num_nodes
    
    if remainder != 0:
        padding = num_nodes - remainder
        pad_tensor = torch.zeros((features.size(0), padding))
        features = torch.cat([features, pad_tensor], dim=1)
        
        print(f"Padded features from {total_features} to {features.size(1)}")
        
    return features

def split_data(features, labels, num_nodes):
    
    features = pad_features(features, num_nodes)
    
    global features_per_node
    
    features_per_node = features.size(1) // num_nodes
    
    # Check if divisible
    assert features.size(1) % num_nodes == 0, "Feature size not divisible by number of nodes!"
    
    node_features = features.reshape(features.size(0), num_nodes, features_per_node)
        
    # Train 80% Test 20%
    train_val_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=42)
    y_all = torch.tensor(labels, dtype=torch.long)
    
    train_val_labels = y_all[train_val_idx]
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.1, stratify=train_val_labels, random_state=42)
    
    return node_features, y_all, train_idx, val_idx, test_idx #train_val_idx,


def build_data_loaders(node_features, y_all, edge_index, train_idx, val_idx, test_idx, batch_size):  # train_val_idx,
    def create_loader(indices):
        return DataLoader(
            [Data(x=node_features[i], edge_index=edge_index, y=y_all[i]) for i in indices],
            batch_size=batch_size, shuffle=True
        )
    return create_loader(train_idx), create_loader(test_idx), create_loader(val_idx)



''''
# For ISIC

features_per_node = 0

def load_features_labels(pickle_path):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)  # Expected: dict with 'features', 'labels', 'filenames'
    
    features = data['features']
    labels = data['labels']
    filenames = data.get('filenames', None)  # Optional
    
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).float()
    else:
        raise ValueError("Features must be a NumPy array.")
    
    labels = torch.tensor(labels, dtype=torch.long)
    return features, labels, filenames


def pad_features(features, num_nodes):
    total_features = features.shape[1]
    remainder = total_features % num_nodes
    
    if remainder != 0:
        padding = num_nodes - remainder
        pad_tensor = torch.zeros((features.size(0), padding))
        features = torch.cat([features, pad_tensor], dim=1)
        print(f"Padded features from {total_features} to {features.size(1)}")
        
    return features


def split_train_val(features, labels, num_nodes, val_size=0.1, random_state=42):
    features = pad_features(features, num_nodes)
    
    global features_per_node
    features_per_node = features.size(1) // num_nodes
    
    assert features.size(1) % num_nodes == 0, "Feature size not divisible by number of nodes!"
    
    node_features = features.reshape(features.size(0), num_nodes, features_per_node)
    
    # Split training data into train and validation
    train_idx, val_idx = train_test_split(np.arange(len(labels)), test_size=val_size, stratify=labels, random_state=random_state)
    
    y_all = labels
    
    return node_features, y_all, train_idx, val_idx


def prepare_test(features, labels, num_nodes):
    features = pad_features(features, num_nodes)
    global features_per_node
    features_per_node = features.size(1) // num_nodes
    assert features.size(1) % num_nodes == 0, "Feature size not divisible by number of nodes!"
    node_features = features.reshape(features.size(0), num_nodes, features_per_node)
    y_all = labels
    return node_features, y_all


def build_data_loaders(node_features, y_all, edge_index, train_idx, val_idx, test_idx, batch_size):
    def create_loader(indices, shuffle=True):
        return DataLoader(
            [Data(x=node_features[i], edge_index=edge_index, y=y_all[i]) for i in indices],
            batch_size=batch_size,
            shuffle=shuffle
        )
    train_loader = create_loader(train_idx, shuffle=True)
    val_loader = create_loader(val_idx, shuffle=False)
    test_loader = create_loader(test_idx, shuffle=False)
    
    return train_loader, val_loader, test_loader
'''