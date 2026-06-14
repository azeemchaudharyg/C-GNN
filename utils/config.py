# General settings

SEED = 42
NUM_NODES = 8 #32 #8 #4 #6

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3 #1e-4 #5e-4 1e-5 
WEIGHT_DECAY=5e-4
EPOCHS = 100

# Model settings
K = 10           # Number of propagation steps for APPNP
ALPHA = 0.1      # Teleport probability for APPNP

# Dataset Paths
#DATA_PATH = "data/PAD Files/pat_ss_metadata.pkl" # "data/PAD-Process/pad_features_labels_filenames.pkl"
DATA_PATH = "data/HAM Files/ham_ss_metadata.pkl" # "data/HAM10000-Process/ham_features_labels_filenames.pkl"   #"data/HAM10000-Process/ham_features_labels_filenames.pkl" 
#DATA_PATH = "data/HIBA Files/hiba_ss_metadata.pkl"  # "data/HIBA-Process/hiba_features_labels_filenames.pkl"
TRAIN_DATA_PATH = "data/ISIC-TBP-2024-Process/isic_tbp_train_ss_metadata.pkl" #"data/ISIC-TBP-2024-Process/isic_tbp_train_features_labels_filenames.pkl"  # isic_up_resnet152_features_labels.pkl - isic_histo_features_balanced_12000.pkl - data/isic_up_resnet152_features_labels.pkl - data/isic_tbp_features_labels.pkl - data/combined_features_all_metadata.pkl
TEST_DATA_PATH =  "data/ISIC-TBP-2024-Process/isic_tbp_test_ss_metadata.pkl" #"data/ISIC-TBP-2024-Process/isic_tbp_test_features_labels_filenames.pkl" 


WANDB_PROJECT = "First_GNN"
WANDB_RUN_NAME = "APPNP_Results"
