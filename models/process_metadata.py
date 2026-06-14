import os
import pandas as pd

# Paths
metadata_path = "data/PAD-Process/PAD_Metadata.csv"   
image_dir = "data/PAD-Process"                    
output_metadata_path = "data/PAD-Process/PAD_Processed_Metadata.csv"

# Load original metadata
df_meta = pd.read_csv(metadata_path)

# Convert isic_id column to string (in case it is not)
df_meta["img_id"] = df_meta["img_id"].astype(str)

# Prepare list to collect new metadata rows
new_metadata = []

# Traverse both folders
for label_folder in ['benign', 'malignant']:
    label = 0 if label_folder == 'benign' else 1
    folder_path = os.path.join(image_dir, label_folder)
    
    for fname in os.listdir(folder_path):
        if not fname.endswith('.png'):
            continue

        img_name = fname.split('.')[0]  # Remove .jpg
        if "_aug_" in img_name:
            original_id = img_name.split("_aug_")[0]
        else:
            original_id = img_name
        
        # Find metadata row for original image
        match = df_meta[df_meta["img_id"] == original_id]
        if not match.empty:
            row = match.iloc[0].to_dict()
            row["filename"] = fname  # Add new filename
            new_metadata.append(row)
        else:
            print(f"Metadata not found for: {original_id}")

# Convert and save
df_new = pd.DataFrame(new_metadata)
df_new.to_csv(output_metadata_path, index=False)
print(f"New metadata saved to {output_metadata_path}")
