import pandas as pd
import numpy as np
import pickle
import re
import sys
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier



# Load extracted image features, labels, and filenames ---
with open('data/HIBA-Process/hiba_features_labels_filenames.pkl', 'rb') as f:
    data = pickle.load(f)

image_features = data['features']
labels = data['labels']
filenames = data['filenames']

# Load metadata CSV 
metadata_df = pd.read_csv("data/HIBA-Process/hiba_metadata.csv")

# Clean column names
metadata_df.columns = metadata_df.columns.str.lower()

# Identify and handle categorical/numerical columns
metadata_df = metadata_df.set_index('isic_id')

# Exclude non-feature columns
non_feature_cols = ['filename'] if 'filename' in metadata_df.columns else []

# Fill missing values
metadata_df = metadata_df.fillna({
    col: 'unknown' if metadata_df[col].dtype == 'object' else metadata_df[col].median()
    for col in metadata_df.columns if col not in non_feature_cols
})

# Identify categorical and numerical columns
categorical_cols = metadata_df.select_dtypes(include='object').columns.tolist()
numerical_cols = metadata_df.select_dtypes(include='number').columns.tolist()

# One-hot encode categorical features
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cat = encoder.fit_transform(metadata_df[categorical_cols])  
encoded_cat_df = pd.DataFrame(encoded_cat, 
                            index=metadata_df.index, 
                            columns=encoder.get_feature_names_out(categorical_cols))

# Scale numerical features
scaler = StandardScaler()
scaled_num = scaler.fit_transform(metadata_df[numerical_cols])
scaled_num_df = pd.DataFrame(scaled_num, index=metadata_df.index, columns=numerical_cols)

# Combine all metadata features
full_metadata_df = pd.concat([scaled_num_df, encoded_cat_df], axis=1)

# print(len(full_metadata_df))
# exit()

y = full_metadata_df['target']

# Train RF the model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(full_metadata_df, y)

# Get importances and feature names
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Ensure correct column names from the transformed DataFrame
feature_names = full_metadata_df.columns

top_features_names = feature_names[indices][:30]
top_metadata = full_metadata_df[top_features_names]

print("Selected Feature Shape: ", top_metadata.shape) 

# Match metadata to image filenames
meta_features = []
missing_ids = []

for fname in filenames:
    match = re.match(r"(ISIC_\d+)", fname)
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {fname}")
    
    base_id = match.group(1)
    
    try:
        row = top_metadata.loc[base_id]
        # If multiple rows (shouldn’t happen), take first
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        meta_features.append(row.to_numpy())
    except KeyError:
        # Missing metadata — add zeros
        missing_ids.append(base_id)
        meta_features.append(np.zeros(top_metadata.shape[1]))

if missing_ids:
    print(f"Warning: {len(missing_ids)} metadata entries were missing and filled with zeros.")

meta_features = np.array(meta_features)

# Combine image and metadata features
combined_features = np.concatenate([image_features, meta_features], axis=1)

# Save combined features 
with open('data/HIBA-Process/hiba_ss_metadata.pkl', 'wb') as f:
    pickle.dump({
        'features': combined_features,
        'labels': np.hstack(labels),
        'filenames': filenames
    }, f)


print("Combined feature shape:", combined_features.shape)
print("Metadata integration complete.")















'''
# Load extracted features, labels, filenames
with open('data/isic_up_features_labels_filenames.pkl', 'rb') as f:
    data = pickle.load(f)

features = data['features']
labels = data['labels']
filenames = data['filenames'] 

#print("Type of features loaded:", type(features))
#sys.exit()

# Load metadata CSV
metadata_df = pd.read_csv("data/isic_up_metadata.csv")

# Lowercase column names for consistency
metadata_df.columns = metadata_df.columns.str.lower()


# Clean column names
metadata_df = metadata_df.rename(columns={
    'num_tbp_lv_H': 'tbp_hue',
    'num_clin_size_long_diam_mm': 'clinical_diameter_mm',
    'num_tbp_lv_perimeterMM': 'tbp_perimeter_mm',
    'num_tbp_lv_minorAxisMM': 'tbp_minor_axis_mm',
    'num_tbp_lv_areaMM2': 'tbp_area_mm2',
    'num_tbp_lv_HexT': 'tbp_hex_t',
    'num_tbp_lv_radial_color_std_max': 'tbp_radial_color_std_max',
    'num_tbp_lv_deltaB': 'tbp_delta_b',
    'num_tbp_lv_nom_color': 'tbp_nominal_color'
})


# Encode categorical variables
for col in ['sex', 'anatom_site_general', 'tbp_lv_location_simple', 'iddx_full']: 
    metadata_df[col] = metadata_df[col].fillna('unknown')
        
    le = OneHotEncoder(sparse_output=False, handle_unknown='ignore')   # LabelEncoder()
    encoded = le.fit_transform(metadata_df[[col]])
        
    # Create new columns for the encoded values
    categories = le.categories_[0]
    new_columns = {f"{col}_{category}": encoded[:, i] 
                for i, category in enumerate(categories)}
        
    # Add new columns to DataFrame
    for new_col_name, new_col_values in new_columns.items():
        metadata_df[new_col_name] = new_col_values
        
    # Drop the original column
    metadata_df.drop(col, axis=1, inplace=True)


print(metadata_df.columns)
sys.exit()

# Fill missing numerical values
metadata_df['age'] = metadata_df['age'].fillna(metadata_df['age'].median()).astype('int')

# Match metadata to each image
metadata_lookup = metadata_df.set_index('isic_id')

#print(metadata_df[['isic_id', 'age', 'sex', 'lesion_site', 'lesion_location']])
#sys.exit()

meta_features = []

for fname in filenames:
    base_id = re.match(r"(ISIC_\d+)", fname).group(1)   # ISIC_0082829_aug_1.jpg

    if base_id in metadata_lookup.index:
        row = metadata_lookup.loc[base_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]  # Use the first match

        age = int(row['age']) if pd.notna(row['age']) else 0
        sex = row['sex'] if pd.notna(row['sex']) else 'unknown'
        site = row['lesion_site'] if pd.notna(row['lesion_site']) else 'unknown'
        location = row['lesion_location'] if pd.notna(row['lesion_location']) else 'unknown'

        meta_features.append([age, sex, site, location])

# Scale metadata
scaler_meta = StandardScaler()
meta_features = scaler_meta.fit_transform(meta_features)

# Combine features and metadata
combined_features = np.concatenate([features, meta_features], axis=1)

combined_features = np.array(combined_features)

# Save final combined features and labels
with open('data/combined_features_with_5_metadata.pkl', 'wb') as f:
    pickle.dump({
        'features': np.vstack(combined_features),
        'labels': np.hstack(labels),
        'filenames': filenames
    }, f)

print("Combined feature shape:", combined_features.shape)
print("Metadata successfully integrated.")

# Load the features, labels, and filenames from the file
with open('data/combined_features_with_metadata.pkl', 'rb') as f:
    data = pickle.load(f)

all_features = data['features']
all_labels = data['labels']
filenames = data['filenames']

print("Features and labels have been loaded successfully.")

print("Type of features loaded:", type(all_features))
#sys.exit()
'''


'''
# Load extracted features, labels, filenames
with open('data/PAD-Process/pad_features_labels_filenames.pkl', 'rb') as f:
    data = pickle.load(f)

features = data['features']
labels = data['labels']
filenames = data['filenames']

# Load metadata CSV
metadata_df = pd.read_csv("data/PAD-Process/PAD-Process-Metadata.csv", low_memory=False)
metadata_df.columns = metadata_df.columns.str.lower()

# Rename relevant columns
metadata_df = metadata_df.rename(columns={
    'diameter_1': 'diameter_1',
    'diameter_2': 'diameter_2',
    'age': 'age',
    'changed': 'change',
    'elevation': 'elevation',
    'itch': 'itch',
    'region': 'region',
    'grew': 'grew',
    'gender': 'sex',
    'fitspatrick': 'fit'
})

# Keep only selected columns + isic_id
selected_columns = [
    'img_id',
    'diameter_1',
    'diameter_2',
    'age',
    'change',
    'elevation',
    'itch',
    'region',
    'grew',
    'sex',
    'fit'
]
metadata_df = metadata_df[selected_columns]

# Fill missing values
#metadata_df['age'] = metadata_df['age'].fillna(metadata_df['age'].median())
for col in ['age', 'fit', 'diameter_1', 'diameter_2']:
    metadata_df[col] = metadata_df[col].fillna(metadata_df[col].median())
    
for col in ['sex', 'change', 'elevation', 'itch', 'region', 'grew']:
    metadata_df[col] = metadata_df[col].fillna('unknown')

# One-hot encode categorical features
categorical_cols = ['sex', 'change', 'elevation', 'itch', 'region', 'grew']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cat = encoder.fit_transform(metadata_df[categorical_cols])
encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(categorical_cols))

# Scale numerical features
numerical_cols = ['age', 'fit', 'diameter_1', 'diameter_2']
metadata_df[numerical_cols] = metadata_df[numerical_cols].fillna(metadata_df[numerical_cols].median())

scaler = StandardScaler()
scaled_num = scaler.fit_transform(metadata_df[numerical_cols])
scaled_num_df = pd.DataFrame(scaled_num, columns=numerical_cols)


# Combine encoded categorical and scaled numerical
full_metadata_df = pd.concat([metadata_df[['img_id']], scaled_num_df, encoded_cat_df], axis=1)
full_metadata_df.set_index('img_id', inplace=True)

#print(full_metadata_df.head(5))
#print(len(full_metadata_df))
#print(full_metadata_df.tail(5))
#exit()

# Match metadata rows with image filenames
meta_features = []
missing_ids = []

for fname in filenames:
    match = re.match(r"(PAT_\d+_\d+_\d+)", fname)
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {fname}")
    base_id = match.group(1)
    print(base_id)
    if base_id in full_metadata_df.index:
        row = full_metadata_df.loc[base_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        meta_features.append(row.to_numpy())
    else:
        missing_ids.append(base_id)
        meta_features.append(np.zeros(full_metadata_df.shape[1]))

if missing_ids:
    print(f"Warning: {len(missing_ids)} metadata entries were missing and filled with zeros.")

meta_features = np.array(meta_features)

print(f"Meta Feature Shape: {meta_features.shape} ")
#exit()



full_metadata_df = pd.concat([metadata_df[['img_id']], scaled_num_df, encoded_cat_df], axis=1)

# Ensure img_id column is the index for faster lookup
full_metadata_df.reset_index(inplace=True)
full_metadata_df['img_id'] = full_metadata_df['img_id'].astype(str)
full_metadata_df.set_index('img_id', inplace=True)

print(full_metadata_df.index)

meta_features = []
missing_ids = []

for fname in filenames:
    # Get full image filename (e.g., PAT_995_1867_5_aug_3.png)
    full_img_id = os.path.basename(fname)
    print(full_img_id)

    if full_img_id in full_metadata_df.index:
        row = full_metadata_df.loc[full_img_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]  # Just in case of duplicates
        meta_features.append(row.to_numpy())
    else:
        missing_ids.append(full_img_id)
        meta_features.append(np.zeros(full_metadata_df.shape[1]))

if missing_ids:
    print(f"Warning: {len(missing_ids)} metadata entries were missing and filled with zeros.")
    print("Example missing IDs:", missing_ids[:5])

meta_features = np.array(meta_features)


# Combine image features with metadata features
combined_features = np.concatenate([features, meta_features], axis=1)

# Save combined data
with open('data/PAD-Process/pat_ss_metadata.pkl', 'wb') as f:
    pickle.dump({
        'features': combined_features,
        'labels': np.hstack(labels),
        'filenames': filenames
    }, f)

print("Combined feature shape:", combined_features.shape)
print("Metadata successfully integrated with selected columns.")
'''