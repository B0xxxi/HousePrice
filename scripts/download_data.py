import os
import subprocess
import shutil
import pandas as pd
import glob

DATA_DIR = "data"
REPO_URL = "https://github.com/emanhamed/Houses-dataset.git"
REPO_DIR = os.path.join(DATA_DIR, "Houses-dataset")

def download_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    if os.path.exists(REPO_DIR):
        print(f"Directory {REPO_DIR} already exists. Skipping download.")
    else:
        print(f"Cloning {REPO_URL} into {DATA_DIR}...")
        subprocess.check_call(["git", "clone", REPO_URL, REPO_DIR])
        print("Download complete.")

def process_metadata():
    print("Processing metadata...")
    # The dataset has a text file 'HousesInfo.txt'
    # Format: 
    # <index> <#bedrooms> <#bathrooms> <area> <zipcode> <price>
    
    info_file = os.path.join(REPO_DIR, "Houses Dataset", "HousesInfo.txt")
    
    if not os.path.exists(info_file):
        print(f"Error: {info_file} not found.")
        return

    cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
    df = pd.read_csv(info_file, sep=" ", header=None, names=cols)
    
    # Add image paths
    # Images are named like: <index>_frontal.jpg, <index>_bedroom.jpg, etc.
    # We only care about frontal for now.
    
    image_dir = os.path.join(REPO_DIR, "Houses Dataset")
    
    image_paths = []
    valid_indices = []
    
    for idx in df.index:
        # The text file index seems to match the line number (1-based in file? or 0-based?)
        # Let's check the file content structure usually.
        # Actually, the file likely doesn't have the index in the column, or maybe it does?
        # Let's assume the lines correspond to 1..N images.
        # Wait, usually these datasets have filenames like 1_frontal.jpg.
        
        # Let's verify by looking for files.
        # The index in the file might not be explicit, so we assume line i corresponds to house i+1.
        
        house_id = idx + 1
        img_name = f"{house_id}_frontal.jpg"
        img_path = os.path.join(image_dir, img_name)
        
        if os.path.exists(img_path):
            image_paths.append(img_path)
            valid_indices.append(idx)
        else:
            # Try with leading zeros or just check what exists?
            # Let's try to find the file pattern if strict naming fails, but usually it is strict.
            print(f"Warning: Image {img_path} not found. Skipping.")
            
    
    clean_df = df.iloc[valid_indices].copy()
    clean_df["image_path"] = image_paths
    
    output_csv = os.path.join(DATA_DIR, "houses.csv")
    clean_df.to_csv(output_csv, index=False)
    print(f"Processed {len(clean_df)} records. Saved to {output_csv}")

if __name__ == "__main__":
    download_data()
    process_metadata()
