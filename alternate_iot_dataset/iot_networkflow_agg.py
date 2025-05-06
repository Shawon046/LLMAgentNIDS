import os
import pandas as pd

# Define root directory and output path
ROOT_DIR = "/home/tauhid/llm_network_intrusion/dataset/iot_dataset"
SELECTED_FOLDERS = {"Benign_traffic", "Mirai", "Bruteforce", "Spoofing"}
OUTPUT_FILE = os.path.join(ROOT_DIR, "iot_network_flow_data_benign_mirai_bruteforce_spoofing.parquet")

# Store all DataFrames
dataframes = []

# Loop through only the selected subdirectories
for folder in os.listdir(ROOT_DIR):
    if folder not in SELECTED_FOLDERS:
        continue  # Skip non-target folders

    folder_path = os.path.join(ROOT_DIR, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                file_path = os.path.join(folder_path, file)
                try:
                    df = pd.read_csv(file_path)

                    # Drop existing "Label" column if present
                    if 'Label' in df.columns:
                        df = df.drop(columns=['Label'])

                    # Add custom label based on folder name
                    df['label'] = folder

                    dataframes.append(df)
                    print(f"✅ Loaded: {file} from {folder}")
                except Exception as e:
                    print(f"❌ Failed to load {file} in {folder}: {e}")

# Concatenate and save to parquet
if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\n🎉 Combined data saved to {OUTPUT_FILE}")
else:
    print("⚠️ No valid CSV files found.")