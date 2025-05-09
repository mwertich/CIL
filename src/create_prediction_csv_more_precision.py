import os
import numpy as np
import pandas as pd
import base64
import zlib
from tqdm import tqdm
import zipfile

# Path definitions
data_root = 'src/data/'
predictions_dir = os.path.join(data_root, 'predictions')
test_list_file = os.path.join(data_root, 'test_list.txt')
output_csv = os.path.join(data_root, 'predictions.csv')

def compress_depth_values(depth_values):
    # Convert depth values to raw bytes (float32)
    depth_bytes = depth_values.astype(np.float32).tobytes()
    # Compress using zlib
    compressed = zlib.compress(depth_bytes, level=9)
    # Encode as base64 for CSV-safe storage
    return base64.b64encode(compressed).decode('utf-8')

def process_depth_maps():
    # Read file list
    with open(test_list_file, 'r') as f:
        file_pairs = [line.strip().split() for line in f]
    
    # Initialize lists to store data
    ids = []
    depths_list = []
    
    # Process each depth map
    for rgb_path, depth_path in tqdm(file_pairs, desc="Processing depth maps"):
        # Get file ID (without extension)
        file_id = os.path.splitext(os.path.basename(depth_path))[0]
        
        # Load depth map
        depth = np.load(os.path.join(predictions_dir, depth_path))
        # Flatten depth without rounding
        flattened_depth = depth.flatten()
        
        # Compress the depth values
        compressed_depths = compress_depth_values(flattened_depth)
        ids.append(file_id)
        depths_list.append(compressed_depths)

    # Create DataFrame
    df = pd.DataFrame({
        'id': ids,
        'Depths': depths_list,
    })
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"CSV file saved to: {output_csv}")
    print(f"Shape of the CSV: {df.shape}")

    # Define ZIP file path
    output_zip = output_csv.replace('.csv', ) + '.zip'

    # Create the ZIP file
    with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_csv, arcname=os.path.basename(output_csv))

    print(f"Zipped file saved to: {output_zip}")

if __name__ == "__main__":
    process_depth_maps()
