import pandas as pd
import cv2
import os
import numpy as np
from tqdm import tqdm

metadata_path = 'data/processed/metadata.csv'
output_dir = 'data/processed/roi_images'
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(metadata_path)
roi_paths = []

print(f"Extracting ROIs for {len(df)} images...")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    img_path = row['image_path']

    # Infer mask path from image path (replacing Images with Masks)
    mask_path = img_path.replace('Images', 'Masks')
    
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        print(f"Warning: Missing file for {row['patient_id']}. Skipping.")
        roi_paths.append(None)
        continue
        
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Find bounding box from mask
    coords = cv2.findNonZero(mask)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        
        # Padding 5% around the lungs
        pad = int(max(w, h) * 0.05)

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)    # shape = (height, width), min/max are used for boundary checking of image 
        y2 = min(img.shape[0], y + h + pad)
        
        # Crop the image
        roi_img = img[y1:y2, x1:x2]
    else:
        # If mask is completely black, use full image
        roi_img = img
        
    
    roi_name = f"{row['patient_id']}_roi.png"
    roi_save_path = os.path.join(output_dir, roi_name)
    cv2.imwrite(roi_save_path, roi_img)
    
    roi_paths.append(roi_save_path)


df['roi_path'] = roi_paths

# Drop rows where extraction failed
df = df.dropna(subset=['roi_path'])
df.to_csv(metadata_path, index=False)

print("ROI Extraction complete. Updated metadata.csv saved.")