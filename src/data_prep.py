import pandas as pd
import os
from sklearn.model_selection import train_test_split

labels_path = 'data/raw/rsna/stage2_train_metadata.csv'
output_path = 'data/processed/metadata.csv'

if not os.path.exists(labels_path):
    print(f"Please place labels at {labels_path}")
    exit()


df = pd.read_csv(labels_path)

# Get max target per patient as single patient has multiple images of X-Ray corresponding to different infection points(so calculate max(0 ,1))
patient_df = df.groupby('patientId')['Target'].max().reset_index()


# 70/15/15 Split
train, temp = train_test_split(patient_df, test_size=0.30, stratify=patient_df['Target'], random_state=42)
val, test = train_test_split(temp, test_size=0.50, stratify=temp['Target'], random_state=42)


# Map which tells which patientid is for training, testing, validation
split_map = {**{p: 'train' for p in train['patientId']}, 
             **{p: 'val' for p in val['patientId']}, 
             **{p: 'test' for p in test['patientId']}}


# Making final dataset
final_df = df[['patientId', 'Target']].drop_duplicates().copy()
final_df['split'] = final_df['patientId'].map(split_map)

final_df['image_path'] = "data/raw/rsna/Training/Images/" + final_df['patientId'] + ".png"
final_df.columns = ['patient_id', 'label', 'split', 'image_path']


os.makedirs(os.path.dirname(output_path), exist_ok=True)
final_df.to_csv(output_path, index=False)
print(f"Saved split metadata to {output_path}")