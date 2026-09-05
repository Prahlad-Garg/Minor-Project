import cv2
import math
import numpy as np
import tensorflow as tf
import albumentations as A

class PneumoniaDataset(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=32, use_roi=True, augment=False, shuffle=True):
        """
        Args:
            df (DataFrame): Metadata containing image paths and labels.
            batch_size (int): Size of the batches.
            use_roi (bool): If True, uses 'roi_path'. If False, uses original 'image_path'.
            augment (bool): If True, applies training augmentations.
            shuffle (bool): If True, shuffles data every epoch.
        """
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.use_roi = use_roi
        self.augment = augment
        self.shuffle = shuffle
        
        if self.augment:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=[0.485], std=[0.229]), # Standard ImageNet single-channel values
            ])
        else:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485], std=[0.229]), # No transformation in case of validation/testing
            ])
            
        self.indices = np.arange(len(self.df)) 
        if self.shuffle:
            np.random.shuffle(self.indices) # So that model not always sees data in same order


    def __len__(self):
        # Denotes the number of batches per epoch
        return math.ceil(len(self.df) / self.batch_size)


    def on_epoch_end(self):
        # Updates indexes after each epoch
        if self.shuffle:
            np.random.shuffle(self.indices)


    def __getitem__(self, idx):
        # Generate one batch of data, Generate indices of the batch
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indices]
        
        # Initialize batch arrays
        X = np.empty((len(batch_df), 224, 224, 1), dtype=np.float32)
        y = np.empty((len(batch_df),), dtype=np.float32)
        
        for i, (_, row) in enumerate(batch_df.iterrows()):
            img_path = row['roi_path'] if self.use_roi else row['image_path']
            
            # Read image in grayscale (Chest X-rays are 1 channel)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                # Fallback to zero array if image doesn't exist to prevent crash
                image = np.zeros((224, 224), dtype=np.uint8)
                
            # Apply transforms
            augmented = self.transform(image=image)
            aug_img = augmented['image']
            
            # Keras Conv2D expects 4D tensor (batch, height, width, channels)
            X[i,] = np.expand_dims(aug_img, axis=-1)
            y[i] = row['label']
            
        return X, y
