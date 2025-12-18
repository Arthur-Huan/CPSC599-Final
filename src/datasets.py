import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ChessboardDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # CSV format: filename, width, height, tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path).convert('RGB')

        # Extract coordinates: CSV has 8 values after width/height in order
        # (tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y)
        # Only tl and br corners needed, as these two can predict the square board area
        all_coords = self.annotations.iloc[index, 3:].values.astype('float32')
        loaded_labels = all_coords[[0, 1, 6, 7]]

        orig_w, orig_h = image.size
        if self.transform:
            image = self.transform(image)
        # Normalize labels to [0, 1] based on original image size
        loaded_labels[0::2] /= orig_w  # x coordinates
        loaded_labels[1::2] /= orig_h  # y coordinates

        return image, torch.tensor(loaded_labels)

