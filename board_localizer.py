import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# Configuration & Hyperparameters
IMG_DIR = "data/augmented_train"
CSV_FILE = "data/augmented_train/_annotations.csv"
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_LOAD_PATH = None
MODEL_SAVE_PATH = "models/chessboard_corners_resnet18.pth"
VAL_SPLIT = 0.2  # fraction of data to use for validation
RANDOM_SEED = 132
PATIENCE = 5        # early stopping patience (epochs)
MIN_DELTA = 1e-4    # minimum change to qualify as improvement


# Dataset class for the data
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
        
        # Extract 8 coordinates
        loaded_labels = self.annotations.iloc[index, 3:].values.astype('float32')
        
        orig_w, orig_h = image.size
        if self.transform:
            image = self.transform(image)
        # Normalize labels to [0, 1] based on original image size
        loaded_labels[0::2] /= orig_w  # x coordinates
        loaded_labels[1::2] /= orig_h  # y coordinates
        
        return image, torch.tensor(loaded_labels)

# Data Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = ChessboardDataset(csv_file=CSV_FILE, img_dir=IMG_DIR, transform=transform)

# Create train / validation split
dataset_size = len(dataset)
val_size = max(1, int(dataset_size * VAL_SPLIT))
train_size = dataset_size - val_size
generator = torch.Generator().manual_seed(RANDOM_SEED)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# TODO: Visualize the dataset

if MODEL_LOAD_PATH is None:
    # Transfer learn the ResNet18 model
    model = models.resnet18(weights='DEFAULT')
    # Freeze layers
    for param in model.parameters():
        param.requires_grad = False
    # Replace the final FC layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 8) # 8 outputs: (x1, y1, x2, y2, x3, y3, x4, y4)

    model = model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

else:  # Load existing local model
    # Init model and optimizer
    model = models.resnet18(weights='DEFAULT')
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Load checkpoint
    assert MODEL_LOAD_PATH is not None, "MODEL_LOAD_PATH must be set when loading a model"
    path = str(MODEL_LOAD_PATH)
    checkpoint = torch.load(path, map_location=DEVICE)
    # Load optimizer and model state
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model.to(DEVICE)

criterion = nn.MSELoss()  # Mean Squared Error for regression

# Training Loop
print(f"Starting training on {DEVICE}...")

# Early stopping trackers
best_val = float('inf')
epochs_no_improve = 0

curr_epoch = 0
curr_loss = 0
for epoch in range(EPOCHS):
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", unit="batch", leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / (len(train_loader) if len(train_loader) > 0 else 1)

    # Validation
    model.eval()
    val_running = 0.0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", unit="batch", leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            vloss = criterion(outputs, labels)
            val_running += vloss.item()
    val_loss = val_running / (len(val_loader) if len(val_loader) > 0 else 1)

    # Early stopping check
    improved = (best_val - val_loss) > MIN_DELTA
    if improved:
        best_val = val_loss
        epochs_no_improve = 0
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }
        torch.save(checkpoint, MODEL_SAVE_PATH)
    else:
        epochs_no_improve += 1

    # Early stopping if patience exceeded
    if epochs_no_improve >= PATIENCE:
        print(f"Early stopping triggered. No improvement for {epochs_no_improve} epochs.")
        curr_epoch = epoch + 1
        curr_loss = best_val
        break

    model.train()

    curr_epoch = epoch + 1
    curr_loss = val_loss
    print(f"\nEpoch {epoch+1}, Train Loss: {train_loss:.7f}, Val Loss: {val_loss:.7f}")
