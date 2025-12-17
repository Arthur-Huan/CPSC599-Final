import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# 2. Configuration & Hyperparameters
IMG_DIR = "data/augmented"
CSV_FILE = "data/augmented/annotations.csv"
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_LOAD_PATH = None
MODEL_SAVE_PATH = "chessboard_corners_resnet18.pth"


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
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

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
    checkpoint = torch.load(MODEL_LOAD_PATH, map_location=DEVICE)
    # Load optimizer and model state
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model.to(DEVICE)

criterion = nn.MSELoss()  # Mean Squared Error for regression

# Training Loop
print(f"Starting training on {DEVICE}...")
model.train()

curr_epoch = 0
curr_loss = 0
for epoch in range(EPOCHS):
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_print = epoch + 1
    loss_print = running_loss / len(train_loader)
    curr_epoch = epoch_print
    curr_loss = loss_print
    print(f"Epoch {epoch_print}, Loss: {loss_print:.6f}")

# Save Model
checkpoint = {
    'epoch': curr_epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': curr_loss,
}
torch.save(checkpoint, MODEL_SAVE_PATH)
print(f"Training complete. Model saved at {MODEL_SAVE_PATH}")



