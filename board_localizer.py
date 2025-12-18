import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ResNet18 to predict chessboard corner coordinates")

    # Paths
    parser.add_argument('--img-dir', type=str, default="data/augmented_train",
                        help='Directory containing training images')
    parser.add_argument('--csv-file', type=str, default="data/augmented_train/_annotations.csv",
                        help='CSV file with annotations')
    parser.add_argument('--model-save-path', type=str, default="models/chessboard_corners_resnet18.pth",
                        help='Path to save the best model checkpoint')
    parser.add_argument('--model-load-path', type=str, default=None,
                        help='Path to an existing model checkpoint to resume training (optional)')

    # Hyperparameters
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Fraction of data to use for validation (0.0-1.0)')
    parser.add_argument('--random-seed', type=int, default=132)
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--min-delta', type=float, default=1e-4,
                        help='Minimum change in validation loss to qualify as improvement')

    return parser.parse_args()


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


def main():
    args = parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Basic validation
    if not os.path.isdir(args.img_dir):
        raise FileNotFoundError(f"Image directory not found: {args.img_dir}")
    if not os.path.isfile(args.csv_file):
        raise FileNotFoundError(f"CSV file not found: {args.csv_file}")

    model_save_dir = os.path.dirname(args.model_save_path)
    if model_save_dir and not os.path.isdir(model_save_dir):
        os.makedirs(model_save_dir, exist_ok=True)

    # Data Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ChessboardDataset(csv_file=args.csv_file, img_dir=args.img_dir, transform=transform)

    # Create train / validation split
    dataset_size = len(dataset)
    val_size = max(1, int(dataset_size * args.val_split)) if dataset_size > 1 else 0
    train_size = dataset_size - val_size
    generator = torch.Generator().manual_seed(args.random_seed)
    if val_size > 0:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    else:
        train_dataset = dataset
        val_dataset = []

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) if val_size > 0 else []

    # Model init / load
    if args.model_load_path is None:
        # Transfer learn the ResNet18 model
        model = models.resnet18(weights='DEFAULT')
        # Freeze layers
        for param in model.parameters():
            param.requires_grad = False
        # Replace the final FC layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 8) # 8 outputs: (x1, y1, x2, y2, x3, y3, x4, y4)

        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    else:  # Load existing local model
        # Init model and optimizer (will be overwritten by checkpoint)
        model = models.resnet18(weights='DEFAULT')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        # Load checkpoint
        path = str(args.model_load_path)
        checkpoint = torch.load(path, map_location=device)
        # Load optimizer and model state
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', None)

        model.to(device)

    criterion = nn.MSELoss()  # Mean Squared Error for regression

    # Training Loop
    print(f"Starting training on {device}...")

    # Early stopping trackers
    best_val = float('inf')
    epochs_no_improve = 0

    curr_epoch = 0
    curr_loss = 0
    for epoch in range(args.epochs):
        running_loss = 0.0
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", unit="batch", leave=False):
            images = images.to(device)
            labels = labels.to(device)
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
        if val_size > 0:
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", unit="batch", leave=False):
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    vloss = criterion(outputs, labels)
                    val_running += vloss.item()
            val_loss = val_running / (len(val_loader) if len(val_loader) > 0 else 1)
        else:
            val_loss = train_loss

        # Early stopping check
        improved = (best_val - val_loss) > args.min_delta
        if improved:
            best_val = val_loss
            epochs_no_improve = 0
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }
            torch.save(checkpoint, args.model_save_path)
        else:
            epochs_no_improve += 1

        # Early stopping if patience exceeded
        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered. No improvement for {epochs_no_improve} epochs.")
            curr_epoch = epoch + 1
            curr_loss = best_val
            break

        curr_epoch = epoch + 1
        curr_loss = val_loss
        print(f"\nEpoch {epoch+1}, Train Loss: {train_loss:.7f}, Val Loss: {val_loss:.7f}")


if __name__ == '__main__':
    main()
