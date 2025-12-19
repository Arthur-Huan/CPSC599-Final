import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from tqdm import tqdm

from src.datasets import CornersDataset
from src.utils import visualize_image_grid


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ResNet18 to predict chessboard corner coordinates")
    # Paths
    parser.add_argument('--img-dir', type=str,
                        default="../data/augmented_train")
    parser.add_argument('--csv-file', type=str,
                        default="../data/augmented_train/_annotations.csv")
    parser.add_argument('--model-save-path', type=str,
                        default="../models/chessboard_corners_resnet18.pth")
    parser.add_argument('--model-load-path', type=str,
                        default=None)
    # Hyperparameters
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--random-seed', type=int, default=132)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min-delta', type=float, default=1e-4)
    return parser.parse_args()


def compute_loss(outputs, labels):
    """
    Compute the corner SmoothL1 loss

    TODO: Implement a grid reprojection loss on top of corner loss.

    @return: loss (tensor)
    """
    # Corner loss with SmoothL1
    criterion = nn.SmoothL1Loss()
    loss = criterion(outputs, labels)
    return loss


def main():
    args = parse_args()

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

    dataset = CornersDataset(csv_file=args.csv_file, img_dir=args.img_dir, transform=transform)
    # Visualize random samples from the dataset
    visualize_image_grid(dataset, 4, 5)

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

    # Data loader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) if val_size > 0 else []

    # Model init / load
    model = models.resnet18(weights='DEFAULT')
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze final layer
    for param in model.layer4.parameters():
        param.requires_grad = True
    # Replace final layer to output 4 coordinates
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 4)  # 4 outputs: (tl_x, tl_y, br_x, br_y)
    model = model.to(device)

    # Create optimizer after model is constructed so its param references match checkpoint
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # If a checkpoint path is provided, load model and optimizer state
    if args.model_load_path is not None:
        path = str(args.model_load_path)
        checkpoint = torch.load(path, map_location=device)
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        # Load optimizer state if present
        if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Move optimizer state tensors to the correct device
            for state in optimizer.state.values():
                for k, v in list(state.items()):
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

        model.to(device)

    """Training loop"""

    print(f"Starting training on {device}...")

    # Early stopping trackers
    best_val = float('inf')
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        running_loss = 0.0

        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", unit="batch", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = compute_loss(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / (len(train_loader) if len(train_loader) > 0 else 1)

        # Validation
        model.eval()
        val_running = 0.0
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]", unit="batch", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = compute_loss(outputs, labels)
            val_running += loss.item()
        val_loss = val_running / (len(val_loader) if len(val_loader) > 0 else 1)

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
            break

        print(f"\n  Epoch {epoch+1}, Train Loss: {train_loss:.7f}, Val Loss: {val_loss:.7f}")


if __name__ == '__main__':
    main()
