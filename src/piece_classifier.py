import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from tqdm import tqdm

from src.datasets import PieceDataset
from src.utils import visualize_image_grid

def parse_args():
    parser = argparse.ArgumentParser(description="Train an EfficientNet-B0 to predict per-square piece classes (13 classes)")
    # Paths
    parser.add_argument('--img-dir', type=str,
                        default="../data/augmented_pieces")
    parser.add_argument('--model-save-path', type=str,
                        default="../data/models/piece_classifier_efficientnetb0.pth")
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
    parser.add_argument('--square-size', type=int, default=224)

    return parser.parse_args()


def image_transform(image):
    # Resize to 224x224 and normalize as per EfficientNet requirements
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(image)


def load_data(img_dir, batch_size, val_split, random_seed):
    dataset = PieceDataset(img_dir=img_dir, transform=image_transform)

    dataset_size = len(dataset)
    val_size = max(1, int(dataset_size * val_split)) if dataset_size > 1 and val_split > 0 else 0
    train_size = dataset_size - val_size
    generator = torch.Generator().manual_seed(random_seed)

    if val_size > 0:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    else:
        train_dataset = dataset
        val_dataset = []

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_size > 0 else []

    return dataset, train_loader, val_loader


def init_model(device):
    args = parse_args()
    model = models.efficientnet_b0(weights='DEFAULT')
    # Replace classifier to match number of classes (13)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 13)
    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze classifier
    for param in model.classifier.parameters():
        param.requires_grad = True
    # Unfreeze last conv layer
    for param in model.features[7].parameters():
        param.requires_grad = True

    model = model.to(device)

    # Optionally load checkpoint
    if args.model_load_path is not None:
        path = str(args.model_load_path)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    return model


def init_optimizer(model):
    args = parse_args()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    # Optionally load checkpoint
    if args.model_load_path is not None:
        path = str(args.model_load_path)
        checkpoint = torch.load(path)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return optimizer


def save_checkpoint(path, model, optimizer, epoch, val_loss):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_loss,
    }
    torch.save(checkpoint, path)

def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Basic validation
    if not os.path.isdir(args.img_dir):
        raise FileNotFoundError(f"Image directory not found: {args.img_dir}")
    model_save_dir = os.path.dirname(args.model_save_path)
    if model_save_dir and not os.path.isdir(model_save_dir):
        os.makedirs(model_save_dir, exist_ok=True)

    # Build dataset and dataloaders
    dataset, train_loader, val_loader = load_data(
        img_dir=args.img_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        random_seed=args.random_seed,
    )
    # Visualize random samples from the dataset
    visualize_image_grid(dataset, 4, 5)

    # Model init / load
    model = init_model(device)
    # Loss function
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = init_optimizer(model)

    # Training loop with early stopping
    print(f"\nStarting training on {device}...\n")
    best_val = float('inf')
    epochs_no_improve = 0
    for epoch in range(args.epochs):
        # Train model
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Training", ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # Validate model
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Validation", ncols=100):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
        val_loss = val_running_loss / len(val_loader.dataset)

        # Early stopping check
        improved = (best_val - val_loss) > args.min_delta
        if improved:
            best_val = val_loss
            epochs_no_improve = 0
            save_checkpoint(args.model_save_path, model, optimizer, epoch + 1, val_loss)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered. No improvement for {epochs_no_improve} epochs.")
            break

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.7f}, Val Loss: {val_loss:.7f}\n")
    print("\nTraining finished.")


if __name__ == '__main__':
    main()
