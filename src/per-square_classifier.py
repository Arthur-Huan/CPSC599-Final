import os
import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import models
from tqdm import tqdm

from src.datasets import BoardDataset

CLASS_NAMES = [
    '0',
    'P', 'N', 'B', 'R', 'Q', 'K',  # white pieces
    'p', 'n', 'b', 'r', 'q', 'k'   # black pieces
]
CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS = {i: c for c, i in CLASS_TO_INDEX.items()}

DEFAULT_EMPTY_WEIGHT = 0.1
DEFAULT_PAWN_WEIGHT = 1.0
DEFAULT_PIECE_WEIGHT = 5.0


def parse_args():
    parser = argparse.ArgumentParser(description="Train an EfficientNet-B0 to predict per-square piece classes (13 classes)")
    # Paths
    parser.add_argument('--img-dir', type=str,
                        default="../data/augmented_train")
    parser.add_argument('--model-save-path', type=str,
                        default="../models/per_square_effb0.pth")
    parser.add_argument('--model-load-path', type=str, default=None)
    # Hyperparameters
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Number of images per batch (each image contains 64 squares -> effective batch size for squares = batch_size*64)')
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--random-seed', type=int, default=132)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min-delta', type=float, default=1e-4)
    parser.add_argument('--square-size', type=int, default=224)
    # Loss weights to handle class imbalance
    parser.add_argument('--empty-weight', type=float, default=DEFAULT_EMPTY_WEIGHT)
    parser.add_argument('--pawn-weight', type=float, default=DEFAULT_PAWN_WEIGHT)
    parser.add_argument('--piece-weight', type=float, default=DEFAULT_PIECE_WEIGHT)

    return parser.parse_args()



def logits_to_class_indices(logits: torch.Tensor):
    return torch.argmax(logits, dim=1)


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Basic validation
    if not os.path.isdir(args.img_dir):
        raise FileNotFoundError(f"Image directory not found: {args.img_dir}")
    model_save_dir = os.path.dirname(args.model_save_path)
    if model_save_dir and not os.path.isdir(model_save_dir):
        os.makedirs(model_save_dir, exist_ok=True)

    # Create dataset using BoardDataset & resize each square to args.square_size
    dataset = BoardDataset(img_dir=args.img_dir, transform=None, square_size=args.square_size)

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
    model = models.efficientnet_b0(weights='DEFAULT')

    # Replace classifier to match number of classes (13)
    try:
        # common torchvision structure: model.classifier = nn.Sequential(Dropout, Linear)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
    except Exception:
        # fallback: classifier might be a single Linear
        try:
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, len(CLASS_NAMES))
        except Exception:
            raise RuntimeError("Unexpected EfficientNet classifier structure; cannot replace final layer to 13 classes")

    # freeze all params then enable head params
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    model = model.to(device)

    # Build class weights tensor that mirrors the weights used in calculate_loss
    # class index mapping: 0=empty, 1..6 white pieces (P..K), 7..12 black pieces (p..k)
    class_weights = torch.zeros(len(CLASS_NAMES), dtype=torch.float)
    class_weights[0] = args.empty_weight
    # pawns: class 1 (P) and class 7 (p)
    class_weights[1] = args.pawn_weight
    class_weights[7] = args.pawn_weight
    # other pieces
    for idx in [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]:
        class_weights[idx] = args.piece_weight
    class_weights = class_weights.to(device)

    # Use weighted CrossEntropyLoss for training (differentiable)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Create optimizer after model is constructed so its param references match checkpoint
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

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
        num_batches = 0
        for squares_batch, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", unit="batch", leave=False):
            # squares_batch: (B, 64, C, H, W), labels_batch: (B, 64)
            B = squares_batch.shape[0]
            C = squares_batch.shape[2]
            H = squares_batch.shape[3]
            W = squares_batch.shape[4]
            inputs = squares_batch.view(B * 64, C, H, W).to(device)
            targets = labels_batch.view(B * 64).to(device)

            outputs = model(inputs)  # (B*64, 13)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        train_loss = running_loss / (num_batches if num_batches > 0 else 1)

        # Validation
        model.eval()
        val_running = 0.0
        val_batches = 0
        with torch.no_grad():
            for squares_batch, labels_batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", unit="batch", leave=False):
                B = squares_batch.shape[0]
                C = squares_batch.shape[2]
                H = squares_batch.shape[3]
                W = squares_batch.shape[4]
                inputs = squares_batch.view(B * 64, C, H, W).to(device)
                targets = labels_batch.view(B * 64).to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_running += loss.item()
                val_batches += 1
        val_loss = val_running / (val_batches if val_batches > 0 else 1)

        # Early stopping / checkpointing
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

        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered. No improvement for {epochs_no_improve} epochs.")
            break

        print(f"\n  Epoch {epoch+1}, Train Loss: {train_loss:.7f}, Val Loss: {val_loss:.7f}")

    print("Training finished.")


if __name__ == '__main__':
    main()
