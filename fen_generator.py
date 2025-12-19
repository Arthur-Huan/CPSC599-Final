import csv
import os
import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

from src.utils import matrix_to_fen


def parse_args():
    parser = argparse.ArgumentParser(description="Generate FEN strings from chess board images")
    # Paths
    parser.add_argument('--img-dir', type=str,
                        default="data/original/test")
    parser.add_argument('--corner-localizer-path', type=str,
                    default='../models/corner_localizer.pth')
    parser.add_argument('--piece-classifier-path', type=str,
                        default='models/piece_classifier.pth')
    parser.add_argument('--output-file', type=str,
                        default='fen_results.csv')

    return parser.parse_args()


def load_piece_classifier(model_path, device):
    """Load the trained piece classifier model.

    :param model_path: Path to checkpoint file
    :param device: Torch device
    :return: Loaded classifier in eval mode
    """
    model = models.efficientnet_b0(weights='DEFAULT')
    # Replace classifier to match number of classes (13)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, 13)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    model.eval()

    return model


def image_transform():
    """Transform used during training."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])


def split_board_into_squares(image):
    """Split a chess board into 64 squares.

    :param image: PIL image of the board
    :return: List of 64 PIL images ordered rank 8 to rank 1, file a to h
    """
    width, height = image.size
    square_width = width / 8
    square_height = height / 8

    squares = []

    # Iterate from top to bottom (rank 8 to rank 1)
    for row in range(8):
        # Iterate from left to right (files a-h)
        for col in range(8):
            left = int(col * square_width)
            top = int(row * square_height)
            right = int((col + 1) * square_width)
            bottom = int((row + 1) * square_height)

            square = image.crop((left, top, right, bottom))
            squares.append(square)

    return squares


def predict_board_fen(image_path, model, transform, device):
    """Generate the piece-placement FEN for a board image.

    :param image_path: Path to the board image
    :param model: Trained piece classifier
    :param transform: Transform to apply to squares
    :param device: Torch device
    :return: FEN string (piece placement only)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')

    # Split into 64 squares
    squares = split_board_into_squares(image)

    # Transform all squares and stack into a batch
    transformed_squares = [transform(square) for square in squares]
    batch = torch.stack(transformed_squares).to(device)

    # Predict classes for all squares
    with torch.no_grad():
        logits = model(batch)  # Shape: (64, 13)
        probabilities = F.softmax(logits, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)  # Shape: (64,)

    # Convert predictions to 8x8 board matrix
    board_matrix = predicted_classes.cpu().numpy().reshape(8, 8)

    # Convert matrix to FEN string
    fen = matrix_to_fen(board_matrix)

    return fen


def show_squares_in_order(squares, title):
    """Display squares in provided order (index 0 = a8).

    :param squares: List of 64 PIL images
    :param title: Figure title
    :return: None
    """
    plt.figure(figsize=(8, 8))
    for idx, square in enumerate(squares):
        ax = plt.subplot(8, 8, idx + 1)
        ax.imshow(square)
        ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Validate paths
    if not os.path.isdir(args.img_dir):
        raise FileNotFoundError(f"Image directory not found: {args.img_dir}")
    if not os.path.isfile(args.piece_classifier_path):
        raise FileNotFoundError(f"Model file not found: {args.piece_classifier_path}")

    # Load the piece classifier model
    print(f"Loading piece classifier from: {args.piece_classifier_path}")
    model = load_piece_classifier(args.piece_classifier_path, device)

    # Get transform
    transform = image_transform()

    # Find all image files in the directory
    img_dir_path = Path(args.img_dir)
    image_files = sorted(
        list(img_dir_path.glob('*.png')) +
        list(img_dir_path.glob('*.PNG')) +
        list(img_dir_path.glob('*.jpg')) +
        list(img_dir_path.glob('*.JPG')) +
        list(img_dir_path.glob('*.jpeg')) +
        list(img_dir_path.glob('*.JPEG'))
    )

    if not image_files:
        print(f"No image files found in {args.img_dir}")
        return

    print(f"Found {len(image_files)} image files")

    # Visualize a sample board's squares (order preserved: a8..h1)
    sample_path = random.choice(image_files)
    sample_img = Image.open(sample_path).convert('RGB')
    sample_squares = split_board_into_squares(sample_img)
    show_squares_in_order(sample_squares, f"Squares: {sample_path.name}")

    # Process each image and collect results
    results = []

    for img_path in tqdm(image_files, desc="Generating FENs"):
        try:
            fen = predict_board_fen(img_path, model, transform, device)
            filename = img_path.name
            results.append((filename, fen))
        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")
            continue

    # Write results to output file
    with open(args.output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'fen'])
        writer.writerows(results)

    print(f"Successfully processed {len(results)} images")
    print(f"\nResults written to: {args.output_file}")


if __name__ == '__main__':
    main()
