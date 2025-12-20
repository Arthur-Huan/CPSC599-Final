import argparse

import torch
from torchvision import models
from tqdm import tqdm
import csv
import os

from src.utils import visualize_image_grid
from src.datasets import PieceDataset
from src import piece_name_mapping
from src.piece_classifier import image_transform as piece_image_transform


def parse_args():
    parser = argparse.ArgumentParser(description="Run tests for chess-vision components")
    # Modes
    parser.add_argument('--mode', type=str, choices=['piece', 'fen'], default='fen',
                        help="Which test to run: 'piece' for piece classifier test, 'fen' for FEN generator test")
    # Paths
    parser.add_argument('--img-dir', type=str,
                        default="data/test_pieces")
    parser.add_argument('--corner_localizer_path', type=str,
                        default="models/corner_localizer.pth")
    parser.add_argument('--piece_classifier_path', type=str,
                        default="models/piece_classifier.pth")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default=None,
                        help='torch device to use, e.g. cpu or cuda')
    return parser.parse_args()


def test_piece_classifier():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    piece_data = PieceDataset(img_dir=args.img_dir, transform=piece_image_transform)

    visualize_image_grid(piece_data, 4, 5)

    # Load model
    model = models.efficientnet_b0(num_classes=13)
    # Load checkpoint and get state_dict (supports both plain state_dict or dict with 'model_state_dict')
    ckpt = torch.load(args.piece_classifier_path, map_location=device)
    sd = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    
    dataloader = torch.utils.data.DataLoader(piece_data, batch_size=args.batch_size, shuffle=False)

    total = 0
    correct = 0
    # Per-class counts (13 classes: 0..12)
    per_class_total = [0] * 13
    per_class_correct = [0] * 13

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing Piece Classifier"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            for t, p in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                per_class_total[t] += 1
                if p == t:
                    per_class_correct[t] += 1

    accuracy = correct / total if total > 0 else 0.0

    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Overall accuracy: {accuracy:.4f}")

    # Print per-class accuracy with token names
    for i in range(13):
        tot = per_class_total[i]
        corr = per_class_correct[i]
        acc = (corr / tot) if tot > 0 else 0.0
        token = piece_name_mapping.number_to_token_style(i)
        print(f"Class {i:2d} ({token}): {corr}/{tot} -> {acc:.3f}")


def _expand_fen_board(fen_board: str):
    """Expand a FEN board row-string (8 ranks separated by '/') into a list of 64 squares.
    Uses '.' to denote empty squares. Keeps piece letters as-is.
    """
    squares = []
    ranks = fen_board.strip().split('/')
    for rank in ranks:
        for ch in rank:
            if ch.isdigit():
                # number of empty squares
                squares.extend(['.'] * int(ch))
            else:
                squares.append(ch)
    return squares


def _filename_to_fen(filename: str) -> str:
    """Process filename into FEN-like board string by removing extension and replacing '-' with '/'.
    Assumes the name encodes only the board part of FEN. Example: 'rnbqkbnr-p1p1....png' -> 'rnbqkbnr/p1p1....'
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    return base.replace('-', '/')


def test_fen_generator():
    """Run fen_generator.py (no args), read fen_results.csv, and report FEN accuracy and per-square accuracy.
    Also prints per-class mistake rates for each ground-truth square symbol (including empty '.')."""
    # Paths relative to this file (src/test.py)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
    fen_generator_path = os.path.join(repo_root, 'fen_generator.py')
    fen_csv_path = os.path.join(repo_root, 'fen_results.csv')

    # 1) Run fen_generator
    pass  # Manually run by hand for now

    # 2) Read fen_results.csv and compute metrics
    if not os.path.exists(fen_csv_path):
        raise FileNotFoundError(f"Expected CSV at {fen_csv_path}, but it was not found.")

    total = 0
    fen_exact_match = 0

    per_square_total = 0  # total squares compared (should be 64 * samples with comparable FENs)
    per_square_correct = 0

    # Per-class mistake tracking (by ground-truth symbol)
    per_class_total = {}
    per_class_mistakes = {}

    with open(fen_csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        if 'filename' not in reader.fieldnames or 'fen' not in reader.fieldnames:
            raise ValueError("fen_results.csv must contain 'filename' and 'fen' columns")
        for row in reader:
            filename = row.get('filename', '')
            pred_board_str = row.get('fen', '')
            # interpret filename into FEN board string
            gt_board_str = _filename_to_fen(filename)

            # FEN accuracy (exact board string match)
            total += 1
            if gt_board_str == pred_board_str:
                fen_exact_match += 1

            # Per-square accuracy
            gt_squares = _expand_fen_board(gt_board_str)
            pred_squares = _expand_fen_board(pred_board_str)
            # Only compare if both expand to 64 squares
            if len(gt_squares) == 64 and len(pred_squares) == 64:
                for g, p in zip(gt_squares, pred_squares):
                    per_square_total += 1
                    if g == p:
                        per_square_correct += 1
                    # track per-class totals and mistakes by ground-truth symbol
                    per_class_total[g] = per_class_total.get(g, 0) + 1
                    if g != p:
                        per_class_mistakes[g] = per_class_mistakes.get(g, 0) + 1

    fen_accuracy = (fen_exact_match / total) if total > 0 else 0.0
    per_square_accuracy = (per_square_correct / per_square_total) if per_square_total > 0 else 0.0

    print(f"FEN accuracy (exact board match): {fen_accuracy:.4f} ({fen_exact_match}/{total})")
    print(f"Per-square accuracy: {per_square_accuracy:.4f} ({per_square_correct}/{per_square_total})")

    # Per-class mistake rates
    if per_class_total:
        print("Per-square mistake rate by ground-truth class (lowercase=black, uppercase=white, '.'=empty):")
        # sort classes by mistake rate descending, then by class name
        items = []
        for cls, tot in per_class_total.items():
            mistakes = per_class_mistakes.get(cls, 0)
            rate = (mistakes / tot) if tot > 0 else 0.0
            items.append((cls, mistakes, tot, rate))
        items.sort(key=lambda x: (-x[3], x[0]))
        for cls, mistakes, tot, rate in items:
            print(f"  {cls}: mistakes {mistakes}/{tot} -> {rate:.4f}")


if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'fen':
        test_fen_generator()
    else:
        test_piece_classifier()
