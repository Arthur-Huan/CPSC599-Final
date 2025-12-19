import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

CLASS_NAMES = [
    '0',
    'P', 'N', 'B', 'R', 'Q', 'K',  # white pieces
    'p', 'n', 'b', 'r', 'q', 'k'   # black pieces
]
CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS = {i: c for c, i in CLASS_TO_INDEX.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ResNet18 to predict chessboard corner coordinates")
    # Paths
    parser.add_argument('--img-dir', type=str,
                        default="../data/augmented_train")
    parser.add_argument('--csv-file', type=str,
                        default="../data/augmented_train/_annotations.csv")
    parser.add_argument('--model-save-path', type=str,
                        default="../models/chessboard_corners_resnet18.pth")
    parser.add_argument('--model-load-path', type=str, default=None)
    # Hyperparameters
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--random-seed', type=int, default=132)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min-delta', type=float, default=1e-4)
    return parser.parse_args()


def calculate_loss(input_matrix, target_matrix):
    """
    Calculate inversely weighted cross-entropy loss for per-square classification.
    Piece numbers are as follows:
    0: empty square,
    -1 to -6: white pieces (P, N, B, R, Q, K),
     1 to  6: black pieces (p, n, b, r, q, k)

    :param input_matrix: np.ndarray of shape (8, 8) of predicted class numbers
    :param target_matrix: np.ndarray of shape (8, 8) of ground truth class numbers
    :return: inversely weighted cross-entropy loss
    """
    empty_weight = 0.1
    pawn_weight = 1.0
    piece_weight = 5.0


    pred = np.asarray(input_matrix)
    tgt = np.asarray(target_matrix)

    # Ensure integer class labels
    pred = pred.astype(np.int32)
    tgt = tgt.astype(np.int32)

    # Build weight matrix based on target class (we weight by target frequency)
    weights = np.zeros_like(tgt, dtype=float)
    weights[tgt == 0] = empty_weight
    weights[np.isin(tgt, [1, -1])] = pawn_weight
    other_piece_ids = [2, 3, 4, 5, 6, -2, -3, -4, -5, -6]
    weights[np.isin(tgt, other_piece_ids)] = piece_weight

    # Error per square
    incorrect = (pred != tgt).astype(float)
    weighted_errors = weights * incorrect
    total_weight = weights.sum()

    # Avoid division by zero (shouldn't happen)
    if total_weight == 0:
        return 0.0

    loss = weighted_errors.sum() / total_weight
    return float(loss)
