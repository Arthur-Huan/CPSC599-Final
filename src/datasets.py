import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


def _default_per_square_transform():
    # Default transform: ToTensor + ImageNet normalization suitable for EfficientNet-B0
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# mapping consistent with CLASS_NAMES in per-square_classifier:
# index 0 = empty, 1-6 = P,N,B,R,Q,K (white uppercase), 7-12 = p,n,b,r,q,k (black lowercase)
_PIECE_TO_INDEX = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12,
}


def fen_like_to_label_indices(fen_like: str):
    """
    Parse a FEN-like string (supports '/' or '-' as rank separators) and return
    a torch.LongTensor of shape (64,) where each element is in [0..12]
    according to _PIECE_TO_INDEX mapping (0 == empty).

    The function is permissive: it extracts the first token (before any space)
    and accepts rank separators '/' or '-'. Ranks should expand to 8 files using
    digits for consecutive empty squares as in FEN.
    """
    # take the first token before spaces (in case full FEN with side-to-move exists)
    token = str(fen_like).split()[0]

    # normalize separators to '/'
    if '-' in token and '/' not in token:
        ranks = token.split('-')
    else:
        ranks = token.split('/')

    if len(ranks) != 8:
        # try to heuristically pad or handle single-rank filenames
        # fallback: return all-empty
        return torch.zeros(64, dtype=torch.long)

    squares = []
    for rank in ranks:
        i = 0
        while i < len(rank):
            ch = rank[i]
            if ch.isdigit():
                # may be multiple digits e.g., '10' (unlikely), accumulate full number
                j = i
                while j < len(rank) and rank[j].isdigit():
                    j += 1
                num = int(rank[i:j])
                squares.extend([0] * num)
                i = j
            else:
                squares.append(_PIECE_TO_INDEX.get(ch, 0))
                i += 1
        # pad if rank shorter/longer than 8
        if len(squares) % 8 != 0:
            # try to adjust: if rank produced fewer than 8, pad empties
            current_in_rank = len(squares) % 8
            if current_in_rank < 8:
                squares.extend([0] * (8 - current_in_rank))
            else:
                # truncate extras
                squares = squares[:-(current_in_rank - 8)]

    if len(squares) != 64:
        # fallback to empty board if parsing failed
        squares = [0] * 64

    return torch.tensor(squares, dtype=torch.long)


class CornersDataset(Dataset):
    """
    Dataset that returns (image, labels) for each row in the CSV.

    CSV format (columns): filename, width, height,
        tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y

    :return: torch.Dataset. Each datum is tuple of (image, labels) where:
        image: PIL Image (or the transformed image if `transform` is provided)
        labels: torch.Tensor of shape (4,) containing normalized coordinates
                [tl_x, tl_y, br_x, br_y]. Coordinates are normalized to [0, 1]
                by dividing x values by the original image width and y values
                by the original image height.
    """
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


class BoardDataset(Dataset):
    """
    Dataset that returns (squares_tensor, labels_tensor) for each row in the CSV.

    Each image is cropped into an 8x8 grid of equal squares. Each square is
    transformed with `per_square_transform` (if provided) or a sensible default
    (ToTensor + ImageNet normalization). The returned tensor has shape
    (64, C, H, W) where H and W depend on the transform or the crop size.

    CSV format (columns): filename, width, height,
        tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y

    The FEN for an image is in the base filename without its extension.

    :return: torch.Dataset. Each datum is tuple of (squares_tensor, labels_tensor) where:
        squares_tensor: torch.Tensor shaped (64, C, H, W)
        labels_tensor: torch.LongTensor shaped (64,) with class indices 0..12
    """

    def __init__(self, img_dir, transform=None, per_square_transform=None, square_size=None):
        """
        Initialize BoardDataset by scanning `img_dir` for image files.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.per_square_transform = per_square_transform or _default_per_square_transform()
        self.square_size = square_size

        # List files in the provided directory and keep common image extensions
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        files = [f for f in os.listdir(self.img_dir) if os.path.isfile(os.path.join(self.img_dir, f))]
        # filter by extension
        files = [f for f in files if os.path.splitext(f)[1].lower() in exts]
        # sort for deterministic order
        files.sort()
        self.filenames = files

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # derive FEN from filename by stripping directory and extension
        fen = os.path.splitext(os.path.basename(str(filename)))[0]

        # Slice image into 8x8 equal squares
        orig_w, orig_h = image.size
        # compute grid boundaries using float division then round to ints
        x_grid = [int(round(orig_w * (i / 8.0))) for i in range(9)]
        y_grid = [int(round(orig_h * (i / 8.0))) for i in range(9)]

        squares = []
        for r in range(8):
            for c in range(8):
                left = x_grid[c]
                upper = y_grid[r]
                right = x_grid[c + 1]
                lower = y_grid[r + 1]
                # safety checks
                if right <= left or lower <= upper:
                    # fallback to minimal 1px crop
                    right = max(right, left + 1)
                    lower = max(lower, upper + 1)
                crop = image.crop((left, upper, right, lower))
                # Resize to desired square size
                if self.square_size is not None:
                    if isinstance(self.square_size, int):
                        crop = crop.resize((self.square_size, self.square_size), Image.BILINEAR)
                    else:
                        crop = crop.resize((self.square_size[0], self.square_size[1]), Image.BILINEAR)

                # apply per-square transform (ToTensor + Normalize by default)
                sq = self.per_square_transform(crop)
                squares.append(sq)

        # stack into tensor (64, C, H, W)
        squares_tensor = torch.stack(squares, dim=0)

        # parse fen-like filename into labels tensor (64,)
        labels_tensor = fen_like_to_label_indices(fen)

        return squares_tensor, labels_tensor
