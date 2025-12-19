import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_image_grid(dataset, rows, cols, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Visualizes a grid of images with their labels.

    @param: image_label_pairs: A list of tuples, where each tuple is (image, label).
    @param: rows: The number of rows where images are shown.
    @param: cols: The number of columns where images are shown.
    """

    indices = np.random.randint(low=0, high=len(dataset), size=rows*cols)
    example_imgs = [dataset[i] for i in indices]

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    if rows * cols == 1:
        axes_iter = [axes]
    else:
        axes_iter = np.array(axes).flatten()

    for ax, (image, label) in zip(axes_iter, example_imgs):
        # Image is expected as torch tensor CxHxW
        img = image.permute(1, 2, 0).cpu().numpy()
        # Unnormalize images
        img = img * np.array(std)[None, None, :] + np.array(mean)[None, None, :]
        # Clip to [0, 1] range
        img = np.clip(img, 0.0, 1.0)

        ax.imshow(img)

        # Format label values to 2 decimal places
        try:
            if isinstance(label, torch.Tensor):
                vals = label.detach().cpu().numpy().ravel()
            else:
                vals = np.array(label).ravel()
            label_str = ", ".join(f"{float(v):.2f}" for v in vals)
        except Exception:
            label_str = str(label)

        # Set title with wrapping and smaller font to avoid clipping; allow full width by
        # reserving top margin via tight_layout rect below.
        ax.set_title(label_str, wrap=True, fontsize=8)
        ax.axis('off')

    plt.show()


def fen_to_matrix(fen: str) -> np.ndarray:
    """Convert FEN string to 8x8 matrix representation."""
    # Standardize FEN by replacing '-' with '/'
    fen = fen.replace('-', '/')

    piece_to_int = {
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
        'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6
    }
    board = np.zeros((8, 8), dtype=int)
    rows = fen.split(' ')[0].split('/')
    for r, row in enumerate(rows):
        c = 0
        for char in row:
            if char.isdigit():
                c += int(char)
            else:
                board[r, c] = piece_to_int.get(char, 0)
                c += 1
    return board


def matrix_to_fen(board: np.ndarray) -> str:
    """Convert 8x8 matrix representation to FEN string.
    (Only the piece placement part of FEN)"""
    int_to_piece = {
        1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K',
        -1: 'p', -2: 'n', -3: 'b', -4: 'r', -5: 'q', -6: 'k'
    }
    fen_rows = []
    for r in range(8):
        fen_row = ''
        empty_count = 0
        for c in range(8):
            piece = board[r, c]
            if piece == 0:
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += int_to_piece.get(piece, '')
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    fen = '/'.join(fen_rows)
    return fen


def num_to_piece_matrix(num_matrix: np.ndarray) -> np.ndarray:
    """Convert numerical matrix to piece matrix with string representations."""
    int_to_piece = {
        1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K',
        -1: 'p', -2: 'n', -3: 'b', -4: 'r', -5: 'q', -6: 'k',
        0: '.'
    }
    piece_matrix = np.empty(num_matrix.shape, dtype='<U2')
    for r in range(num_matrix.shape[0]):
        for c in range(num_matrix.shape[1]):
            piece_matrix[r, c] = int_to_piece.get(num_matrix[r, c], '.')
    return piece_matrix


