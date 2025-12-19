from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
import pandas as pd
import os
import random
import string
import time
from typing import Tuple, Optional
from tqdm import tqdm
import io
import cairosvg


def augment_screenshot(
    input_dir: str,
    output_dir: str,
    csv_path: str,
    max_crop: int = 10,
    max_pad: int = 100,
    scale_range: Tuple[float, float] = (0.6, 1.6),
    random_seed: Optional[int] = None,
):
    """
    Augment the folder of input images to look more like screenshots.
    Images will be randomly cropped/padded, on each side, unevenly, and then resized.
    The coordinates of the four corners resulting will be saved to a CSV file for training.

    :param input_dir: folder containing input images
    :param output_dir: folder where augmented images will be written
    :param csv_path: CSV output path
    :param max_crop: maximum pixels to crop on any side (non-negative)
    :param max_pad:  maximum pixels to pad don any side (non-negative)
    :param scale_range: (min_scale, max_scale) scale factor applied (preserves aspect ratio)
    :param random_seed
    :return: None, writes augmented images and CSV file
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    os.makedirs(output_dir, exist_ok=True)
    rows = []

    fnames = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower()]
    iterator = tqdm(fnames, desc='Augmenting')

    for fname in iterator:
        in_path = os.path.join(input_dir, fname)
        try:
            img = Image.open(in_path).convert('RGB')
        except Exception as e:
            print(f"Warning: cannot open {in_path}: {e}")
            continue

        width, height = img.size

        # Sample signed offsets, negative = crop, positive = pad
        left_off = random.randint(-max_crop, max_pad)
        top_off = random.randint(-max_crop, max_pad)
        right_off = random.randint(-max_crop, max_pad)
        bottom_off = random.randint(-max_crop, max_pad)

        # convert to crop / pad non-negative values
        crop_left = max(0, -left_off)
        crop_right = max(0, -right_off)
        crop_top = max(0, -top_off)
        crop_bottom = max(0, -bottom_off)

        pad_left = max(0, left_off)
        pad_right = max(0, right_off)
        pad_top = max(0, top_off)
        pad_bottom = max(0, bottom_off)

        # Apply cropping
        left = crop_left
        right = max(1, width - crop_right)
        top = crop_top
        bottom = max(1, height - crop_bottom)
        cropped = img.crop((left, top, right, bottom))

        # Random fill color for padding to simulate website background
        fill = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)

        # Apply padding
        if any([pad_left, pad_top, pad_right, pad_bottom]):
            augmented = ImageOps.expand(
                cropped, border=(pad_left, pad_top, pad_right, pad_bottom), fill=fill)
        else:
            augmented = cropped

        aug_w, aug_h = augmented.size

        # Coordinates of image corners before resizing
        tl_x_pre = 0 - crop_left + pad_left
        tl_y_pre = 0 - crop_top + pad_top
        tr_x_pre = width - crop_left + pad_left
        tr_y_pre = 0 - crop_top + pad_top
        br_x_pre = width - crop_left + pad_left
        br_y_pre = height - crop_top + pad_top
        bl_x_pre = 0 - crop_left + pad_left
        bl_y_pre = height - crop_top + pad_top

        # Choose final scale factor
        s = random.uniform(scale_range[0], scale_range[1])
        final_w = max(1, int(round(aug_w * s)))
        final_h = max(1, int(round(aug_h * s)))

        # Resize preserving aspect ratio by scaling both dimensions by s
        final = augmented.resize((final_w, final_h), resample=Image.Resampling.BICUBIC)

        # Coordinates in final image
        tl_x = int(round(tl_x_pre * s))
        tl_y = int(round(tl_y_pre * s))
        tr_x = int(round(tr_x_pre * s))
        tr_y = int(round(tr_y_pre * s))
        br_x = int(round(br_x_pre * s))
        br_y = int(round(br_y_pre * s))
        bl_x = int(round(bl_x_pre * s))
        bl_y = int(round(bl_y_pre * s))

        # Build filename and save (always as JPEG)
        fmain, _ = os.path.splitext(fname)
        out_fname = f"{fmain}.jpg"
        out_path = os.path.join(output_dir, out_fname)
        final.save(out_path, format='JPEG', quality=95, subsampling=0)

        row = {
            'file': out_fname,
            'width': final_w,
            'height': final_h,
            'topleft_x': int(tl_x),
            'topleft_y': int(tl_y),
            'topright_x': int(tr_x),
            'topright_y': int(tr_y),
            'botleft_x': int(bl_x),
            'botleft_y': int(bl_y),
            'botright_x': int(br_x),
            'botright_y': int(br_y),
        }
        rows.append(row)

    # Save CSV
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(rows)} augmented images' metadata to {csv_path}")
    else:
        print("No augmented images were produced (no input images found).")


def augment_piece(
    input_dir: str,
    output_dir: str,
    copies_per_piece: int = 10,
    augment=True,
):
    """
    Have a 1/13 chance to just output a blank square.
    Otherwise:
    Read piece svg files from input_dir and augment them before saving to output_dir.
    Mandatory step includes adding a background color, as pieces appear on a board.
    Then, some color shifts, scaling, etc. can be applied if augment=True.

    :param input_dir: Folder containing input svg files
        Filenames are in the format <piece><color>.svg, e.g. "rb.svg", "qw.svg"
    :param output_dir: Folder where augmented svg files will be written
    :param copies_per_piece: Number of augmented copies to create per input piece
    :param augment: Whether to apply augmentation or just apply a background and save
    :return: None, writes augmented svg files
    """
    os.makedirs(output_dir, exist_ok=True)


    # Collect all svg files under input_dir (walk subfolders) and preserve relative paths
    svg_items = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith('.svg'):
                svg_items.append((root, f))

    for root, fname in tqdm(svg_items, desc='Augmenting pieces'):
        in_path = os.path.join(root, fname)
        with open(in_path, 'r', encoding='utf-8') as f:
            svg_content = f.read()

        fmain, _ = os.path.splitext(fname)

        for _ in range(copies_per_piece):
            # Randomly output a blank square (random color) with 1/13 probability
            if random.randint(1, 13) == 1:
                bg_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                blank_img = Image.new('RGB', (100, 100), bg_color)
                suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
                out_fname = f"empty_{suffix}.png"
                out_path = os.path.join(output_dir, out_fname)
                blank_img.save(out_path, format='PNG')

            # Output the piece
            png_bytes = cairosvg.svg2png(bytestring=svg_content.encode('utf-8'))
            img = Image.open(io.BytesIO(png_bytes))
            img_rgba = img.convert('RGBA')

            if augment:
                augmented = augment_image(img_rgba)
            else:
                augmented = img_rgba

            bg_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            aug_mode = augmented.mode
            bg = Image.new('RGB', augmented.size, bg_color)

            if aug_mode == 'RGBA':
                alpha = augmented.split()[-1]
                bg.paste(augmented.convert('RGB'), (0, 0), mask=alpha)
                final = bg
            else:
                final = augmented.convert('RGB')

            # Generate a random 6-character alphanumeric suffix to avoid filename clashes
            # In the unlikely event of a clash, overwriting is fine
            suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
            out_fname = f"{fmain}_{suffix}.png"
            out_path = os.path.join(output_dir, out_fname)

            final.save(out_path, format='PNG')


def augment_image(img):
    # Ensure RGBA so we keep/propagate alpha
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    # We cannot change the color because white/black matters in chess

    # Flip
    if random.random() < 0.5:
        img = ImageOps.mirror(img)

    # Blur
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))

    # Noise (preserve alpha)
    if random.random() < 0.3:
        arr = np.asarray(img).astype(np.float32)  # shape (H, W, 4)
        sigma = random.uniform(3.0, 12.0)
        noise = np.random.normal(0.0, sigma, size=arr[..., :3].shape).astype(np.float32)
        arr[..., :3] += noise
        arr[..., :3] = np.clip(arr[..., :3], 0, 255)
        arr = arr.astype(np.uint8)
        img = Image.fromarray(arr, mode='RGBA')

    return img


if __name__ == '__main__':
    augment_piece(
        input_dir="../data/pieces",
        output_dir="../data/augmented_pieces",
        copies_per_piece=5,
        augment=True,
    )
    '''    augment_screenshot(
        input_dir="../data/original/train",
        output_dir="../data/augmented_screenshots/train",
        csv_path="../data/augmented_screenshots/train/_annotations.csv",
        max_crop=10,
        max_pad=100,
        scale_range=(0.75, 1.75),
        random_seed=42,
    )
    '''
    '''
    augment_screenshot(
        input_dir="../data/original/test",
        output_dir="../data/augmented_screenshots/test",
        csv_path="../data/augmented_screenshots/test/_annotations.csv",
        max_crop=10,
        max_pad=100,
        scale_range=(0.75, 1.75),
        random_seed=42,
    )
    '''
