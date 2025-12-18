from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import os
import random
from typing import Tuple, Optional
from tqdm import tqdm


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
