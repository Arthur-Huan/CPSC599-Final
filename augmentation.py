from screenshot_augmentation import augment_screenshot

if __name__ == '__main__':
    augment_screenshot(
        input_dir="data/original/test",
        output_dir="data/augmented",
        csv_path="data/augmented/annotations.csv",
        max_crop=10,
        max_pad=100,
        scale_range=(0.75, 1.75),
        random_seed=42,
    )
