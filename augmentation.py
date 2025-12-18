from screenshot_augmentation import augment_screenshot

if __name__ == '__main__':
    augment_screenshot(
        input_dir="data/original/train",
        output_dir="data/augmented_train",
        csv_path="data/augmented_train/_annotations.csv",
        max_crop=10,
        max_pad=100,
        scale_range=(0.75, 1.75),
        random_seed=42,
    )
    augment_screenshot(
        input_dir="data/original/test",
        output_dir="data/augmented_test",
        csv_path="data/augmented_test/_annotations.csv",
        max_crop=10,
        max_pad=100,
        scale_range=(0.75, 1.75),
        random_seed=42,
    )