import os

data_dir = 'dataset'
for split in ['train', 'test']:
    split_path = os.path.join(data_dir, split)
    if not os.path.exists(split_path):
        print(f"❌ Missing folder: {split_path}")
        continue

    for label in os.listdir(split_path):
        label_path = os.path.join(split_path, label)
        if os.path.isdir(label_path):
            img_files = [f for f in os.listdir(label_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"{label} ({split}): {len(img_files)} images")
        else:
            print(f"⚠️ Not a directory: {label_path}")
