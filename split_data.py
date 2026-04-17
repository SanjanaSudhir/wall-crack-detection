import os, shutil, random

# ── Config ──────────────────────────────
RAW_DIR    = "./raw_data"
OUTPUT_DIR = "./data"
SPLIT      = (0.8, 0.1, 0.1)   # train / val / test
SEED       = 42
# ────────────────────────────────────────

random.seed(SEED)

splits = ["train", "val", "test"]

for class_name in ["cracked", "not_cracked"]:
    images = os.listdir(os.path.join(RAW_DIR, class_name))
    images = [f for f in images if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(images)

    total       = len(images)
    train_end   = int(total * SPLIT[0])
    val_end     = train_end + int(total * SPLIT[1])

    split_images = {
        "train": images[:train_end],
        "val":   images[train_end:val_end],
        "test":  images[val_end:]
    }

    for split, files in split_images.items():
        dest = os.path.join(OUTPUT_DIR, split, class_name)
        os.makedirs(dest, exist_ok=True)
        for f in files:
            shutil.copy(
                os.path.join(RAW_DIR, class_name, f),
                os.path.join(dest, f)
            )
        print(f"{class_name} → {split}: {len(files)} images")

print("\nSplit complete ✓")