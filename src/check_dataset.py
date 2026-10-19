from pathlib import Path
from collections import Counter

DATA_DIR = Path("data/PlantVillage")
SPLIT_DIR = DATA_DIR / "splits"


def read_split(filename):
    split_path = SPLIT_DIR / filename

    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    # Preserve spaces inside filenames.
    return [
        line.strip().replace("\\", "/")
        for line in split_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def resolve_image(relative_path):
    return DATA_DIR / Path(*relative_path.split("/"))


def verify_split(name, paths):
    print(f"\n--- {name} ---")
    print(f"Entries: {len(paths)}")
    print(f"Unique entries: {len(set(paths))}")
    print(f"Duplicate entries: {len(paths) - len(set(paths))}")

    missing = [
        path for path in paths
        if not resolve_image(path).is_file()
    ]

    print(f"Missing images: {len(missing)}")

    if missing:
        print("First missing paths:")
        for path in missing[:5]:
            print(f"  {path}")

    class_counts = Counter(
        path.split("/")[2] for path in paths
    )

    print(f"Classes: {len(class_counts)}")
    print("Images per class:")

    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count}")

    return set(paths)


print("=" * 55)
print("LEAFLENS AI - SPLIT INTEGRITY CHECK")
print("=" * 55)

train_paths = read_split("color_train.txt")
test_paths = read_split("color_test.txt")

train_set = verify_split("TRAINING SET", train_paths)
test_set = verify_split("TEST SET", test_paths)

overlap = train_set.intersection(test_set)

print("\n--- FINAL SUMMARY ---")
print(f"Training entries: {len(train_paths)}")
print(f"Test entries: {len(test_paths)}")
print(f"Overlapping paths: {len(overlap)}")

if overlap:
    print("ERROR: Training and test sets overlap!")
    for path in list(overlap)[:5]:
        print(f"  {path}")

missing_train = sum(
    not resolve_image(path).is_file() for path in train_paths
)
missing_test = sum(
    not resolve_image(path).is_file() for path in test_paths
)

if (
    not overlap
    and missing_train == 0
    and missing_test == 0
    and len(train_paths) == len(train_set)
    and len(test_paths) == len(test_set)
):
    print("\nPASS: All split paths exist and the sets do not overlap.")
else:
    print("\nCHECK FAILED: Review the errors above.")

print("\nVerification complete.")