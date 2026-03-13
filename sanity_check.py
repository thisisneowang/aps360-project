from pathlib import Path

'''
This assumes a fixed file directory structure of 
project/
└── data/
    └── raw/
        └── im2latex/
            └── formula_images/
'''
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "im2latex"
IMG_DIR = DATA_ROOT / "formula_images"

''' 
Removes all newline '\n' characters from a file containing formulas
End result contains a list of formula strings
'''
def load_formulas(path):
    # try three possible file encodings
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(path, "r", encoding=enc, newline="\n") as f:
                formulas = [line.rstrip("\n") for line in f]
            print(f"Loaded formulas with encoding: {enc}")
            return formulas
        except UnicodeDecodeError:
            pass
    raise RuntimeError("Could not decode formulas file.")

'''
Creates a list of formatted strings (either into two or three parts)
for matching later.
'''
def load_split(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            if len(parts) == 2:
                # This would be in im2markup-style: <img_name> <label_idx>
                img_name, label_idx = parts
                render_type = None

            elif len(parts) == 3:
                # Otherwise, the Zenodo/Miffyli-style: <formula_idx> <image_name> <render_type>
                label_idx, img_name, render_type = parts

                # Note: the Miffyli format stores image name without ".png"
                if not img_name.endswith(".png"):
                    img_name = img_name + ".png"

            else:
                raise ValueError(f"Unexpected line format in {path}:\n{line}")

            samples.append((img_name, int(label_idx), render_type))

    return samples

# Gather all training, validation, and testing samples for accumulation later
formulas = load_formulas(DATA_ROOT / "im2latex_formulas.lst")
train = load_split(DATA_ROOT / "im2latex_train.lst")
val = load_split(DATA_ROOT / "im2latex_validate.lst")
test = load_split(DATA_ROOT / "im2latex_test.lst")  

print("num formulas:", len(formulas))
print("train:", len(train), "val:", len(val), "test:", len(test))

all_samples = [("train", *x) for x in train] + [("val", *x) for x in val] + [("test", *x) for x in test]

missing_images = []
bad_indices = []

# Ensure that every image can be matched to a corresponding formula, and that
# there are no out-of-range label IDs
for split, img_name, label_idx, render_type in all_samples:
    if not (IMG_DIR / img_name).exists():
        missing_images.append((split, img_name, render_type))
    if not (0 <= label_idx < len(formulas)):
        bad_indices.append((split, img_name, label_idx, render_type))

print("missing images:", len(missing_images))
print("bad label indices:", len(bad_indices))

print("\nExample:")
img_name, label_idx, render_type = train[0]
print("image:", IMG_DIR / img_name)
print("render_type:", render_type)
print("latex:", formulas[label_idx])
