import os
import re
from PIL import Image
import torch
from torchvision import models, transforms
from collections import defaultdict, Counter
import json
import re
from collections import defaultdict

# Paths
root = "src/data/train"
label_file = "categories_places365.txt"
checkpoint_url = "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"

# Load model
model = models.resnet18(num_classes=365)
checkpoint = torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu')
state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load class labels
with open(label_file) as f:
    classes = [line.strip().split(' ')[0][3:] for line in f]

# Initialize data structures
all_files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".png")]

# Modified loop to collect indices per class
cache_file = "train_class_to_indices.json"

# Check if cache exists
if os.path.exists(cache_file):
    print("Loading cached class_to_indices...")
    with open(cache_file, "r") as f:
        class_to_indices = json.load(f)
else:
    print("Cache not found. Processing images...")
    class_counts = Counter()
    class_to_indices = defaultdict(list)
    # Regex to extract the numeric part from filename
    pattern = re.compile(r"(\d{6})")
    # Modified loop to collect indices per class
    for path in all_files:
        try:
            img = Image.open(path).convert("RGB")
            input_tensor = preprocess(img).unsqueeze(0)
            logits = model(input_tensor)
            probs = torch.nn.functional.softmax(logits[0], dim=0)
            label = classes[probs.argmax().item()]
            class_counts[label] += 1

            match = pattern.search(path)
            if match:
                img_id = match.group(1)
                class_to_indices[label].append(img_id)

        except Exception as e:
            print(f"Failed on {path}: {e}")

    # Convert defaultdict to regular dict before saving
    with open(cache_file, "w") as f:
        json.dump(dict(class_to_indices), f, indent=2)
    print(f"Saved class_to_indices to {cache_file}")


# Choose the category you want
target_class = "dorm_room"
output_file = f"{target_class}_files_test.txt"

# Write matching file names to output
with open(output_file, "w") as f:
    for idx in sorted(class_to_indices.get(target_class, [])):
        f.write(f"sample_{idx}_rgb.png sample_{idx}_depth.npy\n")

print(f"Saved list for class '{target_class}' to {output_file}")