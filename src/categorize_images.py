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



    groups = {
        "Sleeping":
        [
        "bedroom", "dorm_room", "hotel_room", "youth_hostel", "childs_room",
        "nursery", "berth", "alcove", "bedchamber"
        ],
        "Living":
        [
        "living_room", "television_room", "home_theater", "recreation_room",
        "playroom", "bow_window/indoor", "balcony/interior", "movie_theater/indoor",
        "music_studio", "porch", "patio", "reception", "corridor", "lobby", "entrance_hall", "courtyard"
        ],
        "Kitchen":
        [
        "kitchen", "dining_room", "pantry", "galley", "wet_bar", "restaurant",
        "coffee_shop", "sushi_bar", "restaurant_kitchen", "ice_cream_parlor",
        "bakery/shop", "fastfood_restaurant", "dining_hall", "cafeteria"
        ],
        "Work":
        [
        "home_office", "office", "office_cubicles", "conference_room",
        "waiting_room", "computer_room", "library/indoor", "classroom",
        "kindergarden_classroom", "art_studio", "lecture_room", "chemistry_lab",
        "physics_laboratory", "biology_laboratory", "art_school"
        ],
        "Remaining":
        [
        "clothing_store", "shoe_shop", "hardware_store", "beauty_salon",
        "bookstore", "department_store", "drugstore", "fabric_store",
        "gift_shop", "jewelry_shop", "general_store/indoor", "market/indoor",
        "bazaar/indoor", "toyshop", "butchers_shop", "pet_shop",
        "bathroom", "shower", "closet", "storage_room", "basement",
        "utility_room", "laundromat", "attic", "garage/indoor", "locker_room",
        "dressing_room", "sauna", "jacuzzi/indoor",
        "hospital_room", "operating_room", "veterinarians_office",
        "clean_room", "science_museum", "natural_history_museum",
        "museum/indoor", "engine_room", "server_room", "repair_shop",
        "elevator_lobby", "elevator/door", "staircase",
        "atrium/public", "television_studio",
        "arena/rodeo", "church/indoor", "train_interior", "bus_interior",
        "elevator_shaft", "jail_cell", "burial_chamber", "catacomb",
        "aquarium", "archive", "phone_booth", "bamboo_forest", "throne_room",
        "art_gallery", "artists_loft", "bank_vault", "bowling_alley", "gymnasium/indoor"
        ],
    }
    with open('class_groups.json', 'w') as f:
        json.dump(groups, f, indent=4)


def get_all_expert_lists(list_file, cache_file, output_dir, list_suffix):

    # Build full paths from file list
    with open(list_file, "r") as f:
        all_files = [
            os.path.join(root, line.strip().split()[0])
            for line in f if line.strip()
        ]


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



    # Load cached class-to-indices mapping
    with open(cache_file, "r") as f:
        class_to_indices = json.load(f)

    # Count how many samples per class
    class_counts = {cls: len(indices) for cls, indices in class_to_indices.items()}

    # Optional: sort by frequency
    sorted_class_counts = dict(sorted(class_counts.items(), key=lambda item: item[1], reverse=True))

    # Print
    print("\n=== Class Frequencies (from cache) ===")
    for cls, count in sorted_class_counts.items():
        print(f"{cls:25s}: {count}")

    group_counts = {}
    for group, cats in groups.items():
        group_counts[group] = sum(class_counts.get(c, 0) for c in cats)
    total_count = 0
    for group, count in group_counts.items():
        print(f"{group:20s}: {count}")
        total_count += count

    # All labels from the dataset (e.g., class_counts)
    all_labels_in_data = set(class_counts.keys())

    # All labels from your group definitions
    all_grouped_labels = [label for group in groups.values() for label in group]
    grouped_label_counts = Counter(all_grouped_labels)

    # Check for duplicates
    duplicates = [label for label, count in grouped_label_counts.items() if count > 1]
    if duplicates:
        print("❌ Duplicate labels in groups:")
        for dup in duplicates:
            print(f"  - {dup}")
    else:
        print("✅ No duplicates in group definitions.")

    # Check for unassigned labels
    grouped_label_set = set(all_grouped_labels)
    unassigned_labels = all_labels_in_data - grouped_label_set
    if unassigned_labels:
        print("\n⚠️ Labels in dataset but not assigned to any group:")
        for label in sorted(unassigned_labels):
            print(f"  - {label}")
    else:
        print("✅ All labels from dataset are assigned to a group.")

    # Optionally: check if there are labels assigned that don't exist in the dataset
    extra_labels = grouped_label_set - all_labels_in_data
    if extra_labels:
        print("\n⚠️ Labels in groups that do not exist in dataset:")
        for label in sorted(extra_labels):
            print(f"  - {label}")
    else:
        print("✅ All grouped labels exist in dataset.")




    # Save file lists per group
    os.makedirs(output_dir, exist_ok=True)

    for group_name, class_list in groups.items():
        output_file = os.path.join(output_dir, f"{group_name.replace(' ', '_').lower()}_{list_suffix}.txt")
        indices = []

        for cls in class_list:
            indices.extend(class_to_indices.get(cls, []))

        # Sort and write to file
        with open(output_file, "w") as f:
            for idx in sorted(indices):
                f.write(f"sample_{idx}_rgb.png sample_{idx}_depth.npy\n")

        print(f"✅ Saved {len(indices)} samples for group '{group_name}' to {output_file}")


get_all_expert_lists("src/data/train_list.txt", "train_class_to_indices.json", "src/data/category_lists", "train_list")
get_all_expert_lists("src/data/val_list.txt", "val_class_to_indices.json", "src/data/category_lists", "val_list")