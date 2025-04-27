import random


def split_train_val():
    import random

    # Read all lines from the original file
    with open('src/data/train_list_original.txt', 'r') as f:
        lines = [line.strip() for line in f if line.strip()]  # <<< STRIP + IGNORE EMPTY

    # Shuffle the lines
    random.seed(42)  # or any number you like
    random.shuffle(lines)

    # Calculate split index
    val_size = 0.2
    split_idx = int(len(lines) * (1 - val_size))

    # Split into train and validation
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    # Write to train_list.txt
    with open('src/data/train_list.txt', 'w') as f:
        f.write('\n'.join(train_lines) + '\n')  # <<< JOIN WITH NEWLINES PROPERLY

    # Write to val_list.txt
    with open('src/data/val_list.txt', 'w') as f:
        f.write('\n'.join(val_lines) + '\n')  # <<< SAME

    print(f"Done! {len(train_lines)} samples in train_list.txt and {len(val_lines)} samples in val_list.txt.")


def check_train_val_split():
    # Read the train and validation lists
    with open('src/data/train_list.txt', 'r') as f:
        train_lines = set(line.strip() for line in f if line.strip())

    with open('src/data/val_list.txt', 'r') as f:
        val_lines = set(line.strip() for line in f if line.strip())

    # Check for overlaps
    intersection = train_lines.intersection(val_lines)

    # Check for total count
    total_lines = train_lines.union(val_lines)

    with open('src/data/train_list_original.txt', 'r') as f:
        original_lines = set(line.strip() for line in f if line.strip())

    print(len(intersection))
    print(len(total_lines))
    print(len(original_lines))
    
    # Assertions
    assert len(intersection) == 0, f"Error: {len(intersection)} duplicate samples found between train and val!"
    assert total_lines == original_lines, "Error: Some samples are missing after the split!"

    print("✅ All good! Every sample is distinct and appears exactly once.")


def spit_category_lists():

    # First load the split master lists
    with open('src/data/train_list.txt', 'r') as f:
        global_train_set = set(line.strip() for line in f)

    with open('src/data/val_list.txt', 'r') as f:
        global_val_set = set(line.strip() for line in f)

    # List of your categories
    categories = ['kitchen', 'living_room', 'dorm_room', 'bathroom', 'home_office']

    for category in categories:
        # Load the original category-specific list
        with open(f'src/data/category_lists/{category}_train_list_original.txt', 'r') as f:
            category_samples = [line.strip() for line in f]

        # Split according to the global split
        category_train_new = []
        category_val_new = []

        for sample in category_samples:
            if sample in global_train_set:
                category_train_new.append(sample)
            elif sample in global_val_set:
                category_val_new.append(sample)
            else:
                print(f"⚠️ Warning: Sample '{sample}' not found in either train or val global lists!")

        # Save the new category-specific train and val lists
        with open(f'src/data/category_lists/{category}_train_list.txt', 'w') as f:
            f.write('\n'.join(category_train_new) + '\n')

        with open(f'src/data/category_lists/{category}_val_list.txt', 'w') as f:
            f.write('\n'.join(category_val_new) + '\n')

        print(f"✅ {category}: {len(category_train_new)} train samples, {len(category_val_new)} val samples.")

#split_train_val()
#check_train_val_split()
#spit_category_lists()