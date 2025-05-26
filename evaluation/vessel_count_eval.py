import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path

# Define the base directories for the four datasets
datasets = [
    "Original_1",
    "Augmented_2",
    "Mixed_3",
    "Synthetic_4"
]

base_dir = r"E:\Datasets\masati-thesis\results"

def count_vessels_in_label_file(label_path):
    """Count the number of vessels in a label file."""
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
            # Each line represents one vessel
            return len(lines)
    except Exception as e:
        print(f"Error reading {label_path}: {e}")
        return 0

def process_dataset(dataset_name):
    """Process one dataset across all 5 folds and count vessels."""
    dataset_path = os.path.join(base_dir, dataset_name)
    data_dir = os.path.join(dataset_path, "data")
    labels_dir = os.path.join(data_dir, "labels")
    
    # Dictionary to store vessel counts for each fold
    fold_counts = {}
    
    # Process each fold (0 to 4)
    for fold in range(5):
        fold_file = os.path.join(data_dir, f"fold_{fold}_train.txt")
        
        try:
            with open(fold_file, 'r') as f:
                image_paths = f.read().strip().split('\n')
            
            # Convert image paths to label paths and count vessels
            total_vessels = 0
            for img_path in image_paths:
                # Extract just the filename without extension
                img_filename = os.path.basename(img_path)
                img_name = os.path.splitext(img_filename)[0]
                
                # Construct the path to the label file
                label_path = os.path.join(labels_dir, f"{img_name}.txt")
                
                # Count vessels in this label file
                vessels = count_vessels_in_label_file(label_path)
                total_vessels += vessels
            
            fold_counts[fold] = total_vessels
            
        except Exception as e:
            print(f"Error processing fold {fold} for dataset {dataset_name}: {e}")
            fold_counts[fold] = 0
            
    return fold_counts

# Process all datasets
all_results = {}
for dataset in datasets:
    print(f"Processing dataset {dataset}...")
    vessel_counts = process_dataset(dataset)
    all_results[dataset] = vessel_counts
    print(f"Completed dataset {dataset}: {vessel_counts}")

# Create visualization
plt.figure(figsize=(14, 8))

# Bar properties
bar_width = 0.15
opacity = 0.8
index = np.arange(5)  # 5 folds

colors = ['skyblue', '#FFB347', 'lightcoral', 'lightgreen']

# Plot bars for each dataset
for i, (dataset, counts) in enumerate(all_results.items()):
    fold_numbers = sorted(counts.keys())
    vessel_counts = [counts[fold] for fold in fold_numbers]
    
    plt.bar(index + i*bar_width, 
            vessel_counts, 
            bar_width,
            alpha=opacity,
            color=colors[i],
            label=dataset)

# Add labels and title
plt.xlabel('Fold Number')
plt.ylabel('Number of Vessels')
plt.title('Number of Vessels per Dataset across 5 Folds')
plt.xticks(index + bar_width * 1.5, [f'Fold {i}' for i in range(5)])
plt.legend()

plt.tight_layout()
plt.savefig('vessel_count_analysis.png', dpi=300)
plt.show()

# Print numerical results
print("\nNumerical Results:")
print("-" * 50)
print(f"{'Dataset':<15} | {'Fold 0':<8} | {'Fold 1':<8} | {'Fold 2':<8} | {'Fold 3':<8} | {'Fold 4':<8} | {'Total':<8}")
print("-" * 50)

for dataset, counts in all_results.items():
    total = sum(counts.values())
    row = f"{dataset:<15} | "
    for fold in range(5):
        row += f"{counts.get(fold, 0):<8} | "
    row += f"{total:<8}"
    print(row)
