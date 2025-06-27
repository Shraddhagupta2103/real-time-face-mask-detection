import os

folders = [
    "dataset/data/with_mask",
    "dataset/data/without_mask",
    "model"
]

files = [
    "train_model.py",
    "test_model.py",
    "realtime_detection.py",
    "utils.py",
    "README.md",
    "requirements.txt"
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create empty files
for file in files:
    open(file, 'w').close()

print("✅ Subfolders and files created inside your current project folder!")
