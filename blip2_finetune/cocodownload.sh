
#!/bin/bash

# ========= SETTINGS =========
TARGET_DIR=~/datasets/coco
MIN_FREE_GB=25  # Minimum required free space
# ============================

echo "📁 Creating target directory at: $TARGET_DIR"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR" || { echo "❌ Failed to enter directory"; exit 1; }

# ⏱️ Check available disk space in GB
FREE_GB=$(df -BG "$TARGET_DIR" | awk 'NR==2 {print $4}' | sed 's/G//')

echo "💽 Available space in $TARGET_DIR: ${FREE_GB}GB"

if [ "$FREE_GB" -lt "$MIN_FREE_GB" ]; then
  echo "❌ Not enough space! At least ${MIN_FREE_GB}GB is required."
  exit 1
fi

echo "✅ Disk space is sufficient. Proceeding with download..."

# Download image zip files
echo "📥 Downloading COCO 2014 train and val images..."
wget -c http://images.cocodataset.org/zips/train2014.zip
wget -c http://images.cocodataset.org/zips/val2014.zip

# Download annotations
echo "📥 Downloading annotations..."
wget -c http://images.cocodataset.org/annotations/annotations_trainval2014.zip

# Unzip each into its own folder
echo "📂 Unzipping files..."
unzip -q train2014.zip   # → train2014/
unzip -q val2014.zip     # → val2014/
unzip -q annotations_trainval2014.zip  # → annotations/

# Optional cleanup
echo "🧹 Cleaning up zip files..."
rm -f train2014.zip val2014.zip annotations_trainval2014.zip

# Final directory structure preview
echo "📁 Final structure:"
ls -lh "$TARGET_DIR"

echo "🎉 COCO 2014 dataset setup complete!"
#-------------------------------------------------------