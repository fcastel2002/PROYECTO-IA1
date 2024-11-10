import os
from pathlib import Path
import argparse

def get_image_files(directory):
    # Common image extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    return [f for f in os.listdir(directory) if f.lower().endswith(image_extensions)]

def rename_images(directory, base_name):
    # Get list of image files
    images = get_image_files(directory)
    
    # Sort files to ensure consistent ordering
    images.sort()
    
    # Rename each file
    for index, image in enumerate(images, start=1):
        # Get original file extension
        old_path = os.path.join(directory, image)
        extension = os.path.splitext(image)[1]
        
        # Create new filename
        new_name = f"{base_name}_{index}{extension}"
        new_path = os.path.join(directory, new_name)
        
        # Rename file
        os.rename(old_path, new_path)
        print(f"Renamed: {image} -> {new_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rename image files with base name and index')
    parser.add_argument('directory', help='Directory containing images')
    parser.add_argument('base_name', help='Base name for renamed files')
    
    args = parser.parse_args()
    
    rename_images(args.directory, args.base_name)