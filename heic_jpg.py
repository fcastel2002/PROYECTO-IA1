import os
import logging
from PIL import Image, UnidentifiedImageError
from pillow_heif import register_heif_opener
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(message)s')


def convert_single_file(heic_path, jpg_path, output_quality):
    """
    Convert a single HEIC file to JPG format.

    Args:
        heic_path (str): Path to the HEIC file.
        jpg_path (str): Path to save the converted JPG file.
        output_quality (int): Quality of the output JPG image.
    """
    try:
        with Image.open(heic_path) as image:
            image.save(jpg_path, "JPEG", quality=output_quality)
        return heic_path, True  # Successful conversion
    except (UnidentifiedImageError, FileNotFoundError, OSError) as e:
        logging.error(f"Error converting '{heic_path}': {e}")
        return heic_path, False  # Failed conversion


def convert_heic_to_jpg(heic_dirs, output_quality=50, max_workers=4):
    """
    Converts HEIC images in specified directories to JPG format using parallel processing.

    Args:
        heic_dirs (list of str): List of paths to directories containing HEIC files.
        output_quality (int, optional): Quality of the output JPG images (1-100). Defaults to 50.
        max_workers (int, optional): Number of parallel threads. Defaults to 4.
    """
    register_heif_opener()

    for heic_dir in heic_dirs:
        if not os.path.isdir(heic_dir):
            logging.error(f"Directory '{heic_dir}' does not exist.")
            continue

        # Get all HEIC files in the specified directory
        heic_files = [file for file in os.listdir(heic_dir) if file.lower().endswith(".heic")]
        total_files = len(heic_files)

        if total_files == 0:
            logging.info(f"No HEIC files found in the directory '{heic_dir}'.")
            continue

        # Prepare file paths for conversion
        tasks = []
        for file_name in heic_files:
            heic_path = os.path.join(heic_dir, file_name)
            jpg_path = os.path.join(heic_dir, os.path.splitext(file_name)[0] + ".jpg")

            # Skip conversion if the JPG already exists
            if os.path.exists(jpg_path):
                logging.info(f"Skipping '{file_name}' as the JPG already exists.")
                continue

            tasks.append((heic_path, jpg_path))

        # Convert HEIC files to JPG in parallel using ThreadPoolExecutor
        num_converted = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(convert_single_file, heic_path, jpg_path, output_quality): heic_path
                for heic_path, jpg_path in tasks
            }

            for future in as_completed(future_to_file):
                heic_file = future_to_file[future]
                try:
                    _, success = future.result()
                    if success:
                        # Delete the HEIC file after successful conversion
                        os.remove(heic_file)
                        num_converted += 1
                    # Display progress
                    progress = int((num_converted / total_files) * 100)
                    print(f"Conversion progress in '{heic_dir}': {progress}%", end="\r", flush=True)
                except Exception as e:
                    logging.error(f"Error occurred during conversion of '{heic_file}': {e}")

        print(f"\nConversion completed in '{heic_dir}'. {num_converted} files converted.")


if __name__ == "__main__":
    # Directories to process
    directories = [
        "anexos/imagenes_mias/berenjena",
        "anexos/imagenes_mias/camote",
        "anexos/imagenes_mias/papa",
        "anexos/imagenes_mias/zanahoria"
    ]

    # Convert HEIC to JPG with parallel processing
    convert_heic_to_jpg(directories, output_quality=90, max_workers=8)
