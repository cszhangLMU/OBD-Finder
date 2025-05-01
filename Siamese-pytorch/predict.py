import os
import shutil
import csv
from PIL import Image
from siamese import Siamese
from pathlib import Path
from collections import defaultdict

# Set environment variables and device
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def calculate_similarity(folder1_path, image_extension='.png'):
    """
    Calculate similarity and save it to each subfolder. Additionally, generate two CSV files:
    1. image_similarity_results.csv - Save the best match result for each Image_1.
    2. image_similarity_results1.csv - Save all match results.

    Subfolders will be copied to different result folders based on average similarity.

    :param folder1_path: The path to the parent folder containing multiple subfolders.
    :param image_extension: The file extension of the image files (default is '.png').
    """
    # Create result folders
    result_folders = {
        'A1': folder1_path / 'result_A1',
        'B1': folder1_path / 'result_B1',
        'C1': folder1_path / 'result_C1',
        'D1': folder1_path / 'result_D1',
    }

    # Ensure result_A1, result_B1, result_C1, result_D1 directories exist
    for folder in result_folders.values():
        if not folder.exists():
            folder.mkdir(parents=True)

    # Iterate through each subfolder in folder1_path
    for folder2 in folder1_path.iterdir():
        if folder2.is_dir():  # Ensure it's a directory
            input_csv = folder2 / 'matched_boxes.csv'  # Input CSV file path
            images_folder = folder2 / 'splits'  # Image folder path
            output_csv = folder2 / 'image_similarity_results.csv'  # Output CSV file path for best match results
            output_csv1 = folder2 / 'image_similarity_results1.csv'  # Output CSV file path for all match results

            if not input_csv.exists() or not images_folder.exists():
                print(f"Skipping folder {folder2} because required files are missing.")
                continue

            # Initialize Siamese model
            model = Siamese()

            total_similarity = 0
            valid_matches_count = 0

            # Store all matches for each Image_1
            image_matches = defaultdict(list)

            # Open the input CSV to read image filenames
            with open(input_csv, mode='r') as infile:
                reader = csv.reader(infile)
                header = next(reader, None)  # Skip the header (if present)

                # Read each image pair filenames
                for row in reader:
                    if len(row) < 2:
                        print(f"Invalid row in {input_csv}: {row} (Skipping)")
                        continue

                    image_1_filename = row[0].strip() + image_extension  # Add extension
                    image_2_filename = row[1].strip() + image_extension  # Add extension

                    # Concatenate the full path of the images based on filenames
                    image_1_path = images_folder / image_1_filename
                    image_2_path = images_folder / image_2_filename

                    try:
                        # Open images
                        image_1 = Image.open(image_1_path).convert('RGB')
                        image_2 = Image.open(image_2_path).convert('RGB')
                    except Exception as e:
                        print(f"Error opening images: {e} (Skipping pair {image_1_filename}, {image_2_filename})")
                        continue

                    # Calculate image similarity
                    probability = model.detect_image(image_1, image_2)

                    # Add the similarity for each match to the dictionary with Image_1 as key
                    image_matches[image_1_filename].append((image_2_filename, probability.item()))

            # Open the output CSV to save all similarity results
            with open(output_csv1, mode='w', newline='') as outfile1:
                writer1 = csv.writer(outfile1)
                writer1.writerow(["Image_1", "Image_2", "Similarity"])

                # Store all matching results for each Image_1
                for image_1_filename, matches in image_matches.items():
                    for match in matches:
                        image_2_filename, similarity = match
                        writer1.writerow([image_1_filename, image_2_filename, similarity])

                        total_similarity += similarity
                        valid_matches_count += 1

                        print(
                            f"Processed: {image_1_filename} -> Match: {image_2_filename} -> Similarity: {similarity:.4f}")

            # Open the output CSV to save the best match similarity results
            with open(output_csv, mode='w', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(["Image_1", "Image_2", "Similarity"])

                # For each Image_1, select the best matching Image_2
                for image_1_filename, matches in image_matches.items():
                    # If there's only one match, use it directly
                    if len(matches) == 1:
                        best_match = matches[0]
                    else:
                        # Sort matches by similarity in descending order
                        matches_sorted = sorted(matches, key=lambda x: x[1], reverse=True)
                        # Select the best match (highest similarity)
                        best_match = matches_sorted[0]

                    image_2_filename, best_similarity = best_match

                    # Save the best match
                    writer.writerow([image_1_filename, image_2_filename, best_similarity])
                    total_similarity += best_similarity
                    valid_matches_count += 1

                    print(
                        f"Processed Best Match: {image_1_filename} -> {image_2_filename} -> Similarity: {best_similarity:.4f}")

            # Calculate the average similarity: for all best matches, calculate the average similarity
            if valid_matches_count > 0:
                average_similarity = total_similarity / valid_matches_count
                print(f"Average similarity for {folder2.name}: {average_similarity:.4f}")
            else:
                average_similarity = 0
                print(f"No valid matches for {folder2.name}.")

            # Write the average similarity to the last row of the CSV
            with open(output_csv, mode='a', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(["Average Similarity", "", average_similarity])

            # Copy subfolder to the corresponding result folder based on the average similarity
            if average_similarity >= 0.9:
                # Copy the subfolder to result_A1
                destination = result_folders['A1'] / folder2.name
                print(f"Average similarity {average_similarity:.4f} -> Copying {folder2.name} to result_A1")
                shutil.copytree(folder2, destination)
            elif 0.8 <= average_similarity < 0.9:
                # Copy the subfolder to result_B1
                destination = result_folders['B1'] / folder2.name
                print(f"Average similarity {average_similarity:.4f} -> Copying {folder2.name} to result_B1")
                shutil.copytree(folder2, destination)
            elif 0.7 <= average_similarity < 0.8:
                # Copy the subfolder to result_C1
                destination = result_folders['C1'] / folder2.name
                print(f"Average similarity {average_similarity:.4f} -> Copying {folder2.name} to result_C1")
                shutil.copytree(folder2, destination)
            elif 0.6 <= average_similarity < 0.7:
                # Copy the subfolder to result_D1
                destination = result_folders['D1'] / folder2.name
                print(f"Average similarity {average_similarity:.4f} -> Copying {folder2.name} to result_D1")
                shutil.copytree(folder2, destination)
            else:
                # If the average similarity is below 0.6, no copying
                print(f"Average similarity {average_similarity:.4f} -> Not copying {folder2.name}")

            print(f"Similarity results saved to {output_csv} and {output_csv1}\n")


if __name__ == "__main__":
    # The input folder path, which contains multiple subfolders
    folder1_path = Path("/data/C1/all_folders30")  # Replace with the actual folder path

    # Call the similarity calculation function
    calculate_similarity(folder1_path, image_extension='.png')  # Assuming the image files are in .png format
