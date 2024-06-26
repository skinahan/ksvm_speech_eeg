import os
import pandas as pd
import csv


def create_replacement_dictionary(csv_file_path):
    """
    Create a replacement dictionary based on a CSV file containing ANS and AWS mappings.

    Parameters:
    - csv_file_path: The path to the CSV file with the mapping data.

    Returns:
    - replacement_dict: A dictionary for replacement, with ANS as keys and ANS_New as values,
      and AWS as keys and AWS_New as values.
    """
    replacement_dict = {}

    try:
        with open(csv_file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            # Read the header to identify column positions
            header = next(csv_reader)
            ans_index = 0
            ans_new_index = 1
            aws_index = 2
            aws_new_index = 3

            for row in csv_reader:
                ans = row[ans_index]
                ans_new = row[ans_new_index]
                aws = row[aws_index]
                aws_new = row[aws_new_index]

                # Map ANS to ANS_New
                if ans and ans_new:
                    replacement_dict[ans] = ans_new

                # Map AWS to AWS_New
                if aws and aws_new:
                    replacement_dict[aws] = aws_new

    except FileNotFoundError:
        print(f"File '{csv_file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return replacement_dict


# Example usage:
csv_file_path = 'Subject_CodeMap.csv'  # Replace with the path to your CSV file
replacement_dict = create_replacement_dictionary(csv_file_path)


# The 'replacement_dict' can now be used with the 'update_column_in_dataframes' method


def update_column_in_dataframes(directory_path, column_name, replacement_dict):
    """
    Update a specified column in all DataFrames stored as .pkl files in a directory.

    Parameters:
    - directory_path: The path to the directory containing the .pkl files.
    - column_name: The name of the column to be updated.
    - replacement_dict: A dictionary mapping original values to replacement values.

    Example replacement_dict:
    {
        'SubjectA': 'ID001',
        'SubjectB': 'ID002',
        # Add more mappings as needed
    }
    """
    # Ensure the directory path exists
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' not found.")
        return

    # List all .pkl files in the directory
    pkl_files = [file for file in os.listdir(directory_path) if file.endswith('.pkl')]

    for pkl_file in pkl_files:
        file_path = os.path.join(directory_path, pkl_file)

        try:
            # Load the DataFrame from the .pkl file
            df = pd.read_pickle(file_path)

            # Update the specified column using the replacement_dict
            df[column_name] = df[column_name].replace(replacement_dict)

            # Filter rows where the "Subject" column starts with 'C' or 'S'
            df = df[~df['Subject'].str.startswith(('C', 'S'))]

            # Save the updated DataFrame back to the .pkl file
            df.to_pickle(file_path)

            print(f"Updated {column_name} column in '{pkl_file}'.")

        except Exception as e:
            print(f"An error occurred while processing '{pkl_file}': {e}")


# Example usage:
# directory_path = './pkl_data'  # Replace with the path to your directory containing .pkl files
# column_name = 'Subject'  # Replace with the name of the column you want to update
#
# update_column_in_dataframes(directory_path, column_name, replacement_dict)

from PIL import Image
import os
import re


def extract_subject_number(filename):
    # Extract the numeric part of the filename
    match = re.search(r'(\d+)', filename[2:])
    if match:
        return int(match.group(0))
    return 0


def merge_images(directory_path):
    # Create a merged directory if it doesn't exist
    merged_dir = os.path.join(directory_path, "merged")
    os.makedirs(merged_dir, exist_ok=True)

    # Collect files matching the naming convention
    ans_files = []
    aws_files = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".png"):
            if filename.startswith("42_ANS"):
                ans_files.append(filename)
            elif filename.startswith("42_AWS"):
                aws_files.append(filename)

    # Sort files based on the extracted numeric part
    ans_files.sort(key=extract_subject_number)
    aws_files.sort(key=extract_subject_number)

    # Merge ANS images
    for i in range(0, len(ans_files), 2):
        image1 = Image.open(os.path.join(directory_path, ans_files[i]))
        image2 = Image.open(os.path.join(directory_path, ans_files[i + 1] if i + 1 < len(ans_files) else ans_files[i]))
        merged_image = Image.new('RGB', (image1.width + image2.width, image1.height), (255, 255, 255))
        merged_image.paste(image1, (0, 0))
        merged_image.paste(image2, (image1.width, 0))

        # Set the DPI for the merged image to 300 DPI
        merged_image.info['dpi'] = (300, 300)

        merged_image.save(os.path.join(merged_dir, f"merged_{ans_files[i]}_{ans_files[i + 1]}"), dpi=(300, 300))

    # Merge AWS images
    for i in range(0, len(aws_files), 2):
        image1 = Image.open(os.path.join(directory_path, aws_files[i]))
        image2 = Image.open(os.path.join(directory_path, aws_files[i + 1] if i + 1 < len(aws_files) else aws_files[i]))
        merged_image = Image.new('RGB', (image1.width + image2.width, image1.height), (255, 255, 255))
        merged_image.paste(image1, (0, 0))
        merged_image.paste(image2, (image1.width, 0))

        # Set the DPI for the merged image to 300 DPI
        merged_image.info['dpi'] = (300, 300)

        merged_image.save(os.path.join(merged_dir, f"merged_{aws_files[i]}_{aws_files[i + 1]}"), dpi=(300, 300))


if __name__ == "__main__":
    directory_path = "results/figures/heatmaps"
    merge_images(directory_path)
