import json
import os
from tqdm import tqdm

def read_json_file(file_path):
    """
    Reads and parses a JSON file.

    :param file_path: Path to the JSON file.
    :return: Parsed JSON content as a Python dictionary.
    """
    with open(file_path, 'r') as file:
        return json.load(file)

file_path_format = '/data/fangping/bulleye/Bullseye UPenn dict data/BE_{}_merged_counts.DNA.json'

categories = ['cntrl', 'log', 'stat']

output_txt_file_path_format = '/data/fangping/bulleye/Bullseye UPenn dict data/BE_{}_peptides_seqs.txt'

for category in categories:
    # Build the file paths for the input JSON and output text file
    complete_data_path = file_path_format.format(category)
    output_txt_file_path = output_txt_file_path_format.format(category)

    if os.path.exists(output_txt_file_path):
        print(f'{output_txt_file_path.split("/")[-1]} already exists, skipping...')
        continue

    # Read the JSON content
    print(f'Reading {complete_data_path.split("/")[-1]}...')
    complete_data = read_json_file(complete_data_path)

    # Write the keys to the output TXT file, one per line
    with open(output_txt_file_path, 'w') as output_file:
        for key in tqdm(complete_data.keys(), desc=f"Writing {category} Peptides"):
            output_file.write(f"{key}\n")

    