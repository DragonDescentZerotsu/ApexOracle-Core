import json
import os
from tqdm import tqdm
import pickle


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

output_txt_file_path_format = '/data/fangping/bulleye/Bullseye UPenn dict data/BE_{}_peptides_count_sum.pkl'

for category in categories:
    sum_result = []
    # Build the file paths for the input JSON and output text file
    complete_data_path = file_path_format.format(category)
    output_txt_file_path = output_txt_file_path_format.format(category)

    if os.path.exists(output_txt_file_path):
        print(f'{output_txt_file_path.split("/")[-1]} already exists, skipping...')
        continue

    # Read the JSON content
    print(f'Reading {complete_data_path.split("/")[-1]}...')
    complete_data = read_json_file(complete_data_path)

    for value in complete_data.values():
        sum_result.append(sum(value['Count']))

    # Write the keys to the output TXT file, one per line
    with open(output_txt_file_path, 'wb') as output_file:
        pickle.dump(sum_result, output_file)

