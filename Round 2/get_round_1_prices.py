

import os

script_dir = os.path.dirname(__file__)
log_file_path = os.path.join(script_dir, 'data', 'round1_final.log')
output_file_path = os.path.join(script_dir, 'data', 'output_round1')


def extract_prices_section(input_file, output_file):
    with open(input_file, 'r') as file:
        data = file.read()
    
    # Find the start of the 'Prices' section and the start of the 'Time' section
    start_index = data.find('Activities log:')
    end_index = data.find('Trade History:', start_index)
    
    # Extract the data between 'Prices' and 'Time'
    if start_index != -1 and end_index != -1:
        prices_data = data[start_index:end_index].strip()
    else:
        prices_data = "No 'Prices' or 'Time' section found."
    
    # Write the extracted data to a new file
    with open(output_file, 'w') as file:
        file.write(prices_data)

# Usage
extract_prices_section(log_file_path, output_file_path)
